import math
from functools import partial
from typing import  Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from timm.models.layers import DropPath,trunc_normal_
from timm.models.registry import register_model
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except:
    pass

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,                    
        d_state=16,                 
        d_conv=3,                  
        expand=2,                 
        dt_rank="auto",            
        dt_min=0.001,              
        dt_max=0.1,         
        dt_init="random",           
        dt_scale=1.0,               
        dt_init_floor=1e-4,        
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
       
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand

        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
    
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,        
            bias=conv_bias,
            kernel_size=d_conv,        
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
    
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

   
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
  
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
     
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)  
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) 
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) 

        # (B, K, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):

        B, H, W, C = x.shape

        xz = self.in_proj(x) 
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)  d = C // 2

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)   

        y1, y2, y3, y4 = self.forward_core(x)   # forward_corev0
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, H, W, d)

        y = self.out_norm(y)  
        y = y * F.silu(z)
        out = self.out_proj(y)  
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        window_size=8,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        
    def forward(self, input: torch.Tensor):
        B, C, H, W = input.shape     # torch.Size([8, 192, 128, 128])
        
        x_reshaped = input.permute(0, 2, 3, 1).reshape(-1, C) 
         
        x_normed = self.ln_1(x_reshaped)  #torch.Size([131072, 192])
        x = x_normed.reshape(B, H, W, C).permute(0, 1, 2, 3)   # torch.Size([8, 128, 128, 192])
   
        x = self.drop_path(self.self_attention(x))
        x = x.permute(0,3,1,2)
        return x + input 
    
class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth=2, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,  
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,   
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,  
            )
            for i in range(depth)])
    
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x) 
        return x
    

config = {
    "tiny":[64, 128, 256, 512],
    "small": [96, 192, 384, 768],
    "base": [128, 256, 512, 1024]
}


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[64, 128, 256, 512],mode="tiny", **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"],strict=False)
    return model


@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768],**kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"],strict=False)
    return model


@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], mode = "base",**kwargs,)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"],strict=False)
    return model


class LRDU(nn.Module):
    """
    large receptive detailed upsample
    """
    def __init__(self,in_c,factor):
        super(LRDU,self).__init__()

        self.up_factor = factor
        self.factor1 = factor*factor//2
        self.factor2 = factor*factor
        self.up = nn.Sequential(
            nn.Conv2d(in_c, self.factor1*in_c, (1,7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1*in_c, self.factor2*in_c, (7,1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor),
            nn.Conv2d(in_c, in_c, 3 ,groups= in_c//4,padding=1), 
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()

        self.up = nn.Sequential(
            LRDU(ch_in,2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], drop_path_rate=0.1, 
                 layer_scale_init_value=1e-6):
        super().__init__()
        # 初始化为一个空的nn.ModuleList
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),    # 3, 96
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
      
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]   
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    

    def forward_features(self, x):
        stages_out = []
        for i in range(4):   # 0 - 3
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # x = self.sps[i](x)
            stages_out.append(x)
        return stages_out

    def forward(self, x):
        x = self.forward_features(x)
        return x


class GLmamba(nn.Module):

    def __init__(self,dim):

        super().__init__()
        self.dim = dim
        self.mamba = VSSLayer_up(dim)
        self.SelfAttention = SelfAttention(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        gmamba = self.mamba(x)
        mid_height = H // 2
        mid_width = W // 2

        upper_left = x[:, :, :mid_height, :mid_width]
        upper_right = x[:, :, :mid_height, mid_width:]
        lower_left = x[:, :, mid_height:, :mid_width]
        lower_right = x[:, :, mid_height:, mid_width:]

        upper_left = self.mamba(upper_left)
        upper_right = self.mamba(upper_right)
        lower_left = self.mamba(lower_left)
        lower_right = self.mamba(lower_right)

        upper_half = torch.cat((upper_left, upper_right), dim=3)
        lower_half = torch.cat((lower_left, lower_right), dim=3)

        lmamba = torch.cat((upper_half, lower_half), dim=2)
	    
        glmamba = lmamba + gmamba
        out = self.SelfAttention(x,lmamba,gmamba) + glmamba 
        return out

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.scale = dim ** 0.5 
       
    def forward(self, Q, K, V):
        B,C,H,W = Q.size() 
        N = W * H
        chunk_size = H # size

        Q = Q.permute(0, 3, 2, 1).reshape(B, N, C)  # torch.Size([8, 4096, 384])
        K = K.permute(0, 3, 2, 1).reshape(B, N, C)
        V = V.permute(0, 3, 2, 1).reshape(B, N, C)  

        output = torch.zeros(B, N, C, device=Q.device)
 
        for i in range(0, N, chunk_size):
            start = i
            end = min(i + chunk_size, N)
            
            Q_chunk = Q[:, start:end]   # torch.Size([8, 1024, 384])
            K_chunk = K[:, start:end]
            V_chunk = V[:, start:end]
            
            attn_scores_chunk = torch.matmul(Q_chunk, K_chunk.transpose(-2, -1)) / self.scale  # torch.Size([8, 1024, 1024])
            attn_weights_chunk = F.softmax(attn_scores_chunk, dim=-1)  # torch.Size([8, 1024, 1024])
            out_chunk = torch.matmul(attn_weights_chunk, V_chunk)  # torch.Size([8, 1024, 384])
          
            output[:, start:end] = out_chunk   # torch.Size([8, 4096, 384])
    
        out = output.view(B, C, H, W)  
        return out

from pytorch_wavelets import DWTForward

class skip(nn.Module):
    # dim1 = 2*dim2
    def __init__(self, dim1,dim2):    
        super(skip, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.conv = nn.Conv2d(dim1,dim2,kernel_size=1)

        self.conv_bn_relu = nn.Sequential(            
            nn.Conv2d(4*dim2, dim2, kernel_size=1, stride=1),
            nn.BatchNorm2d(dim2),
            nn.ReLU(inplace=True),
            )
       
    def forward(self, x1,x2):

        x1 = self.conv(x1)  #torch.Size([8, 384, 32, 32])
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)  #torch.Size([8, 384, 64, 64])
        
        out1 = x1 + x2

        yL1,yH1= self.wt(out1)
        y_HL1 = yH1[0][:,:,0,::]
        y_LH1 = yH1[0][:,:,1,::]
        y_HH1 = yH1[0][:,:,2,::]

        out2 = torch.cat([x1,x2],dim=1)   #torch.Size([8, 768, 64, 64])
        out2 = self.conv(out2)   #torch.Size([8, 384, 64, 64])

        yL2,yH2 = self.wt(out2)
        y_HL2 = yH2[0][:,:,0,::]
        y_LH2 = yH2[0][:,:,1,::]
        y_HH2 = yH2[0][:,:,2,::]

        yL = yL1 + yL2
        y_HL = y_HL1 + y_HL2    
        y_LH = y_LH1 + y_LH2
        y_HH = y_HH1 + y_HH2
        
        out = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv_bn_relu(out)
        out = out + x2
    
        return out

class Model(nn.Module):
    def __init__(self, n_class=6, pretrained = True):
        super(Model, self).__init__()
        self.n_class = n_class
        self.in_channel = 3
        config=[96, 192, 384, 768] # channles of convnext-small
        self.backbone = convnext_small(pretrained,True)

        self.glmamba = nn.ModuleList([
            GLmamba(config[2]),  
            GLmamba(config[1]),
            GLmamba(config[0])]          
        )
        self.skip = nn.ModuleList([
            skip(config[3],config[2]),
            skip(config[2],config[1]),
            skip(config[1],config[0])]    
        )

        self.Up5 = up_conv(ch_in=config[3], ch_out=config[3]//2)
        self.Up_conv5 = conv_block(ch_in=config[3], ch_out=config[3]//2)

        self.Up4 = up_conv(ch_in=config[2], ch_out=config[2]//2)
        self.Up_conv4 = conv_block(ch_in=config[2], ch_out=config[2]//2)

        self.Up3 = up_conv(ch_in=config[1], ch_out=config[1]//2)
        self.Up_conv3 = conv_block(ch_in=config[1], ch_out=config[1]//2)

        self.Up4x = LRDU(config[0],4)      
        self.convout = nn.Conv2d(config[0], n_class, kernel_size=1, stride=1, padding=0)                
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # print(x.shape)  #torch.Size([8, 3, 1024, 1024])

        x128,x64,x32,x16 = self.backbone(x)

        # print(x16.shape)    #torch.Size([8, 768, 32, 32])
        # print(x32.shape)    #torch.Size([8, 384, 64, 64])
        # print(x64.shape)    #torch.Size([8, 192, 128, 128])
        # print(x128.shape)   #torch.Size([8, 96, 256, 256])

        x32_1 = self.skip[0](x16,x32)  # torch.Size([8, 384, 64, 64])
        x64_1 = self.skip[1](x32,x64)
        x128_1 = self.skip[2](x64,x128)
     
        d32 = self.Up5(x16)     #torch.Size([8, 384, 64, 64])
        d32 = torch.cat([x32_1,d32],dim=1) # 768
        d32 = self.Up_conv5(d32) # 384
        d32 = self.glmamba[0](d32)

        d64 = self.Up4(d32)     #torch.Size([8, 192, 128, 128])
        d64 = torch.cat([x64_1,d64],dim=1)  
        d64 = self.Up_conv4(d64) #192
        d64 = self.glmamba[1](d64)


        d128 = self.Up3(d64)    #torch.Size([8, 96, 256, 256])
        d128 = torch.cat([x128_1,d128],dim=1)
        d128 = self.Up_conv3(d128)
        d128 = self.glmamba[2](d128)

        d512 = self.Up4x(d128)   # torch.Size([8, 96, 1024, 1024])
        out = self.convout(d512)   #torch.Size([8, 6, 1024, 1024])
  
        return out

