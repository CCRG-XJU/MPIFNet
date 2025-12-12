# Multi-Perspective Information Fusion Network for Remote Sensing Segmentation


## Datasets
  - [ISPRS Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) 
  - [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)

## Install
```
conda create -n ciap python=3.8
conda activate ciap
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r GeoSeg/requirements.txt

pip install causal  # https://github.com/Dao-AILab/causal-conv1d/releases
pip install mamba  # https://github.com/state-spaces/mamba/releases
```
## Data Preprocessing

Please follw the [GeoSeg](https://github.com/WangLibo1995/GeoSeg) to preprocess the LoveDA, Potsdam and Vaihingen dataset.

## Pretrained Model

The pretrained model adopted in our experiments is available at the following link: https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth.

## Training
"-c" means the path of the config, use different **config** to train different models.

```shell
python MPIFNet/train_supervision.py -c MPIFNet/config/potsdam/mpifnet.py
```

```shell
python MPIFNet/train_supervision.py -c MPIFNet/config/vaihingen/mpifnet.py
```

```shell
python MPIFNet/train_supervision.py -c MPIFNet/config/loveda/mpifnet.py
```
## Testing
**Vaihingen**
```shell
python MPIFNet/test_vaihingen.py -c SFCVPNet/config/vaihingen/mpifnet.py -o ~/fig_results/MPIFNet_vaihingen/ --rgb -t "d4"
```

**Potsdam**
```shell
python MPIFNet/test_potsdam.py -c SFCVPNet/config/potsdam/mpifnet.py -o ~/fig_results/MPIFNet_potsdam/ --rgb -t "d4"
```

**LoveDA** 

```shell
python MPIFNet/test_loveda.py -c SFCVPNet/config/loveda/mpifnet.py -o ~/fig_results/MPIFNet_loveda --rgb -t "d4"
```

## Acknowledgement

Many thanks the following projects's contributions to **SFCVPNet**.
- [GeoSeg](https://github.com/WangLibo1995/GeoSeg)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
