
# LightCL

## Requirements
- Python==3.11.4
- numpy==1.24.3
- torch==2.1.2
- torchvision==0.16.2
- tqdm==4.65.0, etc(which can be easily installed with pip)

## Usage

Run `LightCL.py` with the commands like below:
- Default
```
python LightCL.py 
```
- With sparse
```
python LightCL.py --Sparse
```

Here the major parameters are:
- `lr`: learning rate (default=0.01)
- `Beta`: hyperparameter for regulation loss (default=0.0002)
- `BufferNum`: number of Memory Buffer (default=15)
- `Ratio`: selecting vital feature map with `Ratio` (default=0.15)
- `Seed`: the random seed (default=0)
- `pretrain`: whether use the pre-trained model (default=True)
- `Dataset`: dataset (default: CIFAR10; Other: TinyImageNet)
- `Sparse`: whether sparse (default=False)

Note that we already have the pre-trained model (`ResNet18_for_LightCL.pth`) in the directory. If you want to pre-train the model yourself, you can download dataset `ImageNet32x32` under `data/` and run code `get_parameter.py`.