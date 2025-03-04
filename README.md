# GWAD
# GWAD : Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis

# Dependencies
- pytorch
- numpy
- skimage
- tqdm (optional)
- json
- argparse

# Pretrained Model
networks and their weights can be downloaded from (https://github.com/huyvnphan/PyTorch_CIFAR10?tab=readme-ov-file)

# Attack Classifier (Delta-Net)
- Network and pretrained model (weights) are included in net/delta and model/delta

# Setup 
- Create "data" and "model/cifar10" directories for dataset and weights respectively
- Download weights resnet18.pt (refer to utilities/load_model to use other models and weigths))
- Download cifar10 dataset (CIFAR-10 python version from https://www.cs.toronto.edu/~kriz/cifar.html)

# Run
Run by invoking `demo.py` as follows:

`python demo.py --data cifar10 --attack [attack method] --scenario [attack scenario]`

Arguments 
- attack method   : hsja, nes, sign-flip
- attack scenario : benign, standard, batch

Example for standard HSJA attack: 

`python demo.py --data cifar10 --attack hsja --scenario standard`

# Citation
If you use GWAD for your research, please cite the paper:
```
@Article{GWAD,
  author  = {J. Park and N. McLaughlin and I. Alouani},  
  journal = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  title   = {Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis},
  year    = {2025},
}
```
