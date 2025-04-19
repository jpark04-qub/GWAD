# GWAD
# Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis
Pytorch code for CVPR 2025 paper Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis. This repository contains the demo code for reproducing the experimental results of GWAD and GWAD+. 

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

```
GWAD_defence starts
GWAD_defence : first detection [hsja] made @ 260th query  
GWAD_defence : delta-net predicted 9745 HoDS during 10004 queries
GWAD_defence : accumulated predictions 9745/9745
benign[0], hsja[9745], nes[0], simba[0], sign-opt[0], sign-flip[0], ba[0], --> Attack classes predicted by GWAD 
Attack stats : num - time [true] : [adv][val/suc, dist, ratio, [i1] [i2] [i3]]
1 - 81.777s [3] : [2][1/1, 1.728 0.041, [10004] [11] [0]]
average attack queries - 10004
GWAD screen : [screened dummy] [screened attack] [passed dummy] [passed attack]
[0] [2] [0] [10002]
```

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
