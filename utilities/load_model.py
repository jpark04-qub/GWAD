import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()

    def forward(self, x):
        return 0


def get_models(data_type, model_name):
    if data_type == 'cifar10':
        if model_name == 'resnet18':
            from net.cifar10.resnet import resnet18
            model = resnet18()
            model.load_state_dict(torch.load('model/cifar10/resnet18.pt'))
            model.name = 'resnet-18'
            model.loss = 'cross entropy'
        elif model_name == 'mobilenet':
            from net.cifar10.mobilenetv2 import MobileNetV2
            model = MobileNetV2()
            model.load_state_dict(torch.load('model/cifar10/mobilenet_v2.pt'))
            model.name = 'mobilenet-v2'
            model.loss = 'cross entropy'
        else:
            raise Exception("Invalid model name")
    elif data_type == 'tiny_imagenet':
        if model_name == 'efficientnet':
            from net.tiny_imagenet.efficientnet.efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=200)
            model.load_state_dict(torch.load('model/tiny_imagenet/efficientnet_20e.pt'))
            model.name = 'efficientnet'
            model.loss = 'cross entropy'
        else:
            raise Exception("Invalid model name")
    elif data_type == 'imagenet':
        if model_name == 'resnet18':
            from net.imagenet.resnet.resnet18 import ResNet18
            model = ResNet18()
            model.load_state_dict(torch.load('model/imagenet/resnet18.pt'))
            model.name = 'resnet-18'
            model.loss = 'cross entropy'
            #model.baseline_state = np.load('model/imagenet/baseline/resnet18.npy')
        elif model_name == 'vgg16':
            from net.imagenet.vgg.vgg16 import VGG16
            model = VGG16()
            model.load_state_dict(torch.load('model/imagenet/vgg16.pt'))
            model.name = 'vgg-16'
            model.loss = 'cross entropy'
            #model.baseline_state = np.load('model/imagenet/baseline/vgg16.npy')
        elif model_name == 'efficientnet':
            from net.imagenet.efficientnet.efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=1000)
            model.load_state_dict(torch.load('model/imagenet/efficientnet-b3.pt'))
            model.name = 'efficientnet'
            model.loss = 'cross entropy'
        else:
            raise Exception("Invalid model name")
    else:
        model = DummyNet()
        model.name = 'dummy'
    return model
