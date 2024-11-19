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
    else:
        model = DummyNet()
        model.name = 'dummy'
    return model
