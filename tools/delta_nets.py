import torch
from net.delta.ann_hist.ann_hist import ANN_HIST


def get_delta_net(device):
    delta_net = ANN_HIST(len_op=7)
    delta_net.load_state_dict(torch.load('model/delta/delta_ann.pt'))
    delta_net.name = 'DeltaNet-ANN'
    delta_net.classes = ['benign', 'hsja', 'nes', 'simba', 'sign-opt', 'sign-flip', 'ba']
    delta_net.to(device)
    return delta_net
