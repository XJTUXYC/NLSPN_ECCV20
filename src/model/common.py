"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    Some of useful functions are defined here.
"""


import torch
import torch.nn as nn
import torchvision


model_path = {
    'resnet18': 'pretrained/resnet18.pth',
    'resnet34': 'pretrained/resnet34.pth'
}


def get_resnet18(pretrained=True):
    net = torchvision.models.resnet18(pretrained=False)
    if pretrained:
        state_dict = torch.load(model_path['resnet18'])
        net.load_state_dict(state_dict)

    return net


def get_resnet34(pretrained=True):
    net = torchvision.models.resnet34(pretrained=False)
    if pretrained:
        state_dict = torch.load(model_path['resnet34'])
        net.load_state_dict(state_dict)

    return net


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True, zero_init=False):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    
    conv = nn.Conv2d(ch_in, ch_out, kernel, stride, padding, bias=not bn)
    if zero_init:
        conv.weight.data.zero_()
        if not bn:
            conv.bias.data.zero_()
    layers.append(conv)
    
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  bn=True, relu=True, zero_init=False):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    
    convt = nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding, output_padding, bias=not bn)
    if zero_init:
        convt.weight.data.zero_()
        if not bn:
            convt.bias.data.zero_()
    layers.append(convt)
    
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


