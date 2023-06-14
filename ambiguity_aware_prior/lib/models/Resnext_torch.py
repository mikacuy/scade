#!/usr/bin/env python
# coding: utf-8

"""
https://github.com/CSAILVision/semantic-segmentation-pytorch
"""
import os
import sys
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from lib.configs.config import cfg


try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

__all__ = ['resnext101_32x8d']


model_urls = {
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    #'resnext101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnext101-imagenet.pth'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        features = []
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)

        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return features

    def forward(self, x):
        return self._forward_impl(x)

#### For AdaIN for cIMLE ####
class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_size, output_size, gain=2 ** 0.5, use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)

class StyleMod(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = EqualizedLinear(latent_size,
                                   channels * 2,
                                   gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]

        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


### Rewriting AdaIn layer
class AdaIn(nn.Module):
    def __init__(self, latent_size, out_channels):
        super(AdaIn, self).__init__()
        # self.mlp = EqualizedLinear(latent_size,
        #                            channels * 2,
        #                            gain=1.0, use_wscale=use_wscale)

        self.mlp = nn.Sequential(
                nn.Linear(latent_size, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, out_channels * 2))

    def forward(self, x, latent, mean_shift=0.0, var_shift=0.0, return_scale=False):
        style = self.mlp(latent)  # style => [batch_size, n_channels*2]


        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]

        shape2 = [1, x.size(1)] + (x.dim() - 2) * [1]
        mean_shift = mean_shift.view(shape2)
        var_shift = var_shift.view(shape2)

        # print(style[:, 1].shape)
        # print(mean_shift.shape)
        # print(style[:, 0].shape)
        # print(var_shift.shape)

        mean = style[:, 1] - mean_shift.cuda()
        var = style[:, 0] + 1. - var_shift.cuda()

        # print()
        # print(mean.shape)
        # print(var.shape)
        # print(x.shape)

        x = x * (var) + mean

        if return_scale:
            return x, var

        return x  

#############################

### With AdaIn v2
class ResNet_cIMLE(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, d_latent=512):
        super(ResNet_cIMLE, self).__init__()

        print("Using version 2 of AdaIn layers...")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        ## Noise1
        self.d_latent = d_latent
        self.style_mod0 = AdaIn(d_latent, self.inplanes)
        self.style_mod0_meanshift = torch.zeros(self.inplanes)
        self.style_mod0_varshift = torch.zeros(self.inplanes)


        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])
        self.style_mod1 = AdaIn(d_latent, 256)
        self.style_mod1_meanshift = torch.zeros(256)
        self.style_mod1_varshift = torch.zeros(256)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.style_mod2 = AdaIn(d_latent, 512)
        self.style_mod2_meanshift = torch.zeros(512)
        self.style_mod2_varshift = torch.zeros(512)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.style_mod3 = AdaIn(d_latent, 1024)
        self.style_mod3_meanshift = torch.zeros(1024)
        self.style_mod3_varshift = torch.zeros(1024)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, z):
        # See note [TorchScript super()]

        features = []
        x = self.conv1(x)
        x = self.style_mod0(x, z, self.style_mod0_meanshift, self.style_mod0_varshift)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.style_mod1(x, z, self.style_mod1_meanshift, self.style_mod1_varshift)

        features.append(x)

        x = self.layer2(x)     
        x = self.style_mod2(x, z, self.style_mod2_meanshift, self.style_mod2_varshift)

        features.append(x)

        x = self.layer3(x)      
        x = self.style_mod3(x, z, self.style_mod3_meanshift, self.style_mod3_varshift)

        features.append(x)

        x = self.layer4(x)

        features.append(x)

        return features

    def forward(self, x, z):
        return self._forward_impl(x, z)

    def set_mean_var_shifts(self, mean0, var0, mean1, var1, mean2, var2, mean3, var3):
        
        self.style_mod0_meanshift = mean0
        self.style_mod0_varshift = var0
        self.style_mod1_meanshift = mean1
        self.style_mod1_varshift = var1
        self.style_mod2_meanshift = mean2
        self.style_mod2_varshift = var2
        self.style_mod3_meanshift = mean3
        self.style_mod3_varshift = var3

        return

    def get_adain_init_act(self, x, z):

        x = self.conv1(x)
        x = self.style_mod0(x, z, self.style_mod0_meanshift, self.style_mod0_varshift)
        adain0 = x
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.style_mod1(x, z, self.style_mod1_meanshift, self.style_mod1_varshift)
        adain1 = x

        x = self.layer2(x)     
        x = self.style_mod2(x, z, self.style_mod2_meanshift, self.style_mod2_varshift)
        adain2 = x

        x = self.layer3(x)      
        x = self.style_mod3(x, z, self.style_mod3_meanshift, self.style_mod3_varshift)
        adain3 = x

        return adain0, adain1, adain2, adain3

    # def get_adain_init_act(self, x, z):

    #     ### Debugging function used to find the statistics of the scales
    #     ### Do not use for training

    #     x = self.conv1(x)
    #     x, var0 = self.style_mod0(x, z, self.style_mod0_meanshift, self.style_mod0_varshift, return_scale=True)
    #     adain0 = var0
        
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     x, var1 = self.style_mod1(x, z, self.style_mod1_meanshift, self.style_mod1_varshift, return_scale=True)
    #     adain1 = var1

    #     x = self.layer2(x)     
    #     x, var2 = self.style_mod2(x, z, self.style_mod2_meanshift, self.style_mod2_varshift, return_scale=True)
    #     adain2 = var2

    #     x = self.layer3(x)      
    #     x, var3 = self.style_mod3(x, z, self.style_mod3_meanshift, self.style_mod3_varshift, return_scale=True)
    #     adain3 = var3

    #     return adain0, adain1, adain2, adain3


class AdaIn_v2(nn.Module):
    def __init__(self, latent_size, out_channels):
        super(AdaIn_v2, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3+latent_size, out_channels=32, kernel_size=3, padding=1, stride=1, bias=True),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=3, stride=4, padding=1),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=1, bias=True),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=3, stride=4, padding=1),
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1, stride=1, bias=True),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=3, stride=4, padding=1))
                # nn.Linear(128, out_channels * 2))

        ### Flatten the image
        self.mlp = nn.Sequential(
                nn.Linear(8*7*7, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 128),
                nn.LeakyReLU(),
                nn.Linear(128, out_channels * 2))

    def forward(self, x, latent, input_img, mean_shift=0.0, var_shift=0.0):
        ## x: input feature
        ## latent: random code
        ## input_img: conditioned image

        B, C, H, W = input_img.shape

        latent = latent.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        input_concat = torch.cat((input_img, latent), dim=1)

        style = self.conv(input_concat)  # style => [batch_size, n_channels*2]

        style = style.view(B, -1)
        style = self.mlp(style)

        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]

        shape2 = [1, x.size(1)] + (x.dim() - 2) * [1]
        mean_shift = mean_shift.view(shape2)
        var_shift = var_shift.view(shape2)

        mean = style[:, 1] - mean_shift.cuda()
        var = style[:, 0] + 1. - var_shift.cuda()

        x = x * (var) + mean
        return x  


### With AdaIn v3
class ResNet_cIMLE_v3(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, d_latent=512):
        super(ResNet_cIMLE_v3, self).__init__()

        print("Using version 3 of AdaIn layers...")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        ## Noise1
        self.d_latent = d_latent
        self.style_mod0 = AdaIn_v2(d_latent, self.inplanes)
        self.style_mod0_meanshift = torch.zeros(self.inplanes)
        self.style_mod0_varshift = torch.zeros(self.inplanes)


        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])
        self.style_mod1 = AdaIn_v2(d_latent, 256)
        self.style_mod1_meanshift = torch.zeros(256)
        self.style_mod1_varshift = torch.zeros(256)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.style_mod2 = AdaIn_v2(d_latent, 512)
        self.style_mod2_meanshift = torch.zeros(512)
        self.style_mod2_varshift = torch.zeros(512)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.style_mod3 = AdaIn_v2(d_latent, 1024)
        self.style_mod3_meanshift = torch.zeros(1024)
        self.style_mod3_varshift = torch.zeros(1024)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, z):
        # See note [TorchScript super()]
        input_img = x

        features = []
        x = self.conv1(x)
        x = self.style_mod0(x, z, input_img, self.style_mod0_meanshift, self.style_mod0_varshift)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.style_mod1(x, z, input_img, self.style_mod1_meanshift, self.style_mod1_varshift)

        features.append(x)

        x = self.layer2(x)     
        x = self.style_mod2(x, z, input_img, self.style_mod2_meanshift, self.style_mod2_varshift)

        features.append(x)

        x = self.layer3(x)      
        x = self.style_mod3(x, z, input_img, self.style_mod3_meanshift, self.style_mod3_varshift)

        features.append(x)

        x = self.layer4(x)

        features.append(x)

        return features

    def forward(self, x, z):
        return self._forward_impl(x, z)

    def set_mean_var_shifts(self, mean0, var0, mean1, var1, mean2, var2, mean3, var3):
        
        self.style_mod0_meanshift = mean0
        self.style_mod0_varshift = var0
        self.style_mod1_meanshift = mean1
        self.style_mod1_varshift = var1
        self.style_mod2_meanshift = mean2
        self.style_mod2_varshift = var2
        self.style_mod3_meanshift = mean3
        self.style_mod3_varshift = var3

        return

    def get_adain_init_act(self, x, z):
        
        input_img = x

        x = self.conv1(x)
        x = self.style_mod0(x, z, input_img, self.style_mod0_meanshift, self.style_mod0_varshift)
        adain0 = x
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.style_mod1(x, z, input_img, self.style_mod1_meanshift, self.style_mod1_varshift)
        adain1 = x

        x = self.layer2(x)     
        x = self.style_mod2(x, z, input_img, self.style_mod2_meanshift, self.style_mod2_varshift)
        adain2 = x

        x = self.layer3(x)      
        x = self.style_mod3(x, z, input_img, self.style_mod3_meanshift, self.style_mod3_varshift)
        adain3 = x

        return adain0, adain1, adain2, adain3

### With AdaIn v1
# class ResNet_cIMLE(nn.Module):

#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None, d_latent=512):
#         super(ResNet_cIMLE, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer

#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)

#         ## Noise1
#         self.d_latent = d_latent
#         self.style_mod0 = StyleMod(d_latent, self.inplanes, use_wscale=False)


#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.style_mod1 = StyleMod(d_latent, 256, use_wscale=False)

#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         self.style_mod2 = StyleMod(d_latent, 512, use_wscale=False)

#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1])
#         self.style_mod3 = StyleMod(d_latent, 1024, use_wscale=False)

#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
#                                        dilate=replace_stride_with_dilation[2])
#         #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         #self.fc = nn.Linear(512 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)

#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))

#         return nn.Sequential(*layers)

#     def _forward_impl(self, x, z):
#         # See note [TorchScript super()]
#         features = []
#         x = self.conv1(x)
#         x = self.style_mod0(x, z)

#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.style_mod1(x, z)

#         features.append(x)

#         x = self.layer2(x)     
#         x = self.style_mod2(x, z)

#         features.append(x)

#         x = self.layer3(x)      
#         x = self.style_mod3(x, z)

#         features.append(x)

#         x = self.layer4(x)
        
#         features.append(x)

#         #x = self.avgpool(x)
#         #x = torch.flatten(x, 1)
#         #x = self.fc(x)

#         return features

#     def forward(self, x, z):
#         return self._forward_impl(x, z)


def resnext101_32x8d(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnext101_32x8d'], cfg.ROOT_DIR + '/' + cfg.MODEL.MODEL_REPOSITORY)
        #pretrained_model = torchvision.models.resnet152(pretrained=True)
        #pretrained_model = gcv.models.resnet152(pretrained=True)
        #pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnext101_32x8d_cIMLE(pretrained=True, d_latent=512, version="v2", **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8

    print("In resnext model")
    print(pretrained)

    if version == "v2":
        model = ResNet_cIMLE(Bottleneck, [3, 4, 23, 3], d_latent=d_latent, **kwargs)
    elif version == "v3":
        model = ResNet_cIMLE_v3(Bottleneck, [3, 4, 23, 3], d_latent=d_latent, **kwargs)

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnext101_32x8d'], cfg.ROOT_DIR + '/' + cfg.MODEL.MODEL_REPOSITORY)
        #pretrained_model = torchvision.models.resnet152(pretrained=True)
        #pretrained_model = gcv.models.resnet152(pretrained=True)
        #pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    import torch
    model = resnext101_32x8d(True).cuda()

    rgb = torch.rand((2, 3, 256, 256)).cuda()
    out = model(rgb)
    print(len(out))



"""

def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class GroupBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, groups=1, downsample=None):
        super(GroupBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, groups=32, num_classes=1000):
        self.inplanes = 128
        super(ResNeXt, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], groups=groups)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, groups=groups)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2, groups=groups)

        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(1024 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, groups, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features

def resnext101(pretrained=True, **kwargs):
    '''
    Constructs a resnext-101 model.
    #Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    '''
    model = ResNeXt(GroupBottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(), strict=False)
        #model.load_state_dict(torch.load('./pretrained/resnet101-imagenet.pth', map_location=None), strict=False)
    return model
"""
