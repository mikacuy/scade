#!/usr/bin/env python2
# coding: utf-8

"""
This network structure follows Ke Xian's work, "Structure-guided Ranking Loss for Single Image Depth Prediction".
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from lib.configs.config import cfg
from lib.models import Resnet, Resnext_torch


def resnet18_stride32(cIMLE=False, d_latent=512, version="v2"):
    if not cIMLE:
        return DepthNet(backbone='resnet', depth=18, upfactors=[2, 2, 2, 2])
    else:
        return DepthNet_cIMLE(backbone='resnet', depth=18, upfactors=[2, 2, 2, 2], d_latent=d_latent, version=version)

def resnet34_stride32(cIMLE=False, d_latent=512, version="v2"):
    if not cIMLE:
        return DepthNet(backbone='resnet', depth=34, upfactors=[2, 2, 2, 2])
    else:
        return DepthNet_cIMLE(backbone='resnet', depth=34, upfactors=[2, 2, 2, 2], d_latent=d_latent, version=version)

def resnet50_stride32(cIMLE=False, d_latent=512, version="v2"):
    if not cIMLE:
        return DepthNet(backbone='resnet', depth=50, upfactors=[2, 2, 2, 2])
    else:
        return DepthNet_cIMLE(backbone='resnet', depth=50, upfactors=[2, 2, 2, 2], d_latent=d_latent, version=version)

def resnet101_stride32(cIMLE=False, d_latent=512, version="v2"):
    if not cIMLE:
        return DepthNet(backbone='resnet', depth=101, upfactors=[2, 2, 2, 2])
    else:
        return DepthNet_cIMLE(backbone='resnet', depth=101, upfactors=[2, 2, 2, 2], d_latent=d_latent, version=version)

def resnet152_stride32(cIMLE=False, d_latent=512, version="v2"):
    if not cIMLE:
        return DepthNet(backbone='resnet', depth=152, upfactors=[2, 2, 2, 2])
    else:
        return DepthNet_cIMLE(backbone='resnet', depth=152, upfactors=[2, 2, 2, 2], d_latent=d_latent, version=version)

def resnext101_stride32x8d(cIMLE=False, d_latent=512, version="v2"):
    if not cIMLE:
        return DepthNet(backbone='resnext101_32x8d', depth=101, upfactors=[2, 2, 2, 2])
    else:
        return DepthNet_cIMLE(backbone='resnext101_32x8d', depth=101, upfactors=[2, 2, 2, 2], d_latent=d_latent, version=version)

def mobilenetv2(cIMLE=False, version="v2"):
    if not cIMLE:
        return DepthNet(backbone='mobilenetv2', depth=00, upfactors=[2, 2, 2, 2])
    else:
        return DepthNet_cIMLE(backbone='mobilenetv2', depth=00, upfactors=[2, 2, 2, 2], d_latent=d_latent, version=version)

class AuxiBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.conv1 = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, dilation=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.dim_out, momentum=0.5)
        self.bn2 = nn.BatchNorm2d(self.dim_out, momentum=0.5)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, top, lateral):
        if lateral.shape[2] != top.shape[2]:
            h, w = lateral.size(2), lateral.size(3)
            top = F.interpolate(input=top, size=(h, w), mode='bilinear',align_corners=True)
        out = torch.cat((lateral, top), dim=1)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out

class AuxiNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.inchannels = cfg.MODEL.RESNET_BOTTLENECK_DIM[1:]  # [256, 512, 1024, 2048]
        self.midchannels = cfg.MODEL.LATERAL_OUT[::-1]  # [256, 256, 256, 512]

        self.auxi_block1 = AuxiBlock(self.midchannels[2]+self.midchannels[3], 128)
        self.auxi_block2 = AuxiBlock(128 + self.midchannels[2], 128)
        self.auxi_block3 = AuxiBlock(128 + self.midchannels[2], 128)
        self.auxi_block4 = AuxiBlock(128 + self.midchannels[1], 128)
        self.auxi_block5 = AuxiBlock(128 + self.midchannels[0], 128)
        self.out_conv = AO(128, 1, 2)
        self._init_weights()

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        def init_model_weight(m):
            for child_m in m.children():
                if not isinstance(child_m, nn.ModuleList):
                    child_m.apply(init_func)

        init_model_weight(self)

    def forward(self, auxi_in):
        out = self.auxi_block1(auxi_in[0], auxi_in[1])  # 1/32
        out = self.auxi_block2(out, auxi_in[2])  # 1/16
        out = self.auxi_block3(out, auxi_in[3])  # 1/8
        out = self.auxi_block4(out, auxi_in[4])  # 1/4
        out = self.auxi_block5(out, auxi_in[5])  # 1/2
        out = self.out_conv(out)
        return out

class AuxiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inchannels = cfg.MODEL.RESNET_BOTTLENECK_DIM[1:]  # [256, 512, 1024, 2048]
        self.midchannels = cfg.MODEL.LATERAL_OUT[::-1]  # [256, 256, 256, 512]

        self.auxi_block1 = AuxiBlock(self.midchannels[2]+self.midchannels[3], 256)
        self.auxi_block2 = AuxiBlock(256 + self.midchannels[2], 256)
        self.auxi_block3 = AuxiBlock(256 + self.midchannels[2], 256)
        self.auxi_block4 = AuxiBlock(256 + self.midchannels[1], 256)
        self.auxi_block5 = AuxiBlock(256 + self.midchannels[0], 256)
        self.out_conv = AO(256, 1, 2)
        self._init_weights()

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        def init_model_weight(m):
            for child_m in m.children():
                if not isinstance(child_m, nn.ModuleList):
                    child_m.apply(init_func)

        init_model_weight(self)

    def forward(self, auxi_in):
        out = self.auxi_block1(auxi_in[0], auxi_in[1])  # 1/32
        out = self.auxi_block2(out, auxi_in[2])  # 1/16
        out = self.auxi_block3(out, auxi_in[3])  # 1/8
        out = self.auxi_block4(out, auxi_in[4])  # 1/4
        out = self.auxi_block5(out, auxi_in[5])  # 1/2
        out = self.out_conv(out)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.inchannels = cfg.MODEL.RESNET_BOTTLENECK_DIM[1:]  # [256, 512, 1024, 2048]
        self.midchannels = cfg.MODEL.LATERAL_OUT[::-1]  # [256, 256, 256, 512]
        self.upfactors = [2,2,2,2]
        self.outchannels = cfg.MODEL.DECODER_OUTPUT_C  # 1

        self.conv = FTB(inchannels=self.inchannels[3], midchannels=self.midchannels[3])
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=1, stride=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=self.upfactors[3], mode='bilinear', align_corners=True)
        
        self.ffm2 = FFM(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels = self.midchannels[2], upfactor=self.upfactors[2])
        self.ffm1 = FFM(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels = self.midchannels[1], upfactor=self.upfactors[1])
        self.ffm0 = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0], upfactor=self.upfactors[0])
        
        #self.outconv = nn.Conv2d(in_channels=self.inchannels[0], out_channels=self.outchannels, kernel_size=3, padding=1, stride=1, bias=True)
        self.outconv = AO(inchannels=self.midchannels[0], outchannels=self.outchannels, upfactor=2)
        self._init_params()
        
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): #NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
    def forward(self, features, auxi=True):
        # features' shape: # 1/32, 1/16, 1/8, 1/4
        # _,_,h,w = features[3].size()
        x_32x = self.conv(features[3])  # 1/32
        x_32 = self.conv1(x_32x)
        x_16 = self.upsample(x_32)  # 1/16

        x_8 = self.ffm2(features[2], x_16)  # 1/8
        #print('ffm2:', x.size())
        x_4 = self.ffm1(features[1], x_8)  # 1/4
        #print('ffm1:', x.size())
        x_2 = self.ffm0(features[0], x_4)  # 1/2
        #print('ffm0:', x.size())
        #-----------------------------------------
        x = self.outconv(x_2)  # original size

        if auxi:
            auxi_input = [x_32x, x_32, x_16, x_8, x_4, x_2]
            return x, auxi_input
        else:
            return x

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

    def forward(self, x, latent, mean_shift=0.0, var_shift=0.0, scale=1.0):
        style = self.mlp(latent)  # style => [batch_size, n_channels*2]


        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]

        shape2 = [1, x.size(1)] + (x.dim() - 2) * [1]
        mean_shift = mean_shift.view(shape2)
        var_shift = var_shift.view(shape2)

        mean = style[:, 1] - mean_shift.cuda()
        var = style[:, 0] + 1. - var_shift.cuda()

        x = x * (var*scale) + mean
        return x 

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

    def forward(self, x, latent, input_img, mean_shift=0.0, var_shift=0.0, shift_scale=2.0, mean_scale=2.0):
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

        x = x * (var*shift_scale) + mean*mean_scale

        return x  


class Decoder_cIMLE(nn.Module):
    def __init__(self, d_latent=32, version="v2"):
        super(Decoder_cIMLE, self).__init__()
        self.inchannels = cfg.MODEL.RESNET_BOTTLENECK_DIM[1:]  # [256, 512, 1024, 2048]
        self.midchannels = cfg.MODEL.LATERAL_OUT[::-1]  # [256, 256, 256, 512]
        self.upfactors = [2,2,2,2]
        self.outchannels = cfg.MODEL.DECODER_OUTPUT_C  # 1

        self.conv = FTB(inchannels=self.inchannels[3], midchannels=self.midchannels[3])
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=1, stride=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=self.upfactors[3], mode='bilinear', align_corners=True)
        
        self.ffm2 = FFM(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels = self.midchannels[2], upfactor=self.upfactors[2])
        self.ffm1 = FFM(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels = self.midchannels[1], upfactor=self.upfactors[1])
        self.ffm0 = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0], upfactor=self.upfactors[0])

        self.version = version
        ### Add AdaIn layers
        if version == "v2":
            print("Decoder_cIMLE with AdaIn v2")
            ## Noise1
            self.d_latent = d_latent
            self.style_mod0 = AdaIn(d_latent, self.inchannels[3])
            self.style_mod0_meanshift = torch.zeros(self.inchannels[3])
            self.style_mod0_varshift = torch.zeros(self.inchannels[3])

            ## Noise2
            self.d_latent = d_latent
            self.style_mod1 = AdaIn(d_latent, self.midchannels[3])
            self.style_mod1_meanshift = torch.zeros(self.midchannels[3])
            self.style_mod1_varshift = torch.zeros(self.midchannels[3])            

            ## Noise3
            self.d_latent = d_latent
            self.style_mod2 = AdaIn(d_latent, self.midchannels[2])
            self.style_mod2_meanshift = torch.zeros(self.midchannels[2])
            self.style_mod2_varshift = torch.zeros(self.midchannels[2])

            ## Noise4
            self.d_latent = d_latent
            self.style_mod3 = AdaIn(d_latent, self.midchannels[1])
            self.style_mod3_meanshift = torch.zeros(self.midchannels[1])
            self.style_mod3_varshift = torch.zeros(self.midchannels[1])            

        elif version == "v3":
            print("Decoder_cIMLE with AdaIn v3")
            ## Noise1
            self.d_latent = d_latent
            self.style_mod0 = AdaIn_v2(d_latent, self.inchannels[3])
            self.style_mod0_meanshift = torch.zeros(self.inchannels[3])
            self.style_mod0_varshift = torch.zeros(self.inchannels[3])

            ## Noise2
            self.d_latent = d_latent
            self.style_mod1 = AdaIn_v2(d_latent, self.midchannels[3])
            self.style_mod1_meanshift = torch.zeros(self.midchannels[3])
            self.style_mod1_varshift = torch.zeros(self.midchannels[3])            

            ## Noise3
            self.d_latent = d_latent
            self.style_mod2 = AdaIn_v2(d_latent, self.midchannels[2])
            self.style_mod2_meanshift = torch.zeros(self.midchannels[2])
            self.style_mod2_varshift = torch.zeros(self.midchannels[2])

            ## Noise4
            self.d_latent = d_latent
            self.style_mod3 = AdaIn_v2(d_latent, self.midchannels[1])
            self.style_mod3_meanshift = torch.zeros(self.midchannels[1])
            self.style_mod3_varshift = torch.zeros(self.midchannels[1]) 

        elif version == "v4":
            print("Decoder_cIMLE with AdaIn v4")
            ## Noise1
            self.d_latent = d_latent
            self.style_mod0 = AdaIn_v2(d_latent, self.inchannels[3])
            self.style_mod0_meanshift = torch.zeros(self.inchannels[3])
            self.style_mod0_varshift = torch.zeros(self.inchannels[3])

            ## Noise2
            self.d_latent = d_latent
            self.style_mod1 = AdaIn_v2(d_latent, self.midchannels[3])
            self.style_mod1_meanshift = torch.zeros(self.midchannels[3])
            self.style_mod1_varshift = torch.zeros(self.midchannels[3])            

            ## Noise3
            self.d_latent = d_latent
            self.style_mod2 = AdaIn_v2(d_latent, self.midchannels[2])
            self.style_mod2_meanshift = torch.zeros(self.midchannels[2])
            self.style_mod2_varshift = torch.zeros(self.midchannels[2])


        elif version == "v5":
            print("Decoder_cIMLE with AdaIn v5")
            ## Noise1
            self.d_latent = d_latent
            self.style_mod0 = AdaIn_v2(d_latent, self.inchannels[3])
            self.style_mod0_meanshift = torch.zeros(self.inchannels[3])
            self.style_mod0_varshift = torch.zeros(self.inchannels[3])

            ## Noise2
            self.d_latent = d_latent
            self.style_mod1 = AdaIn_v2(d_latent, self.midchannels[3])
            self.style_mod1_meanshift = torch.zeros(self.midchannels[3])
            self.style_mod1_varshift = torch.zeros(self.midchannels[3])            


        elif version == "v6":
            print("Decoder_cIMLE with AdaIn v6")
            ## Noise1
            self.d_latent = d_latent
            self.style_mod0 = AdaIn_v2(d_latent, self.inchannels[3])
            self.style_mod0_meanshift = torch.zeros(self.inchannels[3])
            self.style_mod0_varshift = torch.zeros(self.inchannels[3])

        else:
            print("Unimplemented AdaIn layer for Decoder_cIMLE.")
            exit()

        
        #self.outconv = nn.Conv2d(in_channels=self.inchannels[0], out_channels=self.outchannels, kernel_size=3, padding=1, stride=1, bias=True)
        self.outconv = AO(inchannels=self.midchannels[0], outchannels=self.outchannels, upfactor=2)
        # self._init_params()
        
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): #NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
    def forward(self, features, z, input_image=None, auxi=True):
        # features' shape: # 1/32, 1/16, 1/8, 1/4
        # _,_,h,w = features[3].size()

        ### AdaIn layer0
        if self.version == "v2":
            features[3] = self.style_mod0(features[3], z, self.style_mod0_meanshift, self.style_mod0_varshift)
        elif self.version in  ["v3", "v4","v5","v6"] :
            features[3] = self.style_mod0(features[3], z, input_image, self.style_mod0_meanshift, self.style_mod0_varshift)

        x_32x = self.conv(features[3])  # 1/32

        ### AdaIn layer1
        if self.version == "v2":
            x_32x = self.style_mod1(x_32x, z, self.style_mod1_meanshift, self.style_mod1_varshift)
        elif self.version in  ["v3", "v4","v5"] :
            x_32x = self.style_mod1(x_32x, z, input_image, self.style_mod1_meanshift, self.style_mod1_varshift)

        x_32 = self.conv1(x_32x)
        x_16 = self.upsample(x_32)  # 1/16

        x_8 = self.ffm2(features[2], x_16)  # 1/8

        ### AdaIn layer2
        if self.version == "v2":
            x_8 = self.style_mod2(x_8, z, self.style_mod2_meanshift, self.style_mod2_varshift)
        elif self.version in  ["v3", "v4"] :
            x_8 = self.style_mod2(x_8, z, input_image, self.style_mod2_meanshift, self.style_mod2_varshift)

        #print('ffm2:', x.size())
        x_4 = self.ffm1(features[1], x_8)  # 1/4

        ### AdaIn layer2
        if self.version == "v2":
            x_4 = self.style_mod3(x_4, z, self.style_mod3_meanshift, self.style_mod3_varshift)
        elif self.version == "v3":
            x_4 = self.style_mod3(x_4, z, input_image, self.style_mod3_meanshift, self.style_mod3_varshift)


        #print('ffm1:', x.size())
        x_2 = self.ffm0(features[0], x_4)  # 1/2
        #print('ffm0:', x.size())
        #-----------------------------------------
        x = self.outconv(x_2)  # original size

        if auxi:
            auxi_input = [x_32x, x_32, x_16, x_8, x_4, x_2]
            return x, auxi_input
        else:
            return x


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

    # def get_adain_init_act(self, features, z, input_image=None):

    #     ### AdaIn layer0
    #     if self.version == "v2":
    #         features[3] = self.style_mod0(features[3], z, self.style_mod0_meanshift, self.style_mod0_varshift)
    #     elif self.version in  ["v3", "v4","v5","v6"] :
    #         features[3] = self.style_mod0(features[3], z, input_image, self.style_mod0_meanshift, self.style_mod0_varshift)

    #     adain0 = features[3]
    #     x_32x = self.conv(features[3])  # 1/32

    #     ### AdaIn layer1
    #     if self.version == "v2":
    #         x_32x = self.style_mod1(x_32x, z, self.style_mod1_meanshift, self.style_mod1_varshift)
    #     elif self.version in  ["v3", "v4","v5"] :
    #         x_32x = self.style_mod1(x_32x, z, input_image, self.style_mod1_meanshift, self.style_mod1_varshift)

    #     adain1 = x_32x
    #     x_32 = self.conv1(x_32x)
    #     x_16 = self.upsample(x_32)  # 1/16

    #     x_8 = self.ffm2(features[2], x_16)  # 1/8

    #     ### AdaIn layer2
    #     if self.version == "v2":
    #         x_8 = self.style_mod2(x_8, z, self.style_mod2_meanshift, self.style_mod2_varshift)
    #     elif self.version in  ["v3", "v4"] :
    #         x_8 = self.style_mod2(x_8, z, input_image, self.style_mod2_meanshift, self.style_mod2_varshift)

    #     adain2 = x_8
    #     #print('ffm2:', x.size())
    #     x_4 = self.ffm1(features[1], x_8)  # 1/4

    #     ### AdaIn layer2
    #     if self.version == "v2":
    #         x_4 = self.style_mod3(x_4, z, self.style_mod3_meanshift, self.style_mod3_varshift)
    #     elif self.version == "v3":
    #         x_4 = self.style_mod3(x_4, z, input_image, self.style_mod3_meanshift, self.style_mod3_varshift)

    #     adain3 = x_4

    #     return adain0, adain1, adain2, adain3


    def get_adain_init_act(self, features, z, input_image=None):
        
        ### AdaIn layer0
        feat3 = features[3]
        feat2 = features[2]
        feat1 = features[1]

        if self.version == "v2":
            x = self.style_mod0(feat3, z, self.style_mod0_meanshift, self.style_mod0_varshift)
        elif self.version in  ["v3", "v4","v5","v6"] :
            x = self.style_mod0(feat3, z, input_image, self.style_mod0_meanshift, self.style_mod0_varshift)

        adain0 = x
        x_32x = self.conv(x)  # 1/32

        ### AdaIn layer1
        if self.version == "v2":
            x_32x = self.style_mod1(x_32x, z, self.style_mod1_meanshift, self.style_mod1_varshift)
        elif self.version in  ["v3", "v4","v5"] :
            x_32x = self.style_mod1(x_32x, z, input_image, self.style_mod1_meanshift, self.style_mod1_varshift)

        adain1 = x_32x
        x_32 = self.conv1(x_32x)
        x_16 = self.upsample(x_32)  # 1/16

        x_8 = self.ffm2(feat2, x_16)  # 1/8

        ### AdaIn layer2
        if self.version == "v2":
            x_8 = self.style_mod2(x_8, z, self.style_mod2_meanshift, self.style_mod2_varshift)
        elif self.version in  ["v3", "v4"] :
            x_8 = self.style_mod2(x_8, z, input_image, self.style_mod2_meanshift, self.style_mod2_varshift)

        adain2 = x_8
        #print('ffm2:', x.size())
        x_4 = self.ffm1(feat1, x_8)  # 1/4

        ### AdaIn layer2
        if self.version == "v2":
            x_4 = self.style_mod3(x_4, z, self.style_mod3_meanshift, self.style_mod3_varshift)
        elif self.version == "v3":
            x_4 = self.style_mod3(x_4, z, input_image, self.style_mod3_meanshift, self.style_mod3_varshift)

        adain3 = x_4

        return adain0, adain1, adain2, adain3


class DepthNet(nn.Module):
    __factory = {
        18: Resnet.resnet18,
        34: Resnet.resnet34,
        50: Resnet.resnet50,
        101: Resnet.resnet101,
        152: Resnet.resnet152
    }
    def __init__(self,
                backbone='resnet',
                depth=50,
                upfactors=[2, 2, 2, 2]):
        super(DepthNet, self).__init__()
        self.backbone = backbone
        self.depth = depth
        self.pretrained = cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS  # True
        self.inchannels = cfg.MODEL.RESNET_BOTTLENECK_DIM[1:]  # [256, 512, 1024, 2048]
        self.midchannels = cfg.MODEL.LATERAL_OUT[::-1]  # [256, 256, 256, 512]
        self.upfactors = upfactors
        self.outchannels = cfg.MODEL.DECODER_OUTPUT_C  # 1
        
        # Build model
        if self.backbone == 'resnet':
            if self.depth not in DepthNet.__factory:
                raise KeyError("Unsupported depth:", self.depth)
            self.encoder = DepthNet.__factory[depth](pretrained=self.pretrained)
        elif self.backbone == 'resnext101_32x8d':
            self.encoder = Resnext_torch.resnext101_32x8d(pretrained=self.pretrained)
        elif self.backbone == 'mobilenetv2':
            self.encoder = MobileNet_torch.mobilenet_v2(pretrained=self.pretrained)
        else:
            self.encoder = Resnext_torch.resnext101(pretrained=self.pretrained)

    def forward(self, x):
        x = self.encoder(x)  # 1/32, 1/16, 1/8, 1/4
        return x

class DepthNet_cIMLE(nn.Module):
    __factory = {
        18: Resnet.resnet18,
        34: Resnet.resnet34,
        50: Resnet.resnet50,
        101: Resnet.resnet101,
        152: Resnet.resnet152
    }
    def __init__(self,
                backbone='resnext101_stride32x8d',
                depth=50,
                upfactors=[2, 2, 2, 2], d_latent=512, version="v2"):
        super(DepthNet_cIMLE, self).__init__()
        self.backbone = backbone
        self.depth = depth
        self.pretrained = cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS  # True
        self.inchannels = cfg.MODEL.RESNET_BOTTLENECK_DIM[1:]  # [256, 512, 1024, 2048]
        self.midchannels = cfg.MODEL.LATERAL_OUT[::-1]  # [256, 256, 256, 512]
        self.upfactors = upfactors
        self.outchannels = cfg.MODEL.DECODER_OUTPUT_C  # 1
        
        print("In DepthNet_cIMLE")
        print(self.backbone)
        print(self.depth)
        print()

        # Build model
        if self.backbone == 'resnet':
            if self.depth not in DepthNet_cIMLE.__factory:
                raise KeyError("Unsupported depth:", self.depth)
            print("Unimplemented.")
            exit()

        elif self.backbone == 'resnext101_32x8d':
            self.encoder = Resnext_torch.resnext101_32x8d_cIMLE(pretrained=self.pretrained, d_latent=d_latent, version=version)
        
        elif self.backbone == 'mobilenetv2':
            print("Unimplemented.")
            exit()
        
        else:
            print("Unimplemented.")
            exit()

    def forward(self, x, z):
        x = self.encoder(x, z)  # 1/32, 1/16, 1/8, 1/4
        return x

    def set_mean_var_shifts(self, mean0, var0, mean1, var1, mean2, var2, mean3, var3):
        return self.encoder.set_mean_var_shifts(mean0, var0, mean1, var1, mean2, var2, mean3, var3)

    def get_adain_init_act(self, x, z):
        return self.encoder.get_adain_init_act(x, z)

class FTB(nn.Module):
    def __init__(self, inchannels, midchannels=512):
        super(FTB, self).__init__()
        self.in1 = inchannels
        self.mid = midchannels
        self.conv1 = nn.Conv2d(in_channels=self.in1, out_channels=self.mid, kernel_size=3, padding=1, stride=1,
                               bias=True)
        # NN.BatchNorm2d
        # self.sample_conv = nn.Sequential(nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3, padding=1, stride=1, bias=True),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.BatchNorm2d(num_features=self.mid),
        #                                 nn.Conv2d(in_channels=self.mid, out_channels= self.mid, kernel_size=3, padding=1, stride=1, bias=True))
        self.conv_branch = nn.Sequential(nn.ReLU(inplace=True), \
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3,
                                                   padding=1, stride=1, bias=True), \
                                         nn.BatchNorm2d(num_features=self.mid), \
                                         nn.ReLU(inplace=True), \
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3,
                                                   padding=1, stride=1, bias=True))
        self.relu = nn.ReLU(inplace=True)

        self.init_params()

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv_branch(x)
        x = self.relu(x)

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ATA(nn.Module):
    def __init__(self, inchannels, reduction=8):
        super(ATA, self).__init__()
        self.inchannels = inchannels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(self.inchannels * 2, self.inchannels // reduction),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.inchannels // reduction, self.inchannels),
                                nn.Sigmoid())
        self.init_params()

    def forward(self, low_x, high_x):
        n, c, _, _ = low_x.size()
        x = torch.cat([low_x, high_x], 1)
        x = self.avg_pool(x)
        x = x.view(n, -1)
        x = self.fc(x).view(n, c, 1, 1)
        x = low_x * x + high_x

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                # init.normal(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                # init.normal_(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class FFM(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, upfactor=2):
        super(FFM, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.ftb1 = FTB(inchannels=self.inchannels, midchannels=self.midchannels)
        # self.ata = ATA(inchannels = self.midchannels)
        self.ftb2 = FTB(inchannels=self.midchannels, midchannels=self.outchannels)

        self.upsample = nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True)

        self.init_params()

    def forward(self, low_x, high_x):
        x = self.ftb1(low_x)
        x = x + high_x
        x = self.ftb2(x)
        x = self.upsample(x)

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.Batchnorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class AO(nn.Module):
    # Adaptive output module
    def __init__(self, inchannels, outchannels, upfactor=2):
        super(AO, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.adapt_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.inchannels, out_channels=self.inchannels // 2, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.BatchNorm2d(num_features=self.inchannels // 2), \
            nn.ReLU(inplace=True), \
            nn.Conv2d(in_channels=self.inchannels // 2, out_channels=self.outchannels, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True))

        self.init_params()

    def forward(self, x):
        x = self.adapt_conv(x)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.Batchnorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ASPP(nn.Module):
    def __init__(self, inchannels=256, planes=128, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.inchannels = inchannels
        self.planes = planes
        self.rates = rates
        self.kernel_sizes = []
        self.paddings = []
        for rate in self.rates:
            if rate == 1:
                self.kernel_sizes.append(1)
                self.paddings.append(0)
            else:
                self.kernel_sizes.append(3)
                self.paddings.append(rate)
        self.atrous_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.inchannels, out_channels=self.planes, kernel_size=self.kernel_sizes[0],
                      stride=1, padding=self.paddings[0], dilation=self.rates[0], bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.planes)
            )
        self.atrous_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inchannels, out_channels=self.planes, kernel_size=self.kernel_sizes[1],
                      stride=1, padding=self.paddings[1], dilation=self.rates[1], bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.planes),
            )
        self.atrous_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.inchannels, out_channels=self.planes, kernel_size=self.kernel_sizes[2],
                      stride=1, padding=self.paddings[2], dilation=self.rates[2], bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.planes),
            )
        self.atrous_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.inchannels, out_channels=self.planes, kernel_size=self.kernel_sizes[3],
                      stride=1, padding=self.paddings[3], dilation=self.rates[3], bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.planes),
            )

        # self.conv = nn.Conv2d(in_channels=self.planes * 4, out_channels=self.inchannels, kernel_size=3, padding=1, stride=1, bias=True)

    def forward(self, x):
        x = torch.cat([self.atrous_0(x), self.atrous_1(x), self.atrous_2(x), self.atrous_3(x)], 1)
        # x = self.conv(x)

        return x


# ==============================================================================================================


class ResidualConv(nn.Module):
    def __init__(self, inchannels):
        super(ResidualConv, self).__init__()
        # NN.BatchNorm2d
        self.conv = nn.Sequential(
            # nn.BatchNorm2d(num_features=inchannels),
            nn.ReLU(inplace=False),
            # nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1, stride=1, groups=inchannels,bias=True),
            # nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=1, padding=0, stride=1, groups=1,bias=True)
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels / 2, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=inchannels / 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=inchannels / 2, out_channels=inchannels, kernel_size=3, padding=1, stride=1,
                      bias=False)
        )
        self.init_params()

    def forward(self, x):
        x = self.conv(x) + x
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class FeatureFusion(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(FeatureFusion, self).__init__()
        self.conv = ResidualConv(inchannels=inchannels)
        # NN.BatchNorm2d
        self.up = nn.Sequential(ResidualConv(inchannels=inchannels),
                                nn.ConvTranspose2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3,
                                                   stride=2, padding=1, output_padding=1),
                                nn.BatchNorm2d(num_features=outchannels),
                                nn.ReLU(inplace=True))

    def forward(self, lowfeat, highfeat):
        return self.up(highfeat + self.conv(lowfeat))

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class SenceUnderstand(nn.Module):
    def __init__(self, channels):
        super(SenceUnderstand, self).__init__()
        self.channels = channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(8)
        self.fc = nn.Sequential(nn.Linear(512 * 8 * 8, self.channels),
                                nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True))
        self.initial_params()

    def forward(self, x):
        n, c, h, w = x.size()
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(n, -1)
        x = self.fc(x)
        x = x.view(n, self.channels, 1, 1)
        x = self.conv2(x)
        x = x.repeat(1, 1, h, w)
        return x

    def initial_params(self, dev=0.01):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # print torch.sum(m.weight)
                m.weight.data.normal_(0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):
                # print torch.sum(m.weight)
                m.weight.data.normal_(0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, dev)


if __name__ == '__main__':
    net = DepthNet(depth=50, pretrained=True)
    print(net)
    inputs = torch.ones(4,3,128,128)
    out = net(inputs)
    print(out.size())

