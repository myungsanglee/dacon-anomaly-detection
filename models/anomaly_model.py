import sys
import os
from turtle import back
sys.path.append(os.getcwd())

import torch
from torch import nn
import torchsummary
import timm

from models.layers.conv_block import Conv2dBnRelu
from models.initialize import weight_initialize
from utils.module_select import get_model


class AnomalyModel01(nn.Module):
    def __init__(self, backbone, in_channels, num_classes, input_size):
        super(AnomalyModel01, self).__init__()
        self.backbone = backbone(in_channels).features
        c= self.backbone(torch.randn((1, in_channels, input_size, input_size), dtype=torch.float32)).size(1)

        self.classifier = nn.Sequential(
            Conv2dBnRelu(c, num_classes, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class AnomalyModel02(nn.Module):
    def __init__(self, backbone, in_channels, num_classes, num_state, input_size):
        super(AnomalyModel02, self).__init__()
        self.backbone = backbone(in_channels).features
        c= self.backbone(torch.randn((1, in_channels, input_size, input_size), dtype=torch.float32)).size(1)

        self.classifier = nn.Sequential(
            Conv2dBnRelu(c, num_classes, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.state_classifier = nn.Sequential(
            Conv2dBnRelu(c, 1024, 3),

            Conv2dBnRelu(1024, 2048, 3, 2),
            Conv2dBnRelu(2048, 2048, 3),
            Conv2dBnRelu(2048, 1024, 1),

            Conv2dBnRelu(1024, num_state, 1),
            Conv2dBnRelu(num_state, num_state, 3, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.backbone(x)
        pred_class = self.classifier(x)
        pred_state = self.state_classifier(x)
        return pred_class, pred_state


class AnomalyModel03(nn.Module):
    def __init__(self, backbone, in_channels, num_classes, num_state, input_size):
        super(AnomalyModel03, self).__init__()
        self.backbone = backbone
        c= self.backbone(torch.randn((1, in_channels, input_size, input_size), dtype=torch.float32)).size(1)

        self.class_classifier = nn.Sequential(
            Conv2dBnRelu(c, num_classes, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.state_classifier = nn.Sequential(
            Conv2dBnRelu(c, num_state, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.backbone(x)
        pred_class = self.class_classifier(x)
        pred_state = self.state_classifier(x)
        return pred_class, pred_state


class AnomalyModel04(nn.Module):
    def __init__(self, backbone, in_channels, num_classes, num_state, input_size):
        super(AnomalyModel03, self).__init__()
        self.backbone = backbone
        c= self.backbone(torch.randn((1, in_channels, input_size, input_size), dtype=torch.float32)).size(1)

        self.class_classifier = nn.Sequential(
            Conv2dBnRelu(c, num_classes, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.state_classifier = nn.Sequential(
            Conv2dBnRelu(c, num_state, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.backbone(x)
        pred_class = self.class_classifier(x)
        pred_state = self.state_classifier(x)
        return pred_class, pred_state






if __name__ == '__main__':
    input_size = 640
    in_channels = 3
    num_classes = 15
    num_state = 88

    # backbone = get_model('darknet19')
    # backbone = timm.create_model('efficientnet_b4', True, num_classes=0, global_pool='')
    # backbone = timm.create_model('resnet50', True, num_classes=0, global_pool='')
    
    backbone = timm.create_model('resnet50', True, num_classes=10)
    torchsummary.summary(backbone, (in_channels, input_size, input_size), batch_size=1, device='cpu')
    
    # model = AnomalyModel01(backbone, in_channels, num_classes, input_size)
    # model = AnomalyModel02(backbone, in_channels, num_classes, num_state, input_size)
    # model = AnomalyModel03(backbone, in_channels, num_classes, num_state, input_size)
    
    # torchsummary.summary(model, (in_channels, input_size, input_size), batch_size=1, device='cpu')
    
    # model = model.features[:17]
    # torchsummary.summary(model, (3, input_size, input_size), batch_size=1, device='cpu')
    
    # print(list(model.children()))
    # print(f'\n-------------------------------------------------------------\n')
    # new_model = nn.Sequential(*list(model.children())[:-1])
    # print(new_model.modules)
    
    # for idx, child in enumerate(model.children()):
    #     print(child)
    #     if idx == 0:
    #         for i, param in enumerate(child.parameters()):
    #             print(i, param)
    #             param.requires_grad = False
    #             if i == 4:
    #                 break

    # torchsummary.summary(model, (3, 64, 64), batch_size=1, device='cpu')
    
    # from torchvision import models

    # model = models.resnet18(num_classes=200)
    # models.vgg16
    # # model = models.resnet50(num_classes=200)
    # model = models.efficientnet_b0(num_classes=200)
    # torchsummary.summary(model, (3, 64, 64), batch_size=1, device='cpu')

