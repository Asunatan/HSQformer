import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet101, densenet201, resnext101_32x8d, ResNet101_Weights, DenseNet201_Weights, \
    ResNeXt101_32X8D_Weights


class ThyNet(nn.Module):
    def __init__(self, num_classes = 2):
        super(ThyNet, self).__init__()
        self.Resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        self.Densnet = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        self.ResNeXt = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        self.Resnet.fc = nn.Linear(in_features=self.Resnet.fc.in_features,out_features=num_classes)
        self.Densnet.classifier = nn.Linear(in_features=self.Densnet.classifier.in_features, out_features=num_classes)
        self.ResNeXt.fc = nn.Linear(in_features=self.ResNeXt.fc.in_features, out_features=num_classes)

    def forward(self, x):
        x_resnet = self.Resnet(x)
        x_densnet = self.Densnet(x)
        x_resnext = self.ResNeXt(x)

        return x_resnet+x_densnet+x_resnext