"""Candidate classifiers for the (3, 16, 17) keypoint window.

A  SmallCNN2D   : 3x3 conv x3 on the "joint image"           (~95k params)
B  TemporalCNN  : first conv spans all 17 joints, then time-only convs
                  (a 1D temporal CNN written as Conv2d)        (~49k params)
R  resnet18     : baseline, same family as the reference project (~11.2M)

Only Conv2d / BatchNorm / ReLU / MaxPool / GlobalAvgPool / Linear are used so
that DX-COM can place the whole graph on the NPU.
"""
import torch
import torch.nn as nn

from .keypoints import N_JOINTS
from .labels import NUM_CLASSES


def _cbr(i, o, k=3, p=1):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=p, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SmallCNN2D(nn.Module):
    def __init__(self, n_cls=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            _cbr(3, 32), _cbr(32, 64), nn.MaxPool2d((2, 1)), _cbr(64, 128), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(128, n_cls)

    def forward(self, x):
        return self.fc(torch.flatten(self.features(x), 1))


class TemporalCNN(nn.Module):
    def __init__(self, n_cls=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            _cbr(3, 64, (3, N_JOINTS), (1, 0)),   # -> 64 x 16 x 1, sees every joint at once
            _cbr(64, 64, (3, 1), (1, 0)),
            nn.MaxPool2d((2, 1)),
            _cbr(64, 128, (3, 1), (1, 0)),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(128, n_cls)

    def forward(self, x):
        return self.fc(torch.flatten(self.features(x), 1))


def resnet18_baseline(n_cls=NUM_CLASSES, pretrained=False):
    from torchvision.models import resnet18, ResNet18_Weights
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    net.fc = nn.Linear(net.fc.in_features, n_cls)
    return net


class Deployable(nn.Module):
    """Input 0..255 (as fed on the board) -> logits. Normalisation is inside the graph."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x / 127.5 - 1.0)


BUILDERS = {"cnn2d": SmallCNN2D, "tcn": TemporalCNN, "resnet18": resnet18_baseline}


def build(name, n_cls=NUM_CLASSES):
    return BUILDERS[name](n_cls)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
