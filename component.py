import math
from typing import List

import torch
import torch.nn as nn
from capybara import imresize
from chameleon import GAP, StarReLU, build_backbone, build_neck
from torch.nn.functional import linear, normalize


class Permute(nn.Module):

    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class Transpose(nn.Module):

    def __init__(self, dim1: int, dim2: int) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim1, self.dim2)


class Backbone(nn.Module):

    def __init__(self, name, **kwargs):
        super().__init__()
        self.backbone = build_backbone(name=name, **kwargs)

        with torch.no_grad():
            dummy = torch.rand(1, 3, 128, 128)
            self.channels = [i.size(1) for i in self.backbone(dummy)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)



class Baseline(nn.Module):

    def __init__(self, in_channels_list, num_classes):
        super().__init__()
        self.proj = nn.Sequential(
            GAP(),
            nn.Linear(in_channels_list[4], num_classes),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.proj(x[-1])


class MarginHead(nn.Module):

    def __init__(self, in_channels_list, hid_dim, num_classes):
        super().__init__()
        self.feat = nn.Sequential(
            GAP(),
            nn.Linear(in_channels_list[4], hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
        )

        self.weight = torch.nn.Parameter(torch.normal(
            0, 0.01, (num_classes, hid_dim)))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feat = self.feat(x[-1])
        norm_embeddings = normalize(feat, dim=1)
        norm_weight_activated = normalize(self.weight)
        logits = linear(norm_embeddings, norm_weight_activated)
        return logits
