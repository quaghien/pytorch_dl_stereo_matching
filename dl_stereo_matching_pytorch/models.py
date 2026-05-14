from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn


@dataclass(frozen=True)
class NetworkSpec:
    kernels: List[int]
    channels: List[int]

    @property
    def receptive_field(self) -> int:
        return 1 + sum(kernel - 1 for kernel in self.kernels)


NETWORK_SPECS = {
    "win19_dep9": NetworkSpec(kernels=[3] * 9, channels=[64] * 9),
    "win37_dep9": NetworkSpec(kernels=[5] * 9, channels=[32, 32] + [64] * 7),
}


def get_network_spec(net_type: str) -> NetworkSpec:
    try:
        return NETWORK_SPECS[net_type]
    except KeyError as exc:
        raise ValueError("net_type must be 'win19_dep9' or 'win37_dep9'") from exc


class SiameseFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, spec: NetworkSpec) -> None:
        super().__init__()
        layers = []
        current_channels = in_channels
        last_idx = len(spec.kernels) - 1
        for idx, (kernel_size, out_channels) in enumerate(zip(spec.kernels, spec.channels)):
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=kernel_size, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            if idx != last_idx:
                layers.append(nn.ReLU(inplace=True))
            current_channels = out_channels
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SiameseStereoMatching(nn.Module):
    def __init__(self, in_channels: int, net_type: str) -> None:
        super().__init__()
        self.spec = get_network_spec(net_type)
        self.net_type = net_type
        self.backbone = SiameseFeatureExtractor(in_channels, self.spec)

    @property
    def receptive_field(self) -> int:
        return self.spec.receptive_field

    def extract_feature_maps(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(left), self.backbone(right)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_features, right_features = self.extract_feature_maps(left, right)
        if left_features.shape[-2:] != (1, 1):
            raise ValueError(
                f"Left patch must collapse to 1x1 after valid convolutions; got {tuple(left_features.shape[-2:])}. "
                f"Use patch size {self.receptive_field} for {self.net_type}."
            )
        if right_features.shape[-2] != 1:
            raise ValueError("Right patch height must collapse to 1 after valid convolutions")

        left_vector = left_features.squeeze(-1).squeeze(-1)
        right_vectors = right_features.squeeze(-2)
        return torch.einsum("bc,bcd->bd", left_vector, right_vectors)


def create_model(net_type: str, in_channels: int) -> SiameseStereoMatching:
    return SiameseStereoMatching(in_channels=in_channels, net_type=net_type)


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def build_cost_volume(left_features: torch.Tensor, right_features: torch.Tensor, disp_range: int) -> torch.Tensor:
    if left_features.shape != right_features.shape:
        raise ValueError("left_features and right_features must have the same shape")

    batch, channels, height, width = left_features.shape
    volume = left_features.new_zeros((batch, disp_range, height, width))
    for disparity in range(disp_range):
        if disparity >= width:
            break
        left_slice = left_features[:, :, :, disparity:]
        right_slice = right_features[:, :, :, : width - disparity]
        volume[:, disparity, :, disparity:] = (left_slice * right_slice).sum(dim=1)
    return volume
