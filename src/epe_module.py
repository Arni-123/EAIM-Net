"""
epe_module.py — EAIM-Net v5 Final

Environmental Parameter Estimator (EPE)
Predicts weather / time-of-day / illumination from input image.

v5 changes from v4:
  - EPELoss lambdas updated: weather=0.5, time=0.3, illum=0.3
  - No other changes — EPE achieved 98.2% accuracy and is stable
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from typing import Dict, Tuple


class EnvironmentalParameterEstimator(nn.Module):
    """
    MobileNetV3-Large backbone with three classification heads.

    Architecture:
        MobileNetV3-Large → AdaptiveAvgPool2d(1) → Linear(960, 256) → ReLU → Dropout(0.2)
        ├── weather_head  →  5 logits  (clear / rain / fog / light_snow / glare)
        ├── time_head     →  4 logits  (dawn / day / dusk / night)
        └── illum_head    →  3 logits  (low / medium / high)

    heavy_snow is OUT OF SCOPE — excluded by design.
    """

    def __init__(self,
                 backbone='mobilenet_v3_large',
                 num_weather_classes=5,
                 num_time_classes=4,
                 num_illum_classes=3,
                 feature_dim=256,
                 pretrained=True):
        super().__init__()

        if backbone == 'mobilenet_v3_large':
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.mobilenet_v3_large(weights=weights)
            self.backbone    = nn.Sequential(*list(net.children())[:-1])
            backbone_dim     = 960
        elif backbone == 'mobilenet_v3_small':
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.mobilenet_v3_small(weights=weights)
            self.backbone    = nn.Sequential(*list(net.children())[:-1])
            backbone_dim     = 576
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.feature_transform = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        def _head(out_dim):
            return nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, out_dim),
            )

        self.weather_head = _head(num_weather_classes)
        self.time_head    = _head(num_time_classes)
        self.illum_head   = _head(num_illum_classes)

        self.num_weather_classes = num_weather_classes
        self.num_time_classes    = num_time_classes
        self.num_illum_classes   = num_illum_classes
        self.feature_dim         = feature_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            features        [B, feature_dim]
            weather_logits  [B, 5]
            time_logits     [B, 4]
            illum_logits    [B, 3]
        """
        feat = self.backbone(x)
        feat = self.pool(feat).view(feat.size(0), -1)
        features = self.feature_transform(feat)
        return (
            features,
            self.weather_head(features),
            self.time_head(features),
            self.illum_head(features),
        )

    def predict_environment(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience method — returns predicted classes and probabilities."""
        features, wl, tl, il = self.forward(x)
        return {
            'features':      features,
            'weather_class': torch.argmax(torch.softmax(wl, 1), 1),
            'weather_probs': torch.softmax(wl, 1),
            'time_class':    torch.argmax(torch.softmax(tl, 1), 1),
            'time_probs':    torch.softmax(tl, 1),
            'illum_class':   torch.argmax(torch.softmax(il, 1), 1),
            'illum_probs':   torch.softmax(il, 1),
        }


class EPELoss(nn.Module):
    """
    L_EPE = lambda_w * CE(weather) + lambda_t * CE(time) + lambda_i * CE(illum)

    v5 defaults: lambda_w=0.5, lambda_t=0.3, lambda_i=0.3
    (weather gets higher weight since it drives filter selection most)
    """

    def __init__(self, lambda_weather=0.5, lambda_time=0.3, lambda_illum=0.3):
        super().__init__()
        self.lw = lambda_weather
        self.lt = lambda_time
        self.li = lambda_illum
        self.ce = nn.CrossEntropyLoss()

    def forward(self, weather_logits, time_logits, illum_logits,
                weather_labels, time_labels, illum_labels):
        lw    = self.ce(weather_logits, weather_labels)
        lt    = self.ce(time_logits,    time_labels)
        li    = self.ce(illum_logits,   illum_labels)
        total = self.lw * lw + self.lt * lt + self.li * li
        return total, {
            'total':   total.item(),
            'weather': lw.item(),
            'time':    lt.item(),
            'illum':   li.item(),
        }


if __name__ == '__main__':
    model = EnvironmentalParameterEstimator(pretrained=False)
    x     = torch.randn(2, 3, 512, 512)
    feats, wl, tl, il = model(x)
    print(f'Features       : {feats.shape}')
    print(f'Weather logits : {wl.shape}')
    print(f'Time logits    : {tl.shape}')
    print(f'Illum logits   : {il.shape}')
    total = sum(p.numel() for p in model.parameters())
    print(f'Total params   : {total:,}')
