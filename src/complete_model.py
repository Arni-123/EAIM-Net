"""
complete_model.py — EAIM-Net v5 Final

Integrates EPE → ESS → AFB pipeline.

v5 changes from v4:
  - weight_entropy now in forward() output dict
  - glare_map passed through to AFB/GlareReductionModule
  - EnhancementLoss replaced by losses.py (text_aware_loss)
  - analyze_enhancement uses tau from ess.tau property
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from epe_module import EnvironmentalParameterEstimator
from ess_module  import EnhancementStrategySelector, AdaptiveFilterCombiner, FILTER_NAMES
from afb_module  import AdaptiveFilterBank

WEATHER_NAMES = ['Clear', 'Rain', 'Fog', 'Light Snow', 'Glare']
TIME_NAMES    = ['Dawn', 'Day', 'Dusk', 'Night']
ILLUM_NAMES   = ['Low', 'Medium', 'High']


class AdaptiveEnhancementModel(nn.Module):
    """
    EAIM-Net v5

    Forward pass:
        x  →  EPE  →  (env_features, weather_logits, time_logits, illum_logits)
                   →  ESS  →  (filter_weights, filter_params)
                   →  AFB  →  filter_outputs  [B, 5, 3, H, W]
                   →  Combiner  →  enhanced   [B, 3, H, W]

    Output dict keys:
        enhanced, env_features,
        weather_logits, time_logits, illum_logits,
        filter_weights, filter_params, filter_outputs,
        weight_entropy    ← NEW in v5 (normalised H per sample)
    """

    def __init__(self,
                 backbone='mobilenet_v3_large',
                 num_weather_classes=5,
                 num_time_classes=4,
                 num_illum_classes=3,
                 feature_dim=256,
                 num_filters=5,
                 pretrained=True):
        super().__init__()

        self.epe = EnvironmentalParameterEstimator(
            backbone            = backbone,
            num_weather_classes = num_weather_classes,
            num_time_classes    = num_time_classes,
            num_illum_classes   = num_illum_classes,
            feature_dim         = feature_dim,
            pretrained          = pretrained,
        )
        self.ess      = EnhancementStrategySelector(
                            feature_dim=feature_dim, num_filters=num_filters)
        self.afb      = AdaptiveFilterBank()
        self.combiner = AdaptiveFilterCombiner()
        self.num_filters = num_filters

    def forward(self, x: torch.Tensor,
                glare_map: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x         : [B, 3, H, W]  degraded input
            glare_map : [B, 1, H, W]  optional SD1 glare map
                        (passed to GlareReductionModule filter)
        Returns:
            dict with all intermediate and final outputs
        """
        # Stage 1: Classify environmental condition
        env_features, weather_logits, time_logits, illum_logits = self.epe(x)

        # Stage 2: Select enhancement strategy
        filter_weights, filter_params = self.ess(env_features)

        # Stage 3: Run all filters in parallel
        filter_outputs = self.afb(x, filter_params, glare_map=glare_map)

        # Stage 4: Weighted combination
        enhanced = self.combiner(filter_outputs, filter_weights)

        # Compute normalised weight entropy H (health metric)
        weight_entropy = self.ess.weight_entropy(filter_weights)

        return {
            'enhanced':       enhanced,
            'env_features':   env_features,
            'weather_logits': weather_logits,
            'time_logits':    time_logits,
            'illum_logits':   illum_logits,
            'filter_weights': filter_weights,
            'filter_params':  filter_params,
            'filter_outputs': filter_outputs,
            'weight_entropy': weight_entropy,   # [B]  normalised H per sample
        }

    def enhance(self, x: torch.Tensor) -> torch.Tensor:
        """Simple enhancement — returns enhanced image only."""
        return self.forward(x)['enhanced']

    def analyze_enhancement(self, x: torch.Tensor) -> Dict:
        """Full analysis with human-readable labels."""
        out = self.forward(x)
        w   = torch.argmax(out['weather_logits'], 1)
        t   = torch.argmax(out['time_logits'],    1)
        i   = torch.argmax(out['illum_logits'],   1)
        fw  = out['filter_weights'][0]
        return {
            'weather':        [WEATHER_NAMES[wi.item()] for wi in w],
            'time':           [TIME_NAMES[ti.item()]    for ti in t],
            'illumination':   [ILLUM_NAMES[ii.item()]   for ii in i],
            'filter_weights': {FILTER_NAMES[k]: fw[k].item()
                               for k in range(self.num_filters)},
            'dominant_filter': FILTER_NAMES[fw.argmax().item()],
            'weight_entropy':  out['weight_entropy'][0].item(),
            'tau':             self.ess.tau.item(),
            'enhanced_image':  out['enhanced'],
        }


class EnhancementLoss(nn.Module):
    """
    Legacy loss kept for backward compatibility with v4 checkpoints.
    New code should use losses.text_aware_loss() directly.
    """

    def __init__(self, lambda_perceptual=0.1, lambda_ssim=0.15):
        super().__init__()
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim       = lambda_ssim
        self.l1  = nn.L1Loss()

    def forward(self, enhanced, target):
        loss_l1 = self.l1(enhanced, target)
        # Gradient loss
        dx_e = enhanced[:,:,:,1:] - enhanced[:,:,:,:-1]
        dx_t = target  [:,:,:,1:] - target  [:,:,:,:-1]
        dy_e = enhanced[:,:,1:,:] - enhanced[:,:,:-1,:]
        dy_t = target  [:,:,1:,:] - target  [:,:,:-1,:]
        loss_percep = self.l1(dx_e, dx_t) + self.l1(dy_e, dy_t)
        total = loss_l1 + self.lambda_perceptual * loss_percep
        return total, {'total': total.item(), 'l1': loss_l1.item(),
                       'perceptual': loss_percep.item()}


if __name__ == '__main__':
    model = AdaptiveEnhancementModel(pretrained=False)
    x     = torch.rand(2, 3, 512, 512)
    out   = model(x)
    print(f"Enhanced       : {out['enhanced'].shape}")
    print(f"Filter weights : {out['filter_weights'].shape}  "
          f"sum={out['filter_weights'][0].sum():.4f}")
    print(f"Weight entropy : {out['weight_entropy']}")
    print(f"Tau            : {model.ess.tau.item():.4f}")
    analysis = model.analyze_enhancement(x)
    print(f"\nAnalysis sample 0:")
    print(f"  Weather  : {analysis['weather'][0]}")
    print(f"  Dominant : {analysis['dominant_filter']}")
    print(f"  H        : {analysis['weight_entropy']:.4f}")
    for name, w in analysis['filter_weights'].items():
        print(f"  {name:<22} {w:.4f}")
    total = sum(p.numel() for p in model.parameters())
    print(f'\nTotal parameters: {total:,}')
