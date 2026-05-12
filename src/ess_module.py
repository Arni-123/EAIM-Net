"""
ess_module.py — EAIM-Net v5 Final

Enhancement Strategy Selector (ESS)

v5 KEY CHANGE: Routing loss REMOVED.
  In v4, a cross-entropy routing loss forced ESS to select a fixed filter
  per weather class. This dominated the enhancement loss 4:1, caused
  temperature tau → 0.10 and entropy H → 0.05 (hard argmax, no blending).
  Removing it allows H → 0.69 and genuine multi-filter blending.

v5 also changes:
  - tau initialised to log(1.0) = 0 so initial weights are uniform (1/5 each)
  - tau clamped 0.1 to 2.0 (wider range than v4's 0.1-1.0)
  - LayerNorm added in shared_net for training stability
  - WEATHER_TO_FILTER kept for visualization / interpretability only
    (not used in any loss computation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


FILTER_NAMES = [
    'Low-light',        # 0
    'Dehazing',         # 1
    'Rain Removal',     # 2
    'Illum. Norm.',     # 3
    'Glare Reduction',  # 4
]

# For visualization / interpretability only — NOT used in any loss
WEATHER_TO_FILTER = {
    0: 0,   # clear      → Low-light (night images)
    1: 2,   # rain       → Rain Removal
    2: 1,   # fog        → Dehazing
    3: 2,   # light_snow → Rain Removal (similar streak pattern)
    4: 4,   # glare      → Glare Reduction
}


class EnhancementStrategySelector(nn.Module):
    """
    Two-layer MLP with LayerNorm, conditioned on EPE features.
    Outputs temperature-scaled softmax blending weights.

    tau is learnable, initialised to 1.0 (uniform weights).
    ESS learns to increase or decrease tau based on what
    blending improves image quality — no routing supervision.
    """

    def __init__(self,
                 feature_dim: int = 256,
                 num_filters:  int = 5,
                 hidden_dim:   int = 128):
        super().__init__()
        self.num_filters = num_filters

        # Shared feature MLP with LayerNorm for stability
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Filter weight logits
        self.weight_fc = nn.Linear(hidden_dim, num_filters)

        # Learnable log-temperature: tau = exp(log_tau)
        # Initialised to 0 → tau=1.0 → uniform softmax (1/5 each at start)
        self.log_tau = nn.Parameter(torch.zeros(1))

        # Per-filter parameter predictors (strength + 7 aux params each)
        self.param_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 8),
                nn.Sigmoid(),
            )
            for _ in range(num_filters)
        ])

    @property
    def tau(self) -> torch.Tensor:
        """Current temperature value (scalar tensor)."""
        return torch.clamp(torch.exp(self.log_tau), min=0.1, max=2.0)

    def forward(self, env_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            env_features: [B, feature_dim]
        Returns:
            filter_weights: [B, num_filters]  sums to 1 per sample
            filter_params:  [B, num_filters, 8]
        """
        shared = self.shared_net(env_features)

        # Temperature-scaled softmax
        logits         = self.weight_fc(shared)
        filter_weights = F.softmax(logits / self.tau, dim=1)

        # Per-filter parameters
        filter_params = torch.stack(
            [pp(shared) for pp in self.param_predictors], dim=1
        )  # [B, num_filters, 8]

        return filter_weights, filter_params

    def weight_entropy(self, filter_weights: torch.Tensor) -> torch.Tensor:
        """
        Normalised weight entropy H = -sum(w * log(w)) / log(K).
        H=1 → uniform (all filters equal weight)
        H=0 → single filter selected (hard argmax)
        H>0.2 → genuine blending (v5 target: H≈0.69)
        """
        eps = 1e-8
        H   = -(filter_weights * torch.log(filter_weights + eps)).sum(dim=1)
        return H / torch.log(torch.tensor(self.num_filters, dtype=torch.float32))

    def get_active_filters(self, env_features: torch.Tensor,
                           threshold: float = 0.10) -> Dict[str, float]:
        """Return filters with weight above threshold (for single sample)."""
        fw, _ = self.forward(env_features)
        return {
            FILTER_NAMES[i]: fw[0, i].item()
            for i in range(self.num_filters)
            if fw[0, i].item() > threshold
        }


class AdaptiveFilterCombiner(nn.Module):
    """
    Weighted combination: I_enh = sum_k ( w_k * F_k(I, theta_k) )
    """

    def forward(self, filter_outputs: torch.Tensor,
                filter_weights:  torch.Tensor) -> torch.Tensor:
        """
        Args:
            filter_outputs: [B, K, C, H, W]
            filter_weights: [B, K]
        Returns:
            combined:       [B, C, H, W]
        """
        B, K, C, H, W = filter_outputs.shape
        weights = filter_weights.view(B, K, 1, 1, 1)
        return (filter_outputs * weights).sum(dim=1)


if __name__ == '__main__':
    ess     = EnhancementStrategySelector(feature_dim=256, num_filters=5)
    feats   = torch.randn(4, 256)
    fw, fp  = ess(feats)

    ent = ess.weight_entropy(fw)
    print(f'Filter weights : {fw.shape}   sum={fw[0].sum():.4f}')
    print(f'Filter params  : {fp.shape}')
    print(f'Temperature tau: {ess.tau.item():.4f}  (init=1.0)')
    print(f'Weight entropy H={ent.mean().item():.4f}  (target >0.20)')
    print()
    print('Weather→filter mapping (for visualization only, NOT a loss):')
    wnames = {0:'clear',1:'rain',2:'fog',3:'light_snow',4:'glare'}
    for w, f in WEATHER_TO_FILTER.items():
        print(f'  {wnames[w]:<12} → {FILTER_NAMES[f]}')
