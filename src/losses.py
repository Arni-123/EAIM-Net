"""
losses.py — EAIM-Net v5 Final
All loss functions in one place.

Losses:
  ssim_loss          — differentiable SSIM (1 - SSIM)
  base_enh_loss      — L1 + SSIM, used in filter pre-training
  glare_masked_loss  — extra weighted loss inside SD1 glare hotspot
  text_region_mask   — Laplacian variance mask for text regions
  text_aware_loss    — base_enh_loss + extra weight on text pixels
  EPELoss            — weather + time + illum classification
  FullPipelineLoss   — combines EPE + text_aware for joint fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SSIM ──────────────────────────────────────────────────────────────────────

def _gauss_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    c = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-c ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              window_size: int = 11) -> torch.Tensor:
    """
    Differentiable 1 - SSIM.
    pred, target: [B, C, H, W] float in [0, 1].
    Returns scalar.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    B, C, H, W = pred.shape
    win = _gauss_kernel(window_size).to(pred.device)
    win = win.expand(C, 1, window_size, window_size)
    pad = window_size // 2

    mu1  = F.conv2d(pred,   win, padding=pad, groups=C)
    mu2  = F.conv2d(target, win, padding=pad, groups=C)
    s1   = F.conv2d(pred * pred,     win, padding=pad, groups=C) - mu1 ** 2
    s2   = F.conv2d(target * target, win, padding=pad, groups=C) - mu2 ** 2
    s12  = F.conv2d(pred * target,   win, padding=pad, groups=C) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return 1.0 - ssim_map.mean()


# ── Filter pre-training loss ───────────────────────────────────────────────────

def base_enh_loss(pred: torch.Tensor, target: torch.Tensor,
                  lam_ssim: float = 0.3) -> torch.Tensor:
    """L1 + SSIM. Used for all individual filter pre-training."""
    return nn.L1Loss()(pred, target) + lam_ssim * ssim_loss(pred, target)


# ── Glare hotspot loss (SD1) ───────────────────────────────────────────────────

def glare_masked_loss(pred: torch.Tensor, target: torch.Tensor,
                      glare_map: torch.Tensor,
                      hot_weight: float = 3.0,
                      threshold:  float = 0.3) -> torch.Tensor:
    """
    Extra weighted loss inside the bright glare hotspot region.
    Uses SD1 glare map (panel 3 of strip) as spatial mask.

    glare_map: [B, 1, H, W] intensity in [0, 1].
               Pixels > threshold are inside the glare hotspot.
    hot_weight: extra multiplier applied inside hotspot (3x recommended).
    """
    mask = (glare_map.to(pred.device) > threshold).float()
    return ((pred - target).abs() * mask * hot_weight).mean()


def glare_filter_loss(pred: torch.Tensor, target: torch.Tensor,
                      batch: dict) -> torch.Tensor:
    """Combined glare training loss: base + masked hotspot."""
    loss = base_enh_loss(pred, target)
    if 'glare_map' in batch:
        loss = loss + glare_masked_loss(pred, target, batch['glare_map'])
    return loss


# ── Text-aware loss ────────────────────────────────────────────────────────────

def text_region_mask(gt: torch.Tensor,
                     kernel_size: int   = 5,
                     threshold:   float = 0.02) -> torch.Tensor:
    """
    Approximate text region mask using Laplacian variance.

    Text pixels have high local frequency (sharp edges between
    character strokes and background). No neural text detector needed.

    gt           : [B, 3, H, W] ground-truth image
    kernel_size  : pooling window for local variance
    threshold    : variance threshold to classify as text region
    Returns      : [B, 1, H, W] binary mask (1 = text region)
    """
    # Convert to grayscale
    gray = (0.299 * gt[:, 0:1] +
            0.587 * gt[:, 1:2] +
            0.114 * gt[:, 2:3])

    # Laplacian edge detection
    lap_k = torch.tensor([[0., -1.,  0.],
                           [-1.,  4., -1.],
                           [0., -1.,  0.]],
                          dtype=torch.float32,
                          device=gt.device).view(1, 1, 3, 3)
    lap  = F.conv2d(gray, lap_k, padding=1).abs()

    # Local max variance — highlights text stroke boundaries
    var  = F.max_pool2d(lap, kernel_size=kernel_size,
                        stride=1, padding=kernel_size // 2)
    return (var > threshold).float()


def text_aware_loss(pred:       torch.Tensor,
                    target:     torch.Tensor,
                    lam_text:   float = 2.0,
                    lam_ssim:   float = 0.3) -> torch.Tensor:
    """
    Base loss + extra weight on text/edge regions.

    Total = L1 + SSIM + lam_text * (masked L1 on text regions)

    lam_text = 2.0 means text pixels contribute 3x total loss weight
    (1x from base L1 + 2x from extra text L1).
    """
    base = base_enh_loss(pred, target, lam_ssim=lam_ssim)
    mask = text_region_mask(target).detach()
    text_l1 = ((pred - target).abs() * mask * lam_text).mean()
    return base + text_l1


# ── EPE classification loss ────────────────────────────────────────────────────

class EPELoss(nn.Module):
    """
    Multi-task classification loss for Environmental Parameter Estimator.
    L_EPE = λ_w * CE(weather) + λ_t * CE(time) + λ_i * CE(illum)
    """

    def __init__(self,
                 lambda_weather: float = 0.5,
                 lambda_time:    float = 0.3,
                 lambda_illum:   float = 0.3):
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
            'total': total.item(),
            'weather': lw.item(),
            'time':    lt.item(),
            'illum':   li.item(),
        }


# ── Full pipeline loss ─────────────────────────────────────────────────────────

class FullPipelineLoss(nn.Module):
    """
    Combined loss for joint fine-tuning.
    L_total = L_EPE + L_text_aware

    No routing_loss (removed in v5 — it caused hard argmax collapse).
    No separate enh_loss — text_aware_loss IS the enhancement loss.
    """

    def __init__(self,
                 lambda_weather: float = 0.5,
                 lambda_time:    float = 0.3,
                 lambda_illum:   float = 0.3,
                 lam_text:       float = 2.0,
                 lam_ssim:       float = 0.3):
        super().__init__()
        self.epe_loss = EPELoss(lambda_weather, lambda_time, lambda_illum)
        self.lam_text = lam_text
        self.lam_ssim = lam_ssim

    def forward(self, model_out: dict, batch: dict, device):
        """
        model_out: output dict from AdaptiveEnhancementModel.forward()
        batch:     dataloader batch (may or may not contain clean_image)
        device:    torch.device
        """
        w_lbl = batch['weather_label'].to(device)
        t_lbl = batch['time_label'].to(device)
        i_lbl = batch['illum_label'].to(device)

        epe_total, epe_breakdown = self.epe_loss(
            model_out['weather_logits'],
            model_out['time_logits'],
            model_out['illum_logits'],
            w_lbl, t_lbl, i_lbl,
        )

        if 'clean_image' in batch:
            clean  = batch['clean_image'].to(device)
            enh_l  = text_aware_loss(
                model_out['enhanced'], clean,
                lam_text=self.lam_text,
                lam_ssim=self.lam_ssim,
            )
        else:
            # Unpaired: very small self-reconstruction signal
            enh_l = base_enh_loss(
                model_out['enhanced'],
                batch['image'].to(device),
                lam_ssim=self.lam_ssim,
            ) * 0.05

        total = epe_total + enh_l
        return total, {
            'total':   total.item(),
            'epe':     epe_total.item(),
            'enh':     enh_l.item(),
            **{f'epe_{k}': v for k, v in epe_breakdown.items() if k != 'total'},
        }
