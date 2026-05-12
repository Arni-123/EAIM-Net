"""
Adaptive Filter Bank (AFB) — EAIM-Net v5 Final

All 5 filters upgraded to deeper architectures as recommended:
  ChatGPT / project review: "filters too shallow — 2-3 conv layers is not enough"

  Filter 0  LowLightEnhancementModule   — Zero-DCE with deeper feature net (6 convs)
  Filter 1  DehazingModule              — Lightweight U-Net (5 levels, skip connections)
  Filter 2  RainRemovalModule           — ResNet blocks (4 residual blocks)
  Filter 3  IlluminationNormalizationModule — Deeper Retinex (4 convs + smoothness)
  Filter 4  GlareReductionModule        — U-Net with glare mask guidance

Each filter:
  - Accepts optional strength [B,1] for ESS-controlled blending
  - Accepts optional glare_map [B,1,H,W] for GlareReductionModule
  - Returns enhanced image clamped to [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Shared building blocks ────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Standard residual block with two conv layers."""
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch, affine=True),
        )

    def forward(self, x):
        return F.relu(x + self.net(x), inplace=True)


class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetUp(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.net = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.net(torch.cat([x, skip], dim=1))


# ─── Filter 0: LowLightEnhancementModule ──────────────────────────────────────

class LowLightEnhancementModule(nn.Module):
    """
    Zero-DCE style iterative curve with deeper 6-conv feature network.
    Deeper feature extraction gives better curve estimation for
    varying illumination levels (LOL dataset range 0.01-0.5 mean brightness).
    """

    def __init__(self, num_iterations: int = 8):
        super().__init__()
        self.num_iterations = num_iterations
        self.conv1 = nn.Conv2d(3,  32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv_out = nn.Conv2d(32, 3 * num_iterations, 3, padding=1)

        # Instance norm for training stability
        self.norm3 = nn.InstanceNorm2d(64, affine=True)
        self.norm4 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x: torch.Tensor, strength: torch.Tensor = None) -> torch.Tensor:
        f = F.relu(self.conv1(x))
        f = F.relu(self.conv2(f))
        f = F.relu(self.norm3(self.conv3(f)))
        f = F.relu(self.norm4(self.conv4(f)))
        f = F.relu(self.conv5(f))
        f = F.relu(self.conv6(f))
        curves = torch.tanh(self.conv_out(f))
        if strength is not None:
            curves = curves * strength.view(-1, 1, 1, 1)
        enhanced = x
        for i in range(self.num_iterations):
            curve = curves[:, i*3:(i+1)*3, :, :]
            enhanced = enhanced + curve * (enhanced.pow(2) - enhanced)
        return torch.clamp(enhanced, 0, 1)


# ─── Filter 1: DehazingModule ─────────────────────────────────────────────────

class DehazingModule(nn.Module):
    """
    Lightweight U-Net for haze/fog removal with skip connections.
    Upgrade from 3-layer encoder-decoder: skip connections preserve
    fine text details that were blurred in the old architecture.
    Gamma lift (x^0.45) applied before encoding for dark fog scenes.
    """

    GAMMA = 0.45

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc1 = UNetDown(32, 64)
        self.enc2 = UNetDown(64, 128)
        self.enc3 = UNetDown(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
        )

        # Decoder with skip connections
        self.dec2 = UNetUp(256, 128, 128)
        self.dec1 = UNetUp(128,  64,  64)
        self.dec0 = UNetUp( 64,  32,  32)

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, strength: torch.Tensor = None) -> torch.Tensor:
        sz = x.shape[2:]

        # Gamma lift for dark fog
        x_lift = x.clamp(1e-6, 1.0).pow(self.GAMMA)

        e0 = self.enc0(x_lift)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bottleneck(e3)
        d2 = self.dec2(b,  e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        out = self.out_conv(d0)

        # Inverse gamma
        out = out.clamp(1e-6, 1.0).pow(1.0 / self.GAMMA)
        if out.shape[2:] != sz:
            out = F.interpolate(out, size=sz, mode='bilinear', align_corners=False)
        if strength is not None:
            s = strength.view(-1, 1, 1, 1)
            out = x * (1 - s) + out * s
        return torch.clamp(out, 0, 1)


# ─── Filter 2: RainRemovalModule ──────────────────────────────────────────────

class RainRemovalModule(nn.Module):
    """
    Residual deraining with 4 residual blocks.
    Upgrade from 2 res blocks: handles Rain100H heavy rain streaks
    which require more non-linear capacity to separate from scene content.
    Shared for rain and light_snow (similar streak/particle patterns).
    """

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, strength: torch.Tensor = None) -> torch.Tensor:
        f        = self.head(x)
        f        = self.res_blocks(f)
        residual = self.tail(f)
        out      = x - residual
        if strength is not None:
            s   = strength.view(-1, 1, 1, 1)
            out = x * (1 - s) + out * s
        return torch.clamp(out, 0, 1)


# ─── Filter 3: IlluminationNormalizationModule ────────────────────────────────

class IlluminationNormalizationModule(nn.Module):
    """
    Retinex-based illumination estimation with 4 convolutions
    and a smoothness-encouraging architecture.
    Upgrade from 3 convs: produces smoother illumination maps
    that reduce the texture artefacts (high LPIPS) seen with
    coarser maps on clear/dusk scenes.
    """

    def __init__(self):
        super().__init__()
        self.illum_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),   # larger kernel for smooth map
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )
        # Noise-reduction branch (reduces texture artefacts)
        self.refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, strength: torch.Tensor = None) -> torch.Tensor:
        illum   = self.illum_net(x)                          # [B,1,H,W]
        refined = self.refine(x)                             # [B,3,H,W] denoised
        norm    = torch.clamp(refined / (illum + 1e-4), 0, 1)
        if strength is not None:
            s    = strength.view(-1, 1, 1, 1)
            norm = x * (1 - s) + norm * s
        return norm


# ─── Filter 4: GlareReductionModule ──────────────────────────────────────────

class GlareReductionModule(nn.Module):
    """
    U-Net glare reduction with SD1 glare-map guidance.

    Two operating modes:
      1. With glare_map [B,1,H,W]: uses the provided mask directly
         (from SD1 panel 3 during training, or estimated during inference)
      2. Without glare_map: estimates hotspot from luminance > 0.82

    The correction is applied ONLY inside the glare region.
    Non-glare pixels are unchanged regardless of strength.
    """

    HIGHLIGHT_THRESHOLD = 0.82
    STRENGTH_CAP        = 0.70

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.enc1 = UNetDown(32,  64)
        self.enc2 = UNetDown(64, 128)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(128), ResBlock(128))

        # Decoder
        self.dec1 = UNetUp(128, 64, 64)
        self.dec0 = UNetUp( 64, 32, 32)

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Tanh(),   # correction can go + or -
        )

    def _estimate_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Soft luminance mask when no glare_map is provided."""
        lum = (0.299 * x[:, 0:1] +
               0.587 * x[:, 1:2] +
               0.114 * x[:, 2:3])
        return torch.sigmoid(20.0 * (lum - self.HIGHLIGHT_THRESHOLD))

    def forward(self, x: torch.Tensor,
                strength: torch.Tensor = None,
                glare_map: torch.Tensor = None) -> torch.Tensor:
        # Encode
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        b  = self.bottleneck(e2)
        d1 = self.dec1(b,  e1)
        d0 = self.dec0(d1, e0)
        correction = self.out_conv(d0)   # [B,3,H,W] tanh correction

        # Spatial mask: use provided glare_map or estimate from luminance
        if glare_map is not None:
            mask = glare_map.to(x.device)
            if mask.shape[2:] != x.shape[2:]:
                mask = F.interpolate(mask, size=x.shape[2:],
                                     mode='bilinear', align_corners=False)
        else:
            mask = self._estimate_mask(x)

        # Apply correction only inside glare region
        # Positive correction reduces brightness; negative brightens shadows
        corrected = torch.clamp(x - mask * torch.relu(correction), 0, 1)

        if strength is not None:
            s         = torch.clamp(strength, max=self.STRENGTH_CAP).view(-1, 1, 1, 1)
            corrected = x * (1 - s) + corrected * s
        return corrected


# ─── Filter Bank ──────────────────────────────────────────────────────────────

class AdaptiveFilterBank(nn.Module):
    """
    Runs all 5 filters in parallel.
    Returns stacked outputs [B, 5, 3, H, W].
    ESS combiner then weights them.
    """

    def __init__(self):
        super().__init__()
        self.filters = nn.ModuleList([
            LowLightEnhancementModule(),       # 0 — LOL
            DehazingModule(),                   # 1 — RESIDE ITS
            RainRemovalModule(),                # 2 — Rain100L/H + DID-MDN
            IlluminationNormalizationModule(),  # 3 — WTT synthetic
            GlareReductionModule(),             # 4 — SD1
        ])
        self.num_filters = len(self.filters)

    def forward(self, x: torch.Tensor,
                filter_params: torch.Tensor,
                glare_map: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x             : [B, 3, H, W]
            filter_params : [B, num_filters, 8]
            glare_map     : [B, 1, H, W] optional — passed to GlareReductionModule
        Returns:
            [B, num_filters, 3, H, W]
        """
        outputs = []
        for i, module in enumerate(self.filters):
            strength = filter_params[:, i, 0:1]
            if i == 4 and glare_map is not None:
                out = module(x, strength, glare_map)
            else:
                out = module(x, strength)
            outputs.append(out)
        return torch.stack(outputs, dim=1)


if __name__ == '__main__':
    afb = AdaptiveFilterBank()
    x   = torch.rand(2, 3, 256, 256)
    fp  = torch.rand(2, 5, 8)
    out = afb(x, fp)
    print(f'AFB output      : {out.shape}')
    total = sum(p.numel() for p in afb.parameters())
    print(f'Total params    : {total:,}')

    # Per-filter param count
    for i, f in enumerate(afb.filters):
        n = sum(p.numel() for p in f.parameters())
        print(f'  Filter {i} ({f.__class__.__name__:<32}) : {n:>10,}')

    # Verify glare fix
    dark = torch.ones(1, 3, 64, 64) * 0.05
    out_dark = afb.filters[4](dark, torch.ones(1,1)*0.6)
    delta = (out_dark - dark).abs().mean().item()
    print(f'\nGlare on dark (should be ~0) : {delta:.6f}  {"OK" if delta<0.005 else "FAIL"}')

    # Verify dehazing
    dark_fog = torch.ones(1, 3, 64, 64) * 0.06
    out_fog  = afb.filters[1](dark_fog, torch.ones(1,1)*0.9)
    delta_fog = abs(out_fog.mean().item() - dark_fog.mean().item())
    print(f'Dehazing dark fog delta      : {delta_fog:.4f}  {"OK" if delta_fog>0.01 else "FAIL"}')
