"""
pretrain_filters.py — EAIM-Net v5 Final

Individual pre-training scripts for each AFB filter.
Run each function independently before joint fine-tuning.

Usage:
    python pretrain_filters.py --filter lowlight  --lol_root /path/LOL
    python pretrain_filters.py --filter dehazing  --reside_root /path/RESIDE_SOTS
    python pretrain_filters.py --filter rain      --rain100l /path/Rain100L \
                                                  --rain100h /path/Rain100H \
                                                  --didmdn   /path/DID-MDN/medium_dataset
    python pretrain_filters.py --filter glare     --sd1_root /path/SD1
    python pretrain_filters.py --filter illum     --wtt_root /path/weather_time_data
"""

import os
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
import torch.optim as optim
from torch.utils.data import (Dataset, DataLoader, ConcatDataset,
                               WeightedRandomSampler)
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity  as ssim_fn


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _gauss_win(size=11, sigma=1.5):
    c = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-c**2 / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g)


def ssim_loss(pred, gt, wsize=11):
    C1, C2 = 0.01**2, 0.03**2
    B, C, H, W = pred.shape
    win = _gauss_win(wsize).to(pred.device).expand(C, 1, wsize, wsize)
    pad = wsize // 2
    mu1 = TF.conv2d(pred, win, padding=pad, groups=C)
    mu2 = TF.conv2d(gt,   win, padding=pad, groups=C)
    s1  = TF.conv2d(pred*pred, win, padding=pad, groups=C) - mu1**2
    s2  = TF.conv2d(gt*gt,     win, padding=pad, groups=C) - mu2**2
    s12 = TF.conv2d(pred*gt,   win, padding=pad, groups=C) - mu1*mu2
    ssim_map = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))
    return 1.0 - ssim_map.mean()


def base_loss(pred, gt, lam_ssim=0.3):
    return nn.L1Loss()(pred, gt) + lam_ssim * ssim_loss(pred, gt)


def glare_masked_loss(pred, gt, glare_map, hot_weight=3.0, thresh=0.3):
    """Extra loss inside the glare hotspot region (SD1 panel 3)."""
    mask = (glare_map.to(pred.device) > thresh).float()
    return ((pred - gt).abs() * mask * hot_weight).mean()


def text_region_mask(gt, ksize=5, thresh=0.02):
    """Laplacian variance mask — approximates text/edge regions."""
    gray = 0.299*gt[:,0:1] + 0.587*gt[:,1:2] + 0.114*gt[:,2:3]
    lap_k = torch.tensor([[0.,-1.,0.],[-1.,4.,-1.],[0.,-1.,0.]],
                          dtype=torch.float32, device=gt.device).view(1,1,3,3)
    lap  = TF.conv2d(gray, lap_k, padding=1).abs()
    var  = TF.max_pool2d(lap, kernel_size=ksize, stride=1, padding=ksize//2)
    return (var > thresh).float()


def text_aware_loss(pred, gt, lam_text=2.0):
    base = base_loss(pred, gt)
    mask = text_region_mask(gt).detach()
    return base + ((pred - gt).abs() * mask * lam_text).mean()


def save_filter(module, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    p = os.path.join(save_dir, f'{name}.pth')
    torch.save(module.state_dict(), p)
    print(f'  saved: {p}')


def load_filter(module, save_dir, name, device):
    p = os.path.join(save_dir, f'{name}.pth')
    if os.path.exists(p):
        module.load_state_dict(torch.load(p, map_location=device))
        print(f'  loaded: {p}')
        return True
    return False


def _paired_aug(inp, gt, size):
    """Paired random crop + horizontal flip."""
    w, h = inp.size
    if w > size and h > size:
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        inp = inp.crop((x, y, x+size, y+size))
        gt  = gt.crop((x, y, x+size, y+size))
    else:
        inp = inp.resize((size, size), Image.LANCZOS)
        gt  = gt.resize((size, size), Image.LANCZOS)
    if random.random() > 0.5:
        inp = inp.transpose(Image.FLIP_LEFT_RIGHT)
        gt  = gt.transpose(Image.FLIP_LEFT_RIGHT)
    return inp, gt


def _to_square(pil, size):
    return pil.resize((size, size), Image.LANCZOS)


# ── Datasets ───────────────────────────────────────────────────────────────────

class LOLDataset(Dataset):
    """LOL: our485/low + our485/high (train) / eval15/low + eval15/high (val)."""

    def __init__(self, inp_dir, tgt_dir, size=256, augment=False):
        self.pairs, self.size, self.augment = [], size, augment
        exts = ('.jpg', '.jpeg', '.png')
        for f in sorted(os.listdir(inp_dir)):
            if not f.lower().endswith(exts):
                continue
            stem = os.path.splitext(f)[0]
            gt   = None
            for ext in exts:
                c = os.path.join(tgt_dir, stem + ext)
                if os.path.exists(c):
                    gt = c
                    break
            if gt:
                self.pairs.append((os.path.join(inp_dir, f), gt))
        print(f'  LOL: {len(self.pairs)} pairs  ({inp_dir})')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, gp = self.pairs[idx]
        inp = Image.open(ip).convert('RGB')
        gt  = Image.open(gp).convert('RGB')
        if self.augment:
            inp, gt = _paired_aug(inp, gt, self.size)
        else:
            inp = _to_square(inp, self.size)
            gt  = _to_square(gt,  self.size)
        to_t = transforms.ToTensor()
        return {'input': to_t(inp), 'target': to_t(gt),
                'name': os.path.basename(ip)}


class ResideDataset(Dataset):
    """
    RESIDE ITS or SOTS.
    ITS structure: hazy/ + clear/  (recommended — 13k pairs)
    SOTS hazy filenames: 0001_0.8_0.2.jpg — GT stem is 0001
    """

    def __init__(self, hazy_dir, clear_dir, size=256, augment=False):
        self.pairs, self.size, self.augment = [], size, augment
        exts = ('.jpg', '.jpeg', '.png')
        for hf in sorted(os.listdir(hazy_dir)):
            if not hf.lower().endswith(exts):
                continue
            stem    = os.path.splitext(hf)[0]
            gt_stem = stem.split('_')[0] if '_' in stem else stem
            gt = None
            for ext in exts:
                for candidate in [gt_stem, stem]:
                    c = os.path.join(clear_dir, candidate + ext)
                    if os.path.exists(c):
                        gt = c
                        break
                if gt:
                    break
            if gt:
                self.pairs.append((os.path.join(hazy_dir, hf), gt))
        print(f'  RESIDE: {len(self.pairs)} pairs')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, gp = self.pairs[idx]
        inp = Image.open(ip).convert('RGB')
        gt  = Image.open(gp).convert('RGB')
        if self.augment:
            inp, gt = _paired_aug(inp, gt, self.size)
        else:
            inp = _to_square(inp, self.size)
            gt  = _to_square(gt,  self.size)
        to_t = transforms.ToTensor()
        return {'input': to_t(inp), 'target': to_t(gt)}


class RainDataset(Dataset):
    """
    Generic rain dataset.
    Supports Rain100L, Rain100H: train/rain + train/norain
    Supports DID-MDN:            train/rain + train/clear
    """

    def __init__(self, rain_dir, clean_dir, size=256, augment=False, label='rain'):
        self.pairs, self.size, self.augment = [], size, augment
        exts = ('.jpg', '.jpeg', '.png')
        for f in sorted(os.listdir(rain_dir)):
            if not f.lower().endswith(exts):
                continue
            stem = os.path.splitext(f)[0]
            gt   = None
            for ext in exts:
                c = os.path.join(clean_dir, stem + ext)
                if os.path.exists(c):
                    gt = c
                    break
            if gt:
                self.pairs.append((os.path.join(rain_dir, f), gt))
        print(f'  {label}: {len(self.pairs)} pairs')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, gp = self.pairs[idx]
        inp = Image.open(ip).convert('RGB')
        gt  = Image.open(gp).convert('RGB')
        if self.augment:
            inp, gt = _paired_aug(inp, gt, self.size)
        else:
            inp = _to_square(inp, self.size)
            gt  = _to_square(gt,  self.size)
        to_t = transforms.ToTensor()
        return {'input': to_t(inp), 'target': to_t(gt)}


class SD1StripDataset(Dataset):
    """
    SD1 glare dataset — 3-panel wide images.
    Strip layout: [clean_GT | glare_degraded | glare_map]
      Panel 0 (left 1/3)   = clean GT
      Panel 1 (middle 1/3) = glare-degraded input
      Panel 2 (right 1/3)  = glare intensity map (Gaussian blob, grayscale)
    """

    def __init__(self, root, size=256, augment=False):
        self.samples, self.size, self.augment = [], size, augment
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        for f in sorted(os.listdir(root)):
            if f.lower().endswith(exts):
                self.samples.append(os.path.join(root, f))
        print(f'  SD1 strip: {len(self.samples)} images  ({root})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img      = Image.open(self.samples[idx]).convert('RGB')
        w, h     = img.size
        pw       = w // 3
        clean    = img.crop((0,    0, pw,   h))   # panel 0: GT
        glare_img= img.crop((pw,   0, pw*2, h))   # panel 1: degraded
        gmap     = img.crop((pw*2, 0, w,    h))   # panel 2: glare map

        clean     = _to_square(clean,     self.size)
        glare_img = _to_square(glare_img, self.size)
        gmap      = _to_square(gmap,      self.size)

        if self.augment and random.random() > 0.5:
            clean     = clean.transpose(Image.FLIP_LEFT_RIGHT)
            glare_img = glare_img.transpose(Image.FLIP_LEFT_RIGHT)
            gmap      = gmap.transpose(Image.FLIP_LEFT_RIGHT)

        to_t = transforms.ToTensor()
        return {
            'input':     to_t(glare_img),
            'target':    to_t(clean),
            'glare_map': to_t(gmap.convert('L')),   # [1,H,W]
            'name':      os.path.basename(self.samples[idx]),
        }


class SD1TestDataset(Dataset):
    """SD1 test split — separate *_gt.* and *_light.* files."""

    def __init__(self, root, size=256):
        self.pairs, self.size = [], size
        exts = ('.jpg', '.jpeg', '.png')
        gt_dir = os.path.join(root, 'gt')
        lt_dir = os.path.join(root, 'light')
        if os.path.isdir(gt_dir) and os.path.isdir(lt_dir):
            for f in sorted(os.listdir(gt_dir)):
                if not f.lower().endswith(exts):
                    continue
                lf = os.path.join(lt_dir, f)
                if os.path.exists(lf):
                    self.pairs.append((lf, os.path.join(gt_dir, f)))
        else:
            for gf in sorted(glob.glob(os.path.join(root, '*_gt.*'))):
                stem = os.path.basename(gf).split('_gt')[0]
                for ext in exts:
                    lf = os.path.join(root, stem + '_light' + ext)
                    if os.path.exists(lf):
                        self.pairs.append((lf, gf))
                        break
        print(f'  SD1 test: {len(self.pairs)} pairs')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lp, gp = self.pairs[idx]
        inp = Image.open(lp).convert('RGB').resize((self.size, self.size), Image.LANCZOS)
        gt  = Image.open(gp).convert('RGB').resize((self.size, self.size), Image.LANCZOS)
        to_t = transforms.ToTensor()
        return {'input': to_t(inp), 'target': to_t(gt),
                'name': os.path.basename(lp)}


class WTTPairedDataset(Dataset):
    """Synthetic Weather-Time-Text paired dataset."""

    def __init__(self, root, split='train', size=256):
        self.pairs, self.size = [], size
        self.augment = (split == 'train')
        split_dir = os.path.join(root, split)
        img_dir   = os.path.join(split_dir, 'images')
        cln_dir   = os.path.join(split_dir, 'clean_images')
        if not os.path.isdir(img_dir):
            return
        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            cp = os.path.join(cln_dir, f)
            if os.path.exists(cp):
                self.pairs.append((os.path.join(img_dir, f), cp))
        print(f'  WTT {split}: {len(self.pairs)} pairs')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, gp = self.pairs[idx]
        inp = Image.open(ip).convert('RGB')
        gt  = Image.open(gp).convert('RGB')
        if self.augment:
            inp, gt = _paired_aug(inp, gt, self.size)
        else:
            inp = _to_square(inp, self.size)
            gt  = _to_square(gt,  self.size)
        to_t = transforms.ToTensor()
        return {'input': to_t(inp), 'target': to_t(gt)}


# ── Dir finders ────────────────────────────────────────────────────────────────

def _lol_dirs(root, split):
    order = [('our485', 'low', 'high'), ('train', 'low', 'high')]
    if split in ('val', 'test'):
        order = [('eval15', 'low', 'high'), ('test', 'low', 'high')]
    for sub, inp_name, tgt_name in order:
        i = os.path.join(root, sub, inp_name)
        t = os.path.join(root, sub, tgt_name)
        if os.path.isdir(i) and os.path.isdir(t):
            return i, t
    raise FileNotFoundError(f'LOL {split} not found under {root}')


def _rain_dirs(root, split):
    base = os.path.join(root, split) if os.path.isdir(os.path.join(root, split)) else root
    for rn in ['rain']:
        for cn in ['norain', 'clean', 'clear', 'gt']:
            rd = os.path.join(base, rn)
            cd = os.path.join(base, cn)
            if os.path.isdir(rd) and os.path.isdir(cd):
                return rd, cd
    raise FileNotFoundError(f'rain dirs not found in {base}')


# ── Generic trainer ────────────────────────────────────────────────────────────

def train_filter(module, tr_ld, va_ld, name, save_dir,
                 device, n_epochs=25, lr=2e-4,
                 loss_fn=None, resume=True):
    """
    Train a single AFB filter module independently.

    loss_fn signature: loss_fn(pred, gt, batch) -> scalar tensor
    If None, uses base_loss (L1 + SSIM).
    """
    module = module.to(device)
    if resume:
        load_filter(module, save_dir, f'{name}_best', device)

    opt = optim.Adam(module.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=1, eta_min=1e-6)

    best_psnr = 0.0
    log_lines = []

    print(f'\nPre-training: {name}  ({n_epochs} epochs)')
    print(f'  train={len(tr_ld)} batches  val={len(va_ld)} batches')
    hdr = f'  {"Ep":>4}  {"loss":>9}  {"psnr":>9}  {"ssim":>8}  {"lr":>9}'
    print(hdr)
    print('  ' + '-' * 48)

    for ep in range(n_epochs):
        module.train()
        total_loss = 0.0

        for batch in tqdm(tr_ld, desc=f'Ep{ep:02d}', leave=False):
            inp = batch['input'].to(device)
            tgt = batch['target'].to(device)
            st  = torch.ones(inp.shape[0], 1).to(device)

            if name == 'glare' and 'glare_map' in batch:
                pred = module(inp, st, batch['glare_map'].to(device))
            else:
                pred = module(inp, st)

            if loss_fn is not None:
                loss = loss_fn(pred, tgt, batch)
            else:
                loss = base_loss(pred, tgt)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        sch.step(ep + 1)

        # Validation
        module.eval()
        psnrs, ssims = [], []
        with torch.no_grad():
            for batch in va_ld:
                inp  = batch['input'].to(device)
                tgt  = batch['target'].to(device)
                st   = torch.ones(inp.shape[0], 1).to(device)
                if name == 'glare' and 'glare_map' in batch:
                    pred = module(inp, st, batch['glare_map'].to(device))
                else:
                    pred = module(inp, st)
                pred = torch.clamp(pred, 0, 1)
                for i in range(inp.shape[0]):
                    pn = np.clip(pred[i].cpu().numpy().transpose(1,2,0), 0, 1)
                    gn = np.clip(tgt[i].cpu().numpy().transpose(1,2,0),  0, 1)
                    psnrs.append(psnr_fn(gn, pn, data_range=1.0))
                    ssims.append(ssim_fn(gn, pn, data_range=1.0, channel_axis=2))

        vp = float(np.mean(psnrs))
        vs = float(np.mean(ssims))
        lr_now = opt.param_groups[0]['lr']
        avg_l  = total_loss / len(tr_ld)

        line = f'  {ep:>4}  {avg_l:>9.4f}  {vp:>9.4f}  {vs:>8.4f}  {lr_now:>9.2e}'
        print(line)
        log_lines.append(line)

        if vp > best_psnr:
            best_psnr = vp
            save_filter(module, save_dir, f'{name}_best')
            print(f'    NEW BEST PSNR={best_psnr:.4f}')

        if (ep + 1) % 5 == 0:
            save_filter(module, save_dir, f'{name}_ep{ep:03d}')

    log_path = os.path.join(save_dir, f'{name}_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f'\n  Done. Best PSNR={best_psnr:.4f}  Log: {log_path}')
    return module


# ── Per-filter entry points ────────────────────────────────────────────────────

def pretrain_lowlight(args, device):
    from afb_module import LowLightEnhancementModule
    tr_i, tr_t = _lol_dirs(args.lol_root, 'train')
    va_i, va_t = _lol_dirs(args.lol_root, 'val')
    tr_ds = LOLDataset(tr_i, tr_t, args.img_size, augment=True)
    va_ds = LOLDataset(va_i, va_t, args.img_size, augment=False)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=2, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)
    module = LowLightEnhancementModule()
    train_filter(module, tr_ld, va_ld, 'lowlight', args.save_dir,
                 device, n_epochs=args.epochs, lr=args.lr)


def pretrain_dehazing(args, device):
    from afb_module import DehazingModule
    hazy_d  = os.path.join(args.reside_root, 'hazy')
    clear_d = os.path.join(args.reside_root, 'clear')
    if not os.path.isdir(hazy_d):
        hazy_d  = os.path.join(args.reside_root, 'indoor', 'hazy')
        clear_d = os.path.join(args.reside_root, 'indoor', 'clear')

    full_ds = ResideDataset(hazy_d, clear_d, args.img_size, augment=True)
    n_va  = max(1, len(full_ds) // 5)
    n_tr  = len(full_ds) - n_va
    tr_ds, va_ds = torch.utils.data.random_split(
        full_ds, [n_tr, n_va],
        generator=torch.Generator().manual_seed(42))
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=2, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)
    module = DehazingModule()
    train_filter(module, tr_ld, va_ld, 'dehazing', args.save_dir,
                 device, n_epochs=args.epochs, lr=args.lr)


def pretrain_rain(args, device):
    from afb_module import RainRemovalModule
    datasets, weights = [], []

    if args.rain100l:
        try:
            rd, cd = _rain_dirs(args.rain100l, 'train')
            ds = RainDataset(rd, cd, args.img_size, augment=True, label='Rain100L')
            datasets.append(ds)
            weights.extend([1.0] * len(ds))
        except Exception as e:
            print(f'Rain100L skipped: {e}')

    if args.didmdn:
        try:
            rd = os.path.join(args.didmdn, 'train', 'rain')
            cd = os.path.join(args.didmdn, 'train', 'clear')
            ds = RainDataset(rd, cd, args.img_size, augment=True, label='DID-MDN')
            datasets.append(ds)
            weights.extend([1.2] * len(ds))
        except Exception as e:
            print(f'DID-MDN skipped: {e}')

    if args.rain100h:
        try:
            rd, cd = _rain_dirs(args.rain100h, 'train')
            ds = RainDataset(rd, cd, args.img_size, augment=True, label='Rain100H')
            datasets.append(ds)
            weights.extend([1.5] * len(ds))
        except Exception as e:
            print(f'Rain100H skipped: {e}')

    assert datasets, 'No rain datasets found!'
    combined = ConcatDataset(datasets)
    sampler  = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float), len(combined), replacement=True)

    # Validation: Rain100L test
    try:
        vrd, vcd = _rain_dirs(args.rain100l or args.rain100h, 'test')
    except Exception:
        vrd, vcd = _rain_dirs(args.rain100l or args.rain100h, 'train')
    va_ds = RainDataset(vrd, vcd, args.img_size, augment=False, label='val')

    tr_ld = DataLoader(combined, batch_size=args.batch_size, sampler=sampler,
                       num_workers=2, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)

    module = RainRemovalModule()
    train_filter(module, tr_ld, va_ld, 'rainremoval', args.save_dir,
                 device, n_epochs=args.epochs, lr=args.lr)


def pretrain_glare(args, device):
    from afb_module import GlareReductionModule

    def _find_subdir(root, names):
        for n in names:
            p = os.path.join(root, n)
            if os.path.isdir(p):
                return p
        return root

    train_dir = _find_subdir(args.sd1_root, ['train', 'Train', 'training'])
    val_dir   = _find_subdir(args.sd1_root, ['val', 'Val', 'validation'])
    test_dir  = _find_subdir(args.sd1_root, ['test', 'Test'])

    tr_ds = SD1StripDataset(train_dir, args.img_size, augment=True)

    if val_dir != args.sd1_root and val_dir != train_dir:
        va_ds = SD1StripDataset(val_dir, args.img_size, augment=False)
    elif test_dir != args.sd1_root:
        va_ds = SD1TestDataset(test_dir, args.img_size)
    else:
        n_va = max(1, len(tr_ds) // 5)
        n_tr = len(tr_ds) - n_va
        tr_ds, va_ds = torch.utils.data.random_split(
            tr_ds, [n_tr, n_va],
            generator=torch.Generator().manual_seed(42))
        print(f'  80/20 split: {n_tr} train / {n_va} val')

    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=2, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)

    def glare_loss_fn(pred, gt, batch):
        base = base_loss(pred, gt)
        if 'glare_map' in batch:
            base = base + glare_masked_loss(pred, gt, batch['glare_map'])
        return base

    module = GlareReductionModule()
    train_filter(module, tr_ld, va_ld, 'glare', args.save_dir,
                 device, n_epochs=args.epochs, lr=args.lr,
                 loss_fn=glare_loss_fn)


def pretrain_illum(args, device):
    from afb_module import IlluminationNormalizationModule
    tr_ds = WTTPairedDataset(args.wtt_root, 'train', args.img_size)
    va_ds = WTTPairedDataset(args.wtt_root, 'val',   args.img_size)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=2, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)
    module = IlluminationNormalizationModule()
    train_filter(module, tr_ld, va_ld, 'illumnorm', args.save_dir,
                 device, n_epochs=args.epochs, lr=args.lr)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='EAIM-Net v5 Filter Pre-training')
    parser.add_argument('--filter',     required=True,
                        choices=['lowlight','dehazing','rain','glare','illum'])
    parser.add_argument('--save_dir',   default='./pretrained_filters')
    parser.add_argument('--epochs',     type=int,   default=25)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--img_size',   type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=2e-4)

    # Dataset roots
    parser.add_argument('--lol_root',    default=None)
    parser.add_argument('--reside_root', default=None)
    parser.add_argument('--rain100l',    default=None)
    parser.add_argument('--rain100h',    default=None)
    parser.add_argument('--didmdn',      default=None)
    parser.add_argument('--sd1_root',    default=None)
    parser.add_argument('--wtt_root',    default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    dispatch = {
        'lowlight': pretrain_lowlight,
        'dehazing': pretrain_dehazing,
        'rain':     pretrain_rain,
        'glare':    pretrain_glare,
        'illum':    pretrain_illum,
    }
    dispatch[args.filter](args, device)


if __name__ == '__main__':
    main()
