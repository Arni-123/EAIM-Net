"""
dataset_real_pairs.py  -  EAIM-Net v5 Final

v5.1 update: All degradation datasets now included in fine-tuning.
  Previously only LOL + Rain100L + WTT were used. This caused ESS to
  learn correct blending only for low-light and rain conditions while
  Dehazing, Glare, and IllumNorm filters were underweighted.

  Now includes:
    LOL       (low-light)       weight=3.0  - small dataset, needs oversampling
    RESIDE    (haze)            weight=1.5  - new in fine-tuning
    Rain100L  (light rain)      weight=1.0
    Rain100H  (heavy rain)      weight=1.5  - new in fine-tuning
    DID-MDN   (medium rain)     weight=1.2  - new in fine-tuning
    WTT       (all conditions)  weight=1.0  subset=0.20

  SD1 (glare) is excluded from fine-tuning because:
    - It uses 3-panel strip format which requires special loading
    - The GlareReductionModule is already well pre-trained
    - SD1 is 11 GB and would dominate disk usage
    - Glare signal comes from WTT synthetic glare images instead

  synthetic_subset_frac: use fraction of WTT to keep epochs fast.
    0.20 -> ~1760 WTT images -> ~8 min/epoch
    0.50 -> ~4400 WTT images -> ~18 min/epoch

  lol_eval_root: separate LOL eval dir for val/test (eval15, 15 pairs)
    prevents train/val leakage.
"""

import os, random
import torch
from torch.utils.data import (DataLoader, ConcatDataset,
                               WeightedRandomSampler, Subset)
from typing import Optional, Dict
import platform
from dataset import EnvironmentalTextDataset, RealPairDataset, safe_collate


def _count_imgs(d):
    if not d or not os.path.isdir(d): return 0
    n = 0
    for _,_,fs in os.walk(d):
        n += sum(1 for f in fs if f.lower().endswith(('.jpg','.jpeg','.png')))
    return n


def _subset_ds(ds, frac):
    if frac >= 1.0 or len(ds) == 0: return ds
    k = max(1, int(len(ds) * frac))
    return Subset(ds, random.sample(range(len(ds)), k))


def _find_reside_dirs(root):
    """Walk to find hazy/ and clear/ inside RESIDE root."""
    for r, dirs, _ in os.walk(root):
        if r[len(root):].count(os.sep) > 3: continue
        dc = {d.lower(): d for d in dirs}
        if 'hazy' in dc and 'clear' in dc:
            return os.path.join(r, dc['hazy']), os.path.join(r, dc['clear'])
    return None, None


class ResideITSDataset(torch.utils.data.Dataset):
    """
    RESIDE-ITS: hazy/1000_10_0.74905.png -> clear/1000.png
    Stem before first underscore = GT image ID.
    1399 clear images x 10 hazy = 13,990 pairs.
    """
    def __init__(self, hazy_dir, clear_dir, size=512, augment=False):
        self.pairs = []; self.size = size; self.aug = augment
        exts = ('.jpg','.jpeg','.png')
        clear_lut = {}
        for f in os.listdir(clear_dir):
            if f.lower().endswith(exts):
                clear_lut[os.path.splitext(f)[0]] = os.path.join(clear_dir, f)
        for f in sorted(os.listdir(hazy_dir)):
            if not f.lower().endswith(exts): continue
            gt_stem = os.path.splitext(f)[0].split('_')[0]
            if gt_stem in clear_lut:
                self.pairs.append((os.path.join(hazy_dir, f), clear_lut[gt_stem]))
        # Labels: fog/day/medium
        self._wl, self._tl, self._il = 2, 1, 1  # fog, day, medium
        print(f'  RESIDE-ITS: {len(self.pairs)} pairs')

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        from PIL import Image
        from torchvision import transforms
        ip, gp = self.pairs[idx]
        inp = Image.open(ip).convert('RGB')
        gt  = Image.open(gp).convert('RGB')
        if self.aug:
            iw, ih = inp.size
            if iw > self.size and ih > self.size:
                x = random.randint(0, iw-self.size)
                y = random.randint(0, ih-self.size)
                inp = inp.crop((x,y,x+self.size,y+self.size))
                gt  = gt.crop( (x,y,x+self.size,y+self.size))
            else:
                inp = inp.resize((self.size,self.size))
                gt  = gt.resize( (self.size,self.size))
            if random.random() > 0.5:
                from PIL import Image as _I
                inp = inp.transpose(_I.FLIP_LEFT_RIGHT)
                gt  = gt.transpose( _I.FLIP_LEFT_RIGHT)
        else:
            inp = inp.resize((self.size,self.size))
            gt  = gt.resize( (self.size,self.size))
        tt = transforms.ToTensor()
        return {
            'image':         tt(inp),
            'clean_image':   tt(gt),
            'weather_label': torch.tensor(self._wl, dtype=torch.long),
            'time_label':    torch.tensor(self._tl, dtype=torch.long),
            'illum_label':   torch.tensor(self._il, dtype=torch.long),
            'image_name':    os.path.basename(ip),
            'source':        'RESIDE',
        }


class RainPairDataset(torch.utils.data.Dataset):
    """
    Generic rain dataset supporting all naming conventions:
      Rain100L: norain-100x2.png -> norain-100.png  (strip x2)
      Rain100H: norain-1089.png  -> norain-1089.png (exact match)
      DID-MDN:  1.jpg            -> 1.jpg           (exact match)
    """
    def __init__(self, rain_dir, clean_dir, size=512, augment=False, label='rain'):
        self.pairs = []; self.size = size; self.aug = augment
        exts = ('.jpg','.jpeg','.png')
        clean_lut = {}
        for f in os.listdir(clean_dir):
            if f.lower().endswith(exts):
                clean_lut[os.path.splitext(f)[0]] = os.path.join(clean_dir, f)
        for f in sorted(os.listdir(rain_dir)):
            if not f.lower().endswith(exts): continue
            stem = os.path.splitext(f)[0]; gt = None
            if stem in clean_lut: gt = clean_lut[stem]
            elif stem.endswith('x2') and stem[:-2] in clean_lut:
                gt = clean_lut[stem[:-2]]
            elif stem.endswith('_x2') and stem[:-3] in clean_lut:
                gt = clean_lut[stem[:-3]]
            else:
                import re
                s2 = re.sub(r'x[0-9]+$', '', stem)
                if s2 in clean_lut: gt = clean_lut[s2]
            if gt: self.pairs.append((os.path.join(rain_dir, f), gt))
        print(f'  {label}: {len(self.pairs)} pairs')
        # Labels: rain/day/medium
        self._wl, self._tl, self._il = 1, 1, 1

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        from PIL import Image
        from torchvision import transforms
        ip, gp = self.pairs[idx]
        inp = Image.open(ip).convert('RGB')
        gt  = Image.open(gp).convert('RGB')
        if self.aug:
            iw, ih = inp.size
            if iw > self.size and ih > self.size:
                x = random.randint(0, iw-self.size)
                y = random.randint(0, ih-self.size)
                inp = inp.crop((x,y,x+self.size,y+self.size))
                gt  = gt.crop( (x,y,x+self.size,y+self.size))
            else:
                inp = inp.resize((self.size,self.size))
                gt  = gt.resize( (self.size,self.size))
            if random.random() > 0.5:
                from PIL import Image as _I
                inp = inp.transpose(_I.FLIP_LEFT_RIGHT)
                gt  = gt.transpose( _I.FLIP_LEFT_RIGHT)
        else:
            inp = inp.resize((self.size,self.size))
            gt  = gt.resize( (self.size,self.size))
        tt = transforms.ToTensor()
        return {
            'image':         tt(inp),
            'clean_image':   tt(gt),
            'weather_label': torch.tensor(self._wl, dtype=torch.long),
            'time_label':    torch.tensor(self._tl, dtype=torch.long),
            'illum_label':   torch.tensor(self._il, dtype=torch.long),
            'image_name':    os.path.basename(ip),
            'source':        'Rain',
        }


def _find_rain_dirs(root):
    for r, dirs, _ in os.walk(root):
        if r[len(root):].count(os.sep) > 3: continue
        dc = {d.lower(): d for d in dirs}
        if 'rain' in dc:
            for cn in ['norain','clean','clear','gt']:
                if cn in dc:
                    return os.path.join(r, dc['rain']), os.path.join(r, dc[cn])
    return None, None


def debug_dataset_roots(**kwargs):
    print('\n' + '='*60)
    print('  FINE-TUNING DATASET VERIFICATION')
    print('='*60)
    for name, root in kwargs.items():
        if not root:
            print(f'  {"--":<16}: {name} not provided'); continue
        if not os.path.isdir(root):
            print(f'  {"MISSING":<16}: {name}  {root}'); continue
        n = _count_imgs(root)
        print(f'  {"OK" if n>0 else "EMPTY":<16}: {name}  {n:,} images')
    print('='*60)


def create_combined_dataloaders(
    # Core datasets
    synthetic_data_root:    Optional[str] = None,
    lol_root:               Optional[str] = None,
    rain100l_root:          Optional[str] = None,
    rain100h_root:          Optional[str] = None,
    # All degradations (v5.1)
    reside_root:            Optional[str] = None,
    didmdn_root:            Optional[str] = None,
    sd1_root:               Optional[str] = None,
    # Split handling
    lol_eval_root:          Optional[str] = None,
    # Loader settings
    batch_size:             int   = 4,
    num_workers:            int   = 2,
    image_size:             int   = 512,
    # Dataset weights
    lol_weight:             float = 3.0,
    rain100l_weight:        float = 1.0,
    rain100h_weight:        float = 1.5,
    reside_weight:          float = 1.5,
    didmdn_weight:          float = 1.2,
    sd1_weight:             float = 2.0,
    # Subset fractions (control epoch time)
    synthetic_subset_frac:  float = 1.0,   # WTT
    reside_subset_frac:     float = 1.0,   # RESIDE-ITS
    sd1_subset_frac:        float = 1.0,   # SD1
    debug:                  bool  = True,
) -> Dict[str, DataLoader]:

    pin = torch.cuda.is_available()
    if platform.system() == 'Windows': num_workers = 0

    lol_val = lol_eval_root if (lol_eval_root and
               os.path.isdir(lol_eval_root)) else lol_root

    loaders = {}

    for split in ['train', 'val', 'test']:
        datasets = []; weights = []

        # ── WTT Synthetic ──────────────────────────────────────────
        if synthetic_data_root:
            sd = os.path.join(synthetic_data_root, split)
            if not os.path.isdir(sd): sd = synthetic_data_root
            if _count_imgs(sd) > 0:
                ds = EnvironmentalTextDataset(data_root=sd, split=split,
                     image_size=image_size, paired=True,
                     augment=(split=='train'))
                if len(ds) > 0:
                    if split=='train' and synthetic_subset_frac < 1.0:
                        ds = _subset_ds(ds, synthetic_subset_frac)
                    if debug: print(f'  [{split}] WTT          : {len(ds):6,}')
                    datasets.append(ds); weights.extend([1.0]*len(ds))

        # ── LOL ────────────────────────────────────────────────────
        _lol = lol_root if split=='train' else lol_val
        if _lol and _count_imgs(_lol) > 0:
            ds = RealPairDataset(root=_lol, split=split,
                 image_size=image_size, augment=(split=='train'),
                 dataset_type='lol', debug=debug)
            if len(ds) > 0:
                datasets.append(ds); weights.extend([lol_weight]*len(ds))
                if debug: print(f'  [{split}] LOL          : {len(ds):6,}  w={lol_weight}')

        # ── RESIDE-ITS (NEW in fine-tuning) ────────────────────────
        if reside_root and _count_imgs(reside_root) > 0:
            hazy_d, clear_d = _find_reside_dirs(reside_root)
            if hazy_d and clear_d:
                ds = ResideITSDataset(hazy_d, clear_d,
                     size=image_size, augment=(split=='train'))
                if len(ds) > 0:
                    if split == 'train' and reside_subset_frac < 1.0:
                        ds = _subset_ds(ds, reside_subset_frac)
                    elif split != 'train':
                        ds = _subset_ds(ds, 0.10)
                    datasets.append(ds); weights.extend([reside_weight]*len(ds))
                    if debug: print(f'  [{split}] RESIDE       : {len(ds):6,}  w={reside_weight}')

        # ── Rain100L ───────────────────────────────────────────────
        if rain100l_root and _count_imgs(rain100l_root) > 0:
            rd, cd = _find_rain_dirs(rain100l_root)
            if rd and cd:
                ds = RainPairDataset(rd, cd, size=image_size,
                     augment=(split=='train'), label='Rain100L')
                if len(ds) > 0:
                    datasets.append(ds); weights.extend([rain100l_weight]*len(ds))
                    if debug: print(f'  [{split}] Rain100L     : {len(ds):6,}  w={rain100l_weight}')

        # ── Rain100H (NEW in fine-tuning) ──────────────────────────
        if rain100h_root and _count_imgs(rain100h_root) > 0:
            rd, cd = _find_rain_dirs(rain100h_root)
            if rd and cd:
                ds = RainPairDataset(rd, cd, size=image_size,
                     augment=(split=='train'), label='Rain100H')
                if len(ds) > 0:
                    datasets.append(ds); weights.extend([rain100h_weight]*len(ds))
                    if debug: print(f'  [{split}] Rain100H     : {len(ds):6,}  w={rain100h_weight}')

        # ── DID-MDN (NEW in fine-tuning) ───────────────────────────
        if didmdn_root and _count_imgs(didmdn_root) > 0:
            # DID-MDN: medium_dataset/train/rain + train/clear
            _split_d = 'train' if split == 'train' else 'test'
            rd  = os.path.join(didmdn_root, _split_d, 'rain')
            _cd_names = ['clear', 'clean']
            cd = None
            for cn in _cd_names:
                _p = os.path.join(didmdn_root, _split_d, cn)
                if os.path.isdir(_p): cd = _p; break
            if rd and cd and os.path.isdir(rd) and os.path.isdir(cd):
                ds = RainPairDataset(rd, cd, size=image_size,
                     augment=(split=='train'), label='DID-MDN')
                if len(ds) > 0:
                    datasets.append(ds); weights.extend([didmdn_weight]*len(ds))
                    if debug: print(f'  [{split}] DID-MDN      : {len(ds):6,}  w={didmdn_weight}')

        # ── SD1 Glare (NEW in fine-tuning) ────────────────────────────
        if sd1_root and _count_imgs(sd1_root) > 0:
            try:
                ds = SD1FineTuneDataset(sd1_root, size=image_size,
                     augment=(split=='train'),
                     subset_frac=sd1_subset_frac if split=='train' else 0.10)
                if len(ds) > 0:
                    datasets.append(ds); weights.extend([sd1_weight]*len(ds))
                    if debug: print(f'  [{split}] SD1 (glare)  : {len(ds):6,}  w={sd1_weight}')
            except Exception as _se:
                if debug: print(f'  [{split}] SD1 skipped: {_se}')

        if not datasets:
            if debug: print(f'  [{split}] No datasets')
            continue

        combined = ConcatDataset(datasets)

        if split == 'train' and weights:
            sampler = WeightedRandomSampler(
                torch.tensor(weights, dtype=torch.float),
                len(combined), replacement=True)
            loaders[split] = DataLoader(combined,
                batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=pin,
                drop_last=True, collate_fn=safe_collate)
        else:
            loaders[split] = DataLoader(combined,
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin,
                drop_last=False, collate_fn=safe_collate)

        if debug:
            print(f'  [{split}] TOTAL        : {len(combined):6,}')

    return loaders


class SD1FineTuneDataset(torch.utils.data.Dataset):
    """
    SD1 dataset adapted for fine-tuning (not pre-training).

    At pre-training time SD1 uses 3-panel strips [GT|glare|map].
    For fine-tuning we use the same strips but only need input+GT,
    ignoring the glare map panel (the GlareReductionModule handles
    that internally using luminance thresholding at inference).

    Alternatively if test pairs exist (*_gt.* + *_light.*) those
    are used directly as paired data.

    Labels: Glare / Day / High  (weather=4, time=1, illum=2)
    """
    def __init__(self, root, size=512, augment=False, subset_frac=1.0):
        self.pairs = []; self.size = size; self.aug = augment
        exts = ('.jpg','.jpeg','.png','.bmp')

        # Try test pairs first (*_gt.* + *_light.*)
        for subdir in ['test','Test','val','Val','']:
            d = os.path.join(root, subdir) if subdir else root
            if not os.path.isdir(d): continue
            gt_files = [f for f in os.listdir(d) if '_gt.' in f.lower()
                        and f.lower().endswith(exts)]
            for gf in gt_files:
                stem = gf.lower().split('_gt')[0]
                for ext in exts:
                    lf = os.path.join(d, stem + '_light' + ext)
                    if os.path.exists(lf):
                        self.pairs.append((lf, os.path.join(d, gf)))
                        break
            if self.pairs: break

        # Fall back to 3-panel strips if no test pairs found
        if not self.pairs:
            for subdir in ['train','Train','']:
                d = os.path.join(root, subdir) if subdir else root
                if not os.path.isdir(d): continue
                strip_files = [f for f in os.listdir(d)
                               if f.lower().endswith(exts)]
                for sf in strip_files:
                    self.pairs.append(('strip:' + os.path.join(d, sf), None))
                if self.pairs: break

        # Apply subset
        if subset_frac < 1.0 and self.pairs:
            k = max(1, int(len(self.pairs) * subset_frac))
            self.pairs = random.sample(self.pairs, k)

        self._is_strip = self.pairs and self.pairs[0][0].startswith('strip:')
        # Labels: glare/day/high
        self._wl, self._tl, self._il = 4, 1, 2
        print(f'  SD1 fine-tune: {len(self.pairs)} pairs '
              f'({"strip" if self._is_strip else "paired"} format)')

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        from PIL import Image
        from torchvision import transforms
        ip, gp = self.pairs[idx]
        tt = transforms.ToTensor()

        if self._is_strip:
            # 3-panel strip: use left third as GT, middle as glare input
            img = Image.open(ip[6:]).convert('RGB')
            W, H = img.size; pw = W // 3
            gt_pil   = img.crop((0,  0, pw,   H))
            inp_pil  = img.crop((pw, 0, pw*2, H))
        else:
            inp_pil = Image.open(ip).convert('RGB')
            gt_pil  = Image.open(gp).convert('RGB')

        # Resize
        if self.aug:
            iw, ih = inp_pil.size
            if iw > self.size and ih > self.size:
                x = random.randint(0, iw-self.size)
                y = random.randint(0, ih-self.size)
                inp_pil = inp_pil.crop((x,y,x+self.size,y+self.size))
                gt_pil  = gt_pil.crop( (x,y,x+self.size,y+self.size))
            else:
                inp_pil = inp_pil.resize((self.size, self.size))
                gt_pil  = gt_pil.resize( (self.size, self.size))
            if random.random() > 0.5:
                from PIL import Image as _I
                inp_pil = inp_pil.transpose(_I.FLIP_LEFT_RIGHT)
                gt_pil  = gt_pil.transpose( _I.FLIP_LEFT_RIGHT)
        else:
            inp_pil = inp_pil.resize((self.size, self.size))
            gt_pil  = gt_pil.resize( (self.size, self.size))

        return {
            'image':         tt(inp_pil),
            'clean_image':   tt(gt_pil),
            'weather_label': torch.tensor(self._wl, dtype=torch.long),
            'time_label':    torch.tensor(self._tl, dtype=torch.long),
            'illum_label':   torch.tensor(self._il, dtype=torch.long),
            'image_name':    os.path.basename(ip[6:] if self._is_strip else ip),
            'source':        'SD1',
        }


class FlatPairDataset(torch.utils.data.Dataset):
    """
    Simple paired dataset from flat input/ + target/ folders.
    Used by ManifestDataLoader to load pre-saved fine-tuning data.
    """
    def __init__(self, inp_dir, tgt_dir, weight_label, image_size=512,
                 augment=False):
        self.pairs = []
        self.size  = image_size
        self.aug   = augment
        exts = ('.jpg', '.jpeg', '.png')
        tgt_lut = {os.path.splitext(f)[0]: os.path.join(tgt_dir, f)
                   for f in os.listdir(tgt_dir)
                   if f.lower().endswith(exts)}
        for f in sorted(os.listdir(inp_dir)):
            if not f.lower().endswith(exts): continue
            stem = os.path.splitext(f)[0]
            if stem in tgt_lut:
                self.pairs.append((os.path.join(inp_dir, f), tgt_lut[stem]))
        self.wl = weight_label  # (weather, time, illum) labels

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        from PIL import Image
        from torchvision import transforms
        ip, gp = self.pairs[idx]
        inp = Image.open(ip).convert('RGB')
        gt  = Image.open(gp).convert('RGB')
        if self.aug:
            iw, ih = inp.size
            if iw > self.size and ih > self.size:
                x = random.randint(0, iw - self.size)
                y = random.randint(0, ih - self.size)
                inp = inp.crop((x, y, x+self.size, y+self.size))
                gt  = gt.crop( (x, y, x+self.size, y+self.size))
            else:
                inp = inp.resize((self.size, self.size), Image.LANCZOS)
                gt  = gt.resize( (self.size, self.size), Image.LANCZOS)
            if random.random() > 0.5:
                from PIL import Image as _I
                inp = inp.transpose(_I.FLIP_LEFT_RIGHT)
                gt  = gt.transpose( _I.FLIP_LEFT_RIGHT)
        else:
            inp = inp.resize((self.size, self.size), Image.LANCZOS)
            gt  = gt.resize( (self.size, self.size), Image.LANCZOS)
        tt = transforms.ToTensor()
        wl, tl, il = self.wl
        return {
            'image':         tt(inp),
            'clean_image':   tt(gt),
            'weather_label': torch.tensor(wl, dtype=torch.long),
            'time_label':    torch.tensor(tl, dtype=torch.long),
            'illum_label':   torch.tensor(il, dtype=torch.long),
            'image_name':    os.path.basename(ip),
            'source':        'FT',
        }


# Default condition labels per dataset
_LABEL_MAP = {
    'LOL':      (0, 3, 0),   # clear / night / low
    'WTT':      (0, 1, 1),   # clear / day / medium (mixed, approx)
    'Rain100L': (1, 1, 1),   # rain / day / medium
    'Rain100H': (1, 1, 1),   # rain / day / medium
    'DIDMDN':   (1, 1, 1),   # rain / day / medium
    'RESIDE':   (2, 1, 1),   # fog / day / medium
    'SD1':      (4, 1, 2),   # glare / day / high
}


class ManifestDataLoader:
    """
    Builds train/val/test DataLoaders from the manifest saved by Cell 8b.
    No zip extraction needed — loads directly from Drive flat folders.

    Usage:
        loaders = ManifestDataLoader(FT_MANIFEST, batch_size=4).get_loaders()
    """

    def __init__(self, manifest: dict, batch_size=4, num_workers=2,
                 image_size=512, val_frac=0.15, debug=True):
        self.manifest   = manifest
        self.batch_size = batch_size
        self.nw         = num_workers
        self.size       = image_size
        self.val_frac   = val_frac
        self.debug      = debug

    def get_loaders(self) -> Dict[str, DataLoader]:
        pin = torch.cuda.is_available()
        if platform.system() == 'Windows': self.nw = 0

        tr_datasets = []; va_datasets = []
        tr_weights  = []; va_weights  = []

        for label, info in self.manifest.items():
            inp_dir = info['inp_dir']
            tgt_dir = info['tgt_dir']
            weight  = info.get('weight', 1.0)
            wl      = _LABEL_MAP.get(label, (0, 1, 1))

            if not os.path.isdir(inp_dir):
                if self.debug:
                    print(f'  MISS  {label:<12} {inp_dir}')
                continue

            ds = FlatPairDataset(inp_dir, tgt_dir, wl,
                                 image_size=self.size, augment=True)
            if len(ds) == 0:
                if self.debug: print(f'  EMPTY {label}')
                continue

            # Split into train / val
            n_val = max(1, int(len(ds) * self.val_frac))
            n_tr  = len(ds) - n_val
            idx   = list(range(len(ds)))
            random.shuffle(idx)
            tr_ds = torch.utils.data.Subset(ds, idx[:n_tr])
            va_ds_item = FlatPairDataset(inp_dir, tgt_dir, wl,
                              image_size=self.size, augment=False)
            va_ds = torch.utils.data.Subset(va_ds_item, idx[n_tr:])

            tr_datasets.append(tr_ds); tr_weights.extend([weight]*n_tr)
            va_datasets.append(va_ds); va_weights.extend([weight]*n_val)

            if self.debug:
                print(f'  OK    {label:<12} {n_tr:>4} train  {n_val:>3} val  w={weight}')

        loaders = {}
        if tr_datasets:
            combined_tr = torch.utils.data.ConcatDataset(tr_datasets)
            sampler = WeightedRandomSampler(
                torch.tensor(tr_weights, dtype=torch.float),
                len(combined_tr), replacement=True)
            loaders['train'] = DataLoader(combined_tr,
                batch_size=self.batch_size, sampler=sampler,
                num_workers=self.nw, pin_memory=pin,
                drop_last=True, collate_fn=safe_collate)
            if self.debug:
                print(f'  Train total: {len(combined_tr)} pairs  '
                      f'{len(loaders["train"])} batches/epoch')

        if va_datasets:
            combined_va = torch.utils.data.ConcatDataset(va_datasets)
            loaders['val'] = DataLoader(combined_va,
                batch_size=self.batch_size, shuffle=False,
                num_workers=self.nw, pin_memory=pin,
                drop_last=False, collate_fn=safe_collate)

        return loaders
