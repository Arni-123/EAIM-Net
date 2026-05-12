"""
dataset.py — EAIM-Net v5 Final

CRITICAL FIX in v5:
  'high' added to _TARGET_DIRS.
  LOL dataset uses low/ (input) and high/ (clean GT).
  v4 omitted 'high' from the search list → LOL returned 0 samples silently
  → network associated dark images with Rain Removal → weight 0.91 on clear/night.

Also:
  diagnose_lol() function for debugging dataset paths.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Optional, Tuple
import platform


class EnvironmentalTextDataset(Dataset):
    """
    Synthetic Weather-Time-Text (WTT) dataset.

    Expected structure:
        data_root/
            images/          degraded inputs
            labels/          JSON: {"weather":..., "time":..., "illumination":...}
            clean_images/    paired ground-truth
    """

    WEATHER_MAP = {
        'clear':      0,
        'rain':       1,
        'fog':        2,
        'light_snow': 3,
        'snow':       3,   # legacy alias
        'glare':      4,
    }
    TIME_MAP  = {'dawn': 0, 'day': 1, 'dusk': 2, 'night': 3}
    ILLUM_MAP = {'low': 0, 'medium': 1, 'high': 2}

    def __init__(self, data_root, split='train', image_size=512,
                 paired=False, augment=True):
        self.data_root  = data_root
        self.split      = split
        self.image_size = image_size
        self.paired     = paired
        self.augment    = augment and (split == 'train')

        self.image_dir = os.path.join(data_root, 'images')
        self.label_dir = os.path.join(data_root, 'labels')
        self.clean_dir = os.path.join(data_root, 'clean_images') if paired else None

        if not os.path.isdir(self.image_dir):
            self.image_files = []
            return

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.transform = self._get_transforms()

    def _get_transforms(self):
        ops = [transforms.Resize((self.image_size, self.image_size))]
        if self.augment:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.append(transforms.ToTensor())
        return transforms.Compose(ops)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')

        label_path = os.path.join(self.label_dir, os.path.splitext(fname)[0] + '.json')
        try:
            with open(label_path) as f:
                labels = json.load(f)
        except Exception:
            labels = {'weather': 'clear', 'time': 'day', 'illumination': 'medium'}

        weather = labels.get('weather', 'clear')
        if weather in ('heavy_snow', 'snow_medium', 'snow_heavy'):
            weather = 'light_snow'

        weather_label = self.WEATHER_MAP.get(weather, 0)
        time_label    = self.TIME_MAP.get(labels.get('time', 'day'), 1)
        illum_label   = self.ILLUM_MAP.get(labels.get('illumination', 'medium'), 1)

        image_tensor = self.transform(image)
        sample = {
            'image':         image_tensor,
            'weather_label': torch.tensor(weather_label, dtype=torch.long),
            'time_label':    torch.tensor(time_label,    dtype=torch.long),
            'illum_label':   torch.tensor(illum_label,   dtype=torch.long),
            'image_name':    fname,
            'source':        'Synthetic',
        }

        if self.paired and self.clean_dir:
            clean_path = os.path.join(self.clean_dir, fname)
            clean_pil  = Image.open(clean_path).convert('RGB') \
                         if os.path.exists(clean_path) else image
            sample['clean_image'] = self.transform(clean_pil)

        return sample


class RealPairDataset(Dataset):
    """
    Generic paired dataset for LOL / Rain100L / Rain100H / RESIDE.

    FIX v5: 'high' added to _TARGET_DIRS.
    LOL ground-truth folder is named 'high/' — missing from v4.
    """

    _INPUT_DIRS  = ['input', 'low',    'rain',    'haze',  'degraded']
    _TARGET_DIRS = ['target', 'normal', 'gt',     'clean', 'reference',
                    'norain', 'high']   # ← 'high' was missing in v4

    _LOL_SUBDIRS = ['our485', 'eval15', 'train', 'test', '']

    def __init__(self, root, split='train', image_size=512,
                 augment=True, dataset_type='lol', debug=True):
        self.root         = root
        self.split        = split
        self.image_size   = image_size
        self.augment      = augment and (split == 'train')
        self.dataset_type = dataset_type
        self.debug        = debug

        self.input_dir, self.target_dir = self._find_dirs(root, split)

        if self.input_dir is None:
            if debug:
                print(f'  WARNING [{dataset_type}/{split}] No input/target dirs in {root}')
                print(f'    Tried input names  : {self._INPUT_DIRS}')
                print(f'    Tried target names : {self._TARGET_DIRS}')
            self.image_files = []
        else:
            self.image_files = sorted([
                f for f in os.listdir(self.input_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            if debug:
                print(f'  OK [{dataset_type}/{split}]')
                print(f'    input  : {self.input_dir} ({len(self.image_files)} imgs)')
                print(f'    target : {self.target_dir}')

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # Labels: LOL = clear/night/low — NOT rain
        label_map = {
            'lol':     (0, 3, 0),   # weather=clear, time=night, illum=low
            'rain100l': (1, 1, 1),
            'rain100h': (1, 1, 1),
        }
        self._weather_label, self._time_label, self._illum_label = \
            label_map.get(dataset_type, (0, 1, 1))

    def _find_dirs(self, root, split):
        candidates = []
        if self.dataset_type == 'lol':
            subdirs = ['our485', 'train'] if split == 'train' else ['eval15', 'test']
            for sub in subdirs:
                for base in [os.path.join(root, sub), root]:
                    if os.path.isdir(base):
                        candidates.append(base)
        else:
            for base in [os.path.join(root, split), root]:
                if os.path.isdir(base):
                    candidates.append(base)

        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        for base in candidates:
            result = self._search_base(base)
            if result[0] is not None:
                return result
        return None, None

    def _search_base(self, base):
        if not os.path.isdir(base):
            return None, None
        subs       = [d for d in os.listdir(base)
                      if os.path.isdir(os.path.join(base, d))]
        subs_lower = {s.lower(): s for s in subs}
        inp = next((os.path.join(base, subs_lower[k])
                    for k in self._INPUT_DIRS if k in subs_lower), None)
        tgt = next((os.path.join(base, subs_lower[k])
                    for k in self._TARGET_DIRS if k in subs_lower), None)
        if inp and tgt:
            return inp, tgt
        imgs = [f for f in os.listdir(base)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if imgs:
            return base, base
        return None, None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        inp   = Image.open(os.path.join(self.input_dir,  fname)).convert('RGB')

        tgt_p = os.path.join(self.target_dir, fname)
        if not os.path.exists(tgt_p):
            stem = os.path.splitext(fname)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                alt = os.path.join(self.target_dir, stem + ext)
                if os.path.exists(alt):
                    tgt_p = alt
                    break

        tgt = Image.open(tgt_p).convert('RGB') \
              if os.path.exists(tgt_p) else inp

        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed); inp_t = self.transform(inp)
        torch.manual_seed(seed); tgt_t = self.transform(tgt)

        return {
            'image':         inp_t,
            'clean_image':   tgt_t,
            'weather_label': torch.tensor(self._weather_label, dtype=torch.long),
            'time_label':    torch.tensor(self._time_label,    dtype=torch.long),
            'illum_label':   torch.tensor(self._illum_label,   dtype=torch.long),
            'image_name':    fname,
            'source':        self.dataset_type.upper(),
        }


def safe_collate(batch):
    from torch.utils.data._utils.collate import default_collate
    for s in batch:
        if 'source'     not in s: s['source']     = 'Synthetic'
        if 'image_name' not in s: s['image_name'] = ''
    string_keys  = ['image_name', 'source']
    tensor_batch = [{k: v for k, v in s.items() if k not in string_keys} for s in batch]
    collated     = default_collate(tensor_batch)
    for k in string_keys:
        collated[k] = [s[k] for s in batch]
    return collated


def diagnose_lol(lol_root):
    """Run this to verify LOL folder structure and pair counts."""
    print(f'\n{"="*55}')
    print(f'  LOL DIAGNOSIS: {lol_root}')
    print(f'{"="*55}')
    if not os.path.isdir(lol_root):
        print(f'  NOT FOUND: {lol_root}'); return

    for name in sorted(os.listdir(lol_root)):
        full = os.path.join(lol_root, name)
        if os.path.isdir(full):
            subs = os.listdir(full)
            n    = sum(1 for f in subs if f.lower().endswith(('.jpg','.png','.jpeg')))
            print(f'  {name}/')
            for s in sorted(subs):
                sp = os.path.join(full, s)
                if os.path.isdir(sp):
                    ni = len([f for f in os.listdir(sp)
                              if f.lower().endswith(('.jpg','.png','.jpeg'))])
                    print(f'    {s}/  ({ni} images)')

    print()
    for split in ['train', 'val', 'test']:
        ds = RealPairDataset(root=lol_root, split=split,
                             image_size=256, dataset_type='lol', debug=False)
        status = 'OK' if len(ds) > 0 else 'EMPTY'
        print(f'  [{split}] {status}: {len(ds)} pairs')
    print('='*55)


def create_dataloaders(data_root, batch_size=4, num_workers=2,
                       image_size=512, paired=True):
    loaders = {}
    pin = torch.cuda.is_available()
    if platform.system() == 'Windows':
        num_workers = 0
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        ds = EnvironmentalTextDataset(
            data_root  = split_dir,
            split      = split,
            image_size = image_size,
            paired     = paired,
            augment    = (split == 'train'),
        )
        if len(ds) == 0:
            continue
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == 'train'),
            num_workers = num_workers,
            pin_memory  = pin,
            drop_last   = (split == 'train'),
            collate_fn  = safe_collate,
        )
    return loaders
