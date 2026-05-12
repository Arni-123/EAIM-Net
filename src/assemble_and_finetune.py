"""
assemble_and_finetune.py — EAIM-Net v5 Final

Step 7+8 of the proposed plan:
  7. Assemble: load pre-trained filter weights into full model
  8. Joint fine-tune with text-aware loss

  Phase A (epochs 0-10):  Freeze AFB. Train EPE + ESS only.
                          ESS learns to blend the pre-trained filters.
  Phase B (epochs 10-40): Unfreeze all. Joint fine-tuning.
                          Lower LR for filters vs ESS.

Usage:
    python assemble_and_finetune.py \\
        --pretrained_dir ./pretrained_filters \\
        --ckpt_dir       ./checkpoints_v5 \\
        --data_root      ./weather_time_data \\
        --lol_root       ./LOL \\
        --rain100l       ./Rain100L \\
        --rain100h       ./Rain100H \\
        --epochs         40
"""

import os
import glob
import json
import time
import argparse

import torch
import torch.optim as optim
from tqdm import tqdm

from complete_model import AdaptiveEnhancementModel
from losses import FullPipelineLoss
from config  import CONFIG, load_config, save_config
from dataset_real_pairs import create_combined_dataloaders
from ess_module import FILTER_NAMES


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ckpt_valid(p: str) -> bool:
    if not os.path.exists(p) or os.path.getsize(p) < 1024:
        return False
    try:
        import zipfile
        with zipfile.ZipFile(p, 'r') as z:
            return len(z.namelist()) > 0
    except Exception:
        return False


def find_latest_ckpt(d: str):
    for p in sorted(glob.glob(os.path.join(d, 'checkpoint_epoch_*.pth')), reverse=True):
        if _ckpt_valid(p):
            return p
    best = os.path.join(d, 'best_model.pth')
    return best if _ckpt_valid(best) else None


def _cleanup(d: str, keep_n: int = 5):
    all_ep = sorted(glob.glob(os.path.join(d, 'checkpoint_epoch_*.pth')))
    for p in all_ep[:-keep_n]:
        try:
            os.remove(p)
        except Exception:
            pass


def load_pretrained_filters(model, pretrained_dir: str, device):
    """
    Load best pre-trained weights for each AFB filter.
    Filter index → save name mapping:
      0 = lowlight_best.pth
      1 = dehazing_best.pth
      2 = rainremoval_best.pth
      3 = illumnorm_best.pth
      4 = glare_best.pth
    """
    filter_map = [
        (0, 'lowlight_best.pth'),
        (1, 'dehazing_best.pth'),
        (2, 'rainremoval_best.pth'),
        (3, 'illumnorm_best.pth'),
        (4, 'glare_best.pth'),
    ]
    loaded = []
    for idx, fname in filter_map:
        p = os.path.join(pretrained_dir, fname)
        if os.path.exists(p):
            state = torch.load(p, map_location=device)
            model.afb.filters[idx].load_state_dict(state)
            loaded.append(FILTER_NAMES[idx])
            print(f'  [filter {idx}] {FILTER_NAMES[idx]:<22} loaded from {fname}')
        else:
            print(f'  [filter {idx}] {FILTER_NAMES[idx]:<22} NOT FOUND — random init')
    return loaded


def make_optimizer(model, phase: str):
    """
    Phase A: only ESS + combiner train (AFB and EPE frozen).
    Phase B: all modules train, lower LR on AFB and EPE.
    """
    if phase == 'A':
        for p in model.afb.parameters(): p.requires_grad = False
        for p in model.epe.parameters(): p.requires_grad = False
        for p in model.ess.parameters(): p.requires_grad = True
        for p in model.combiner.parameters(): p.requires_grad = True
        trainable = [p for p in model.parameters() if p.requires_grad]
        return optim.Adam(trainable, lr=5e-5, weight_decay=1e-5)
    else:  # Phase B
        for p in model.parameters(): p.requires_grad = True
        return optim.Adam([
            {'params': model.epe.parameters(),      'lr': 5e-6,  'name': 'epe'},
            {'params': model.afb.parameters(),      'lr': 1e-5,  'name': 'afb'},
            {'params': model.ess.parameters(),      'lr': 5e-5,  'name': 'ess'},
            {'params': model.combiner.parameters(), 'lr': 5e-5,  'name': 'combiner'},
        ], weight_decay=1e-5)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='EAIM-Net v5 Assembly + Joint Fine-tuning')
    parser.add_argument('--pretrained_dir', required=True,
                        help='Folder with *_best.pth filter weights')
    parser.add_argument('--ckpt_dir',       required=True)
    parser.add_argument('--data_root',      required=True,
                        help='Synthetic WTT dataset root')
    parser.add_argument('--lol_root',       default=None)
    parser.add_argument('--rain100l',       default=None)
    parser.add_argument('--rain100h',       default=None)
    parser.add_argument('--didmdn',         default=None)
    parser.add_argument('--config',         default='config.yaml')
    parser.add_argument('--epochs',         type=int, default=40)
    parser.add_argument('--batch_size',     type=int, default=4)
    parser.add_argument('--phase_b_start',  type=int, default=10,
                        help='Epoch at which to switch from Phase A to Phase B')
    parser.add_argument('--keep_ckpts',     type=int, default=5)
    parser.add_argument('--log_file',       default='finetune_log.txt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Config ─────────────────────────────────────────────────────────────────
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = CONFIG.copy()
    config['training']['batch_size'] = args.batch_size
    config['training']['num_epochs'] = args.epochs
    config['paths']['checkpoints']   = args.ckpt_dir
    save_config(config, args.config)

    # ── Dataloaders ────────────────────────────────────────────────────────────
    def _exists(p): return p and os.path.isdir(p) and len(os.listdir(p)) > 0

    print('\nBuilding dataloaders...')
    dataloaders = create_combined_dataloaders(
        synthetic_data_root = args.data_root if _exists(args.data_root) else None,
        lol_root            = args.lol_root  if _exists(args.lol_root)  else None,
        rain100l_root       = args.rain100l  if _exists(args.rain100l)  else None,
        rain100h_root       = args.rain100h  if _exists(args.rain100h)  else None,
        batch_size  = config['training']['batch_size'],
        num_workers = 2,
        image_size  = config['data']['image_size'],
        lol_weight  = 3.0,
        debug       = True,
    )
    assert 'train' in dataloaders, 'No training data found!'

    # ── Build model ────────────────────────────────────────────────────────────
    print('\nBuilding model...')
    model = AdaptiveEnhancementModel(
        backbone            = config['model']['backbone'],
        num_weather_classes = config['model']['num_weather_classes'],
        num_time_classes    = config['model']['num_time_classes'],
        num_illum_classes   = config['model']['num_illum_classes'],
        feature_dim         = config['model']['feature_dim'],
        num_filters         = config['model']['num_filters'],
        pretrained          = True,
    ).to(device)

    # ── Load pre-trained filter weights ───────────────────────────────────────
    print('\nLoading pre-trained filters:')
    loaded_filters = load_pretrained_filters(model, args.pretrained_dir, device)
    print(f'  {len(loaded_filters)}/5 filters pre-trained: {loaded_filters}')

    # ── Load EPE from existing checkpoint (preserves 98% accuracy) ───────────
    existing_ckpt = find_latest_ckpt(args.ckpt_dir)
    start_epoch   = 0
    best_val_enh  = float('inf')

    assembled_path = os.path.join(args.ckpt_dir, 'assembled_pretrained.pth')

    if existing_ckpt and os.path.basename(existing_ckpt) != 'assembled_pretrained.pth':
        print(f'\nLoading EPE weights from: {os.path.basename(existing_ckpt)}')
        ck = torch.load(existing_ckpt, map_location=device)
        epe_state = {k[4:]: v for k, v in ck['model_state_dict'].items()
                     if k.startswith('epe.')}
        model.epe.load_state_dict(epe_state, strict=False)
        print(f'  EPE loaded  epoch={ck["epoch"]}  version={ck.get("version","?")}')

    # Save assembled checkpoint as starting point
    torch.save({
        'epoch':             -1,
        'model_state_dict':  model.state_dict(),
        'best_val_enh':      float('inf'),
        'config':            config,
        'version':           'v5_assembled',
        'loaded_filters':    loaded_filters,
    }, assembled_path)
    print(f'  Assembled model saved: {assembled_path}')

    # ── Loss ───────────────────────────────────────────────────────────────────
    criterion = FullPipelineLoss(
        lambda_weather = config['training'].get('lambda_weather', 0.5),
        lambda_time    = config['training'].get('lambda_time',    0.3),
        lambda_illum   = config['training'].get('lambda_illum',   0.3),
        lam_text       = 2.0,
        lam_ssim       = 0.3,
    )

    # ── Optimizer — start Phase A ──────────────────────────────────────────────
    current_phase = 'A'
    optimizer = make_optimizer(model, 'A')
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    # ── Logging ────────────────────────────────────────────────────────────────
    log_path = os.path.join(args.ckpt_dir, args.log_file)
    _lf = open(log_path, 'a', buffering=1)

    def log(msg):
        print(msg)
        _lf.write(msg + '\n')
        _lf.flush()

    log(f'\n{"="*72}')
    log(f'EAIM-Net v5 Joint Fine-tuning  {time.strftime("%Y-%m-%d %H:%M")}')
    log(f'Epochs: {start_epoch}->{start_epoch+args.epochs-1}')
    log(f'Phase A (ESS only) until epoch {args.phase_b_start}')
    log(f'Phase B (all): AFB lr=1e-5  ESS lr=5e-5')
    log(f'Text-aware loss: Laplacian mask weight=2.0')
    log(f'Loaded filters: {loaded_filters}')
    log(f'{"="*72}')
    log(f'  {"Ep":>4}  {"Ph":>2}  {"epe":>7}  {"enh":>7}  '
        f'{"H":>6}  {"val_enh":>8}  {"W":>5}  {"tau":>6}  {"lr":>9}')
    log('  ' + '-' * 70)

    # ── Training loop ──────────────────────────────────────────────────────────
    end_epoch = start_epoch + args.epochs

    for epoch in range(start_epoch, end_epoch):

        # Phase switch
        if epoch == args.phase_b_start and current_phase == 'A':
            log(f'\n  Switching to Phase B (unfreezing AFB filters)')
            optimizer = make_optimizer(model, 'B')
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=1, eta_min=1e-6)
            current_phase = 'B'

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        t_epe = t_enh = t_ent = t_n = 0

        for batch in tqdm(dataloaders['train'], desc=f'Ep{epoch:03d}', leave=True):
            imgs = batch['image'].to(device)
            out  = model(imgs)

            loss, breakdown = criterion(out, batch, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            t_epe += breakdown['epe']
            t_enh += breakdown['enh']
            t_ent += out['weight_entropy'].mean().item()
            t_n   += 1

        scheduler.step(epoch + 1)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        v_enh = v_ent = v_wa = v_n = 0

        with torch.no_grad():
            for batch in dataloaders['val']:
                imgs  = batch['image'].to(device)
                w_lbl = batch['weather_label'].to(device)
                out   = model(imgs)
                _, bd = criterion(out, batch, device)

                v_enh += bd['enh']
                v_ent += out['weight_entropy'].mean().item()
                v_wa  += (out['weather_logits'].argmax(1) == w_lbl).float().mean().item()
                v_n   += 1

        val_enh = v_enh / v_n
        tau     = model.ess.tau.item()
        lr_now  = optimizer.param_groups[-1]['lr']

        is_best = val_enh < best_val_enh
        if is_best:
            best_val_enh = val_enh

        # ── Save checkpoint every epoch ────────────────────────────────────────
        ck_data = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_enh':         best_val_enh,
            'config':               config,
            'version':              'v5_finetune',
        }
        ep_path = os.path.join(args.ckpt_dir, f'checkpoint_epoch_{epoch:04d}.pth')
        torch.save(ck_data, ep_path)
        if is_best:
            torch.save(ck_data, os.path.join(args.ckpt_dir, 'best_model.pth'))

        _cleanup(args.ckpt_dir, keep_n=args.keep_ckpts)

        line = (f'  {epoch:>4}  {current_phase:>2}  '
                f'{t_epe/t_n:>7.4f}  {t_enh/t_n:>7.4f}  '
                f'{t_ent/t_n:>6.3f}  {val_enh:>8.4f}  '
                f'{v_wa/v_n:>5.3f}  {tau:>6.3f}  {lr_now:>9.2e}'
                + ('  BEST' if is_best else ''))
        log(line)

    _lf.close()
    log(f'\nFine-tuning complete. Best val_enh={best_val_enh:.6f}')


if __name__ == '__main__':
    main()
