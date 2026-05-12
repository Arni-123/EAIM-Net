"""
config.py

"""

import yaml

CONFIG = {
    # ── Model Architecture ──────────────────────────────────────────
    'model': {
        'backbone':            'mobilenet_v3_large',
        'num_weather_classes': 5,     # clear / rain / fog / light_snow / glare
        'num_time_classes':    4,     # dawn / day / dusk / night
        'num_illum_classes':   3,     # low / medium / high
        'num_filters':         5,
        'feature_dim':         256,
    },

    # ── Training Parameters ─────────────────────────────────────────
    'training': {
        'batch_size':        4,
        'num_epochs':        80,       # absolute target epoch
        'learning_rate':     5e-5,     # ESS + AFB + Combiner
        'lr_epe':            5e-6,     # EPE backbone (10x lower)
        'weight_decay':      1e-5,
        'grad_clip':         1.0,
        # lambda_route is GONE — routing loss caused τ→0.1, H→0 (v4 bug)
        'lambda_weather':    0.5,
        'lambda_time':       0.3,
        'lambda_illum':      0.3,
        'lambda_ssim':       0.3,      # SSIM weight in base_enh_loss
        'lambda_perceptual': 0.1,
        'cosine_T0':         10,       # CosineAnnealingWarmRestarts
        'phase_b_start':     10,       # epoch to unfreeze AFB (Phase A→B)
    },

    # ── Data ────────────────────────────────────────────────────────
    'data': {
        'image_size':   512,
        'train_split':  0.80,
        'val_split':    0.10,
        'test_split':   0.10,
    },

    # ── Paths (overridden in notebook / CLI) ────────────────────────
    'paths': {
        'checkpoints': './checkpoints_v5',
        'logs':        './logs_v5',
        'results':     './results_v5',
        'pretrained':  './pretrained_filters',
    },

    # ── Dataset weights for combined loader ─────────────────────────
    'dataset_weights': {
        'lol_weight':      3.0,   # raised from 2.0 — LOL was missing (high/ bug)
        'rain100l_weight': 1.0,
        'rain100h_weight': 1.5,
        'didmdn_weight':   1.2,
    },

    # ── Scope note ──────────────────────────────────────────────────
    'scope': {
        'supported_degradations': [
            'low_light', 'rain_light', 'rain_medium', 'rain_heavy',
            'haze_light', 'haze_medium', 'glare', 'illumination_variation',
        ],
        'excluded_degradations': ['heavy_snow'],
        'exclusion_reason': (
            'Heavy snow occludes scene text beyond recovery. '
            'Including it degrades all other conditions. '
            'Explicit design decision — see paper Section 3.1.'
        ),
    },
}


def save_config(config, path):
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    save_config(CONFIG, 'config.yaml')
    loaded = load_config('config.yaml')
    print("Config saved → config.yaml")
    print(f"  Weather classes : {CONFIG['model']['num_weather_classes']}")
    print(f"  routing loss    : REMOVED (v5 fix)")
    print(f"  lr_epe          : {CONFIG['training']['lr_epe']}")
    print(f"  LOL weight      : {CONFIG['dataset_weights']['lol_weight']}")
