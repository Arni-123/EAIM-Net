# EAIM-Net v5: Environment-Aware Adaptive Image Enhancement Network

**A unified multi-degradation image enhancement framework for scene text recognition under adverse conditions.**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset on Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/shilpiagrawal08/weather-text-time-wtt)
[![Paper](https://img.shields.io/badge/Paper-Expert%20Systems%20with%20Applications-blue)](https://doi.org/XXXXX)

> **Dataset available on Kaggle:** [Weather-Text-Time (WTT)](https://www.kaggle.com/datasets/shilpiagrawal08/weather-text-time-wtt) — 11,000 image pairs across 5 degradation types with per-image condition labels.

---

## Downloads

| File | Size | Link |
|------|------|------|
| `best_model.pth` — trained checkpoint | ~500 MB | [![Download](https://img.shields.io/badge/Google%20Drive-Download-4285F4?logo=googledrive)](https://drive.google.com/file/d/1cr28nEx2FYEt7lEenth1KD16aCYf_1ho/view?usp=sharing) |
| WTT Dataset (11,000 pairs) | 5.7 GB | [![Kaggle](https://img.shields.io/badge/Kaggle-Download-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/shilpiagrawal08/weather-text-time-wtt) |

**Download checkpoint in Python / Colab:**
```python
# Option 1 — gdown (easiest)
pip install gdown
gdown "https://drive.google.com/uc?id=1cr28nEx2FYEt7lEenth1KD16aCYf_1ho" -O checkpoints/best_model.pth

# Option 2 — wget
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1cr28nEx2FYEt7lEenth1KD16aCYf_1ho" -O checkpoints/best_model.pth
```

---

## Overview

EAIM-Net handles **five environmental degradation types** in a single adaptive framework:

| Degradation | Dataset | dPSNR |
|-------------|---------|-------|
| Low-light | LOL eval15 | **+14.59 dB** |
| Haze | RESIDE-ITS | **+12.25 dB** |
| Glare | SD1 | **+8.69 dB** |
| Heavy rain | Rain100H | **+8.67 dB** |
| Medium rain | DID-MDN | **+7.43 dB** |
| All conditions | WTT Synthetic | **+4.61 dB** |
| Light rain | Rain100L | **+2.59 dB** |

---

## Architecture

```
Input → EPE (98.2% weather acc.) → ESS (adaptive blend, H=0.69) → AFB (5 filters) → Enhanced
```

- **EPE** — MobileNetV3-Large, classifies weather / time / illumination
- **ESS** — temperature-scaled softmax blending, no routing supervision
- **AFB** — 5 specialist filters: LowLight, Dehazing, Rain, IllumNorm, Glare

---

## Repository Structure

```
eaim_net/
├── src/                        # Core model source files
│   ├── config.py               # All hyperparameters
│   ├── complete_model.py       # Full EAIM-Net pipeline
│   ├── epe_module.py           # Environmental Parameter Estimator
│   ├── ess_module.py           # Enhancement Strategy Selector
│   ├── afb_module.py           # Adaptive Filter Bank (5 filters)
│   ├── dataset.py              # WTT dataset loader
│   ├── dataset_real_pairs.py   # Multi-dataset loader + ManifestDataLoader
│   ├── dataset_loader_lazy.py  # On-demand zip extraction
│   ├── losses.py               # All loss functions
│   ├── pretrain_filters.py     # Stage 1 filter pre-training
│   └── assemble_and_finetune.py# Stage 2 joint fine-tuning
│
├── scripts/
│   ├── enhance_single.py       # Enhance one image with full analysis
│   ├── enhance_folder.py       # Batch enhance a folder of images
│   └── evaluate.py             # Full PSNR/SSIM/LPIPS evaluation
│
├── notebooks/
│   ├── EAIM_Net_Training.ipynb      # Complete training pipeline (Colab)
│   ├── EAIM_Net_Evaluation.ipynb    # Evaluation notebook (Colab)
│   └── EAIM_Net_Enhancement_Tool.ipynb  # Interactive enhancement tool
│
├── figures/                    # Publication figures (400 dpi)
├── results/                    # Sample enhancement results
├── requirements.txt
└── README.md
```

---


## Dataset — Weather-Text-Time (WTT)

[![Kaggle](https://img.shields.io/badge/Download-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/shilpiagrawal08/weather-text-time-wtt)

| Split | Pairs | Conditions |
|-------|-------|------------|
| Train | 8,800 | All types |
| Val   | 1,100 | All types |
| Test  | 1,100 | All types |

Each image includes a **per-image JSON label** with `weather`, `time`, and `illumination` fields — used to train the EPE classifier (98.2% accuracy).

```bash
# Download via Kaggle CLI
pip install kaggle
kaggle datasets download -d shilpiagrawal08/weather-text-time-wtt --unzip
```

---
## Quick Start

### 1. Install

```bash
git clone https://github.com/yourusername/eaim-net.git
cd eaim-net
pip install -r requirements.txt
```

### 2. Download checkpoint

```bash
pip install gdown
mkdir -p checkpoints
gdown "https://drive.google.com/uc?id=1cr28nEx2FYEt7lEenth1KD16aCYf_1ho" -O checkpoints/best_model.pth
```

Or download manually from [Google Drive](https://drive.google.com/file/d/1cr28nEx2FYEt7lEenth1KD16aCYf_1ho/view?usp=sharing) and place in `checkpoints/`.

### 3. Enhance a single image

```bash
python scripts/enhance_single.py \
    --input  path/to/image.jpg \
    --output path/to/enhanced.jpg \
    --checkpoint checkpoints/best_model.pth
```

### 4. Enhance a folder

```bash
python scripts/enhance_folder.py \
    --input_dir  path/to/images/ \
    --output_dir path/to/enhanced/ \
    --checkpoint checkpoints/best_model.pth
```

### 5. Run evaluation

```bash
python scripts/evaluate.py \
    --dataset lol \
    --data_root path/to/LOL/eval \
    --checkpoint checkpoints/best_model.pth
```

---

## Training

See `notebooks/EAIM_Net_Training.ipynb` for the full Colab pipeline.

**Stage 1** — Pre-train each filter independently:
```bash
python src/pretrain_filters.py --filter lowlight --data_root /path/to/LOL
python src/pretrain_filters.py --filter dehazing  --data_root /path/to/RESIDE
python src/pretrain_filters.py --filter rain      --data_root /path/to/Rain100L
python src/pretrain_filters.py --filter glare     --data_root /path/to/SD1
python src/pretrain_filters.py --filter illum     --data_root /path/to/WTT
```

**Stage 2** — Joint fine-tuning:
```bash
python src/assemble_and_finetune.py \
    --pretrain_dir checkpoints/pretrained_filters/ \
    --data_manifest data/finetune_manifest.json \
    --epochs 60
```

---

## Results

| Method | LOL PSNR | LOL SSIM |
|--------|----------|----------|
| RetinexNet (2018) | 16.77 | 0.560 |
| Zero-DCE (2020) | 14.86 | 0.562 |
| SNR-Aware (2022) | 21.48 | 0.849 |
| Retinexformer (2023) | 22.80 | 0.840 |
| **EAIM-Net (ours)** | **22.36** | **0.7991** |

---

## Citation

```bibtex
@article{eaimnet2024,
  title   = {EAIM-Net: Environment-Aware Adaptive Image Enhancement Network
             for Scene Text Recognition Under Adverse Conditions},
  journal = {Expert Systems with Applications},
  year    = {2024}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
