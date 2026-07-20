# TALVAD — Temporal Audio-Visual Deepfake Detection

> *TALVAD: Temporal Audio-Visual Learning with Differentiable Feature Selection for Deepfake Detection*

TALVAD is a multimodal deepfake detection system that jointly analyses audio and video streams using learned temporal attention and frequency-aware feature selection. It is designed for the LAV-DF benchmark and is extensible to other audio-visual forgery datasets.

---

## Core Contributions

| Component | What It Does |
|-----------|-------------|
| **GFC** (Global Frequency Consistency) | Extracts 8 FFT annular band energies per frame to detect high-frequency GAN synthesis artifacts that spatial CNNs miss |
| **TDFS** (Temporal Differentiable Feature Selector) | Learns a per-timestep soft feature mask end-to-end via a BiGRU gating network; selects discriminative features at each frame independently |
| **Cross-Modal Attention** | Bidirectional audio↔video multi-head attention that forces the model to measure audio-visual synchrony as a forgery signal |
| **A/V Agreement Loss** | Supervises cross-modal contexts: real streams must align (high cosine sim), fake streams must diverge (low cosine sim) |
| **TFLH** (Temporal Frame-Level Head) | Per-frame forgery localisation head trained with `fake_periods` annotations for interpretability and auxiliary supervision |

---

## Project Structure

```
df-multi/
├── configs/                    # YAML experiment configs (v5.5 stable, v6 experimental)
├── datasets/                   # Metadata loaders, sequence builders, annotation loader
├── features/                   # MFCC, GFC, ResNet CNN, fusion, caching
├── models/                     # BiGRUSoftplusDetector, attention layers, TDFS
├── training/                   # Trainer, train.py, kfold_train.py
├── evaluation/                 # Metrics, evaluate.py
├── inference/                  # predict.py (single video or batch directory)
├── preprocessing/              # Audio/video extraction pipeline
├── scripts/                    # eval_temporal.py, precompute_gfc.py, gen_dashboard.py
├── utils/                      # Config loader, reproducibility, I/O helpers
├── checkpoints/                # Saved model weights
├── logs/                       # TensorBoard event files + training_status.json
└── visualizations/             # Plots, ROC curves, attention maps
```

---

## Key Modules

| Module | Description |
|--------|-------------|
| `features/audio_features.py` | MFCC (40 coeffs) + Δ + ΔΔ = 120-dim audio features per frame |
| `features/video_cnn.py` | ResNet-18 (512-dim) frozen ImageNet backbone |
| `features/gfc.py` | 2-D FFT → 8 annular frequency band energies per frame |
| `features/fusion.py` | Per-modality normalisation and sequence alignment |
| `features/build_cache.py` | Build and version feature caches |
| `features/parallel_extract.py` | Parallel feature extraction with thread pool |
| `models/bigru_detector.py` | `BiGRUSoftplusDetector` — full TALVAD model |
| `models/attention.py` | `CrossModalAttentionFusion`, `ModalAttentionFusion`, `TemporalAttention` |
| `models/feature_selector.py` | `TemporalDifferentiableFeatureSelector` (TDFS) |
| `training/trainer.py` | Mixed-precision trainer, multi-task loss, TensorBoard, early stopping |
| `training/train.py` | Main training entry point |
| `training/kfold_train.py` | K-fold cross-validation training |
| `evaluation/evaluate.py` | Held-out test set evaluation |
| `evaluation/metrics.py` | Accuracy, Balanced Accuracy, F1, AUC, ROC, Confusion Matrix |
| `inference/predict.py` | Single-video or batch-directory inference |
| `scripts/eval_temporal.py` | Frame-level + segment AP temporal evaluation |
| `scripts/precompute_gfc.py` | Pre-compute GFC features (CPU-only, fast) |
| `datasets/annotation_loader.py` | `fake_periods` loader from LAV-DF metadata JSON |
| `datasets/sequence_dataset.py` | `MultiModalSequenceDataset`, `PackedSequenceDataset` |

---

## Feature Dimensions

| Stream | Extractor | Dimension |
|--------|-----------|-----------|
| Audio | MFCC + Δ + ΔΔ (40 coeffs each) | 120 |
| Visual — spatial | ResNet-18 (ImageNet, frozen) | 512 |
| Visual — frequency | GFC (8 FFT annular bands) | 8 |
| **Total input per frame** | Concatenated | **640** |

Sequences are sampled at **2 FPS** over **24 timesteps** (≈ 12-second clips).

---

## Installation

```powershell
cd C:\Users\HP\df-multi
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Ensure `ffmpeg` is installed and available in PATH.

**Dependencies:** PyTorch ≥ 2.2, torchvision ≥ 0.17, librosa ≥ 0.10, OpenCV ≥ 4.8, scikit-learn ≥ 1.4

---

## Dataset Setup

### Supported Configs

| Dataset | Config | Status |
|---------|--------|--------|
| LAV-DF | `configs/lavdf_talvad_v5_5.yaml` | Stable |
| LAV-DF | `configs/lavdf_talvad_v6.yaml` | Experimental |

### Metadata CSV Format

| Column | Description |
|--------|-------------|
| `video_path` | Path to video file |
| `label` | `0` (real) or `1` (fake) |
| `sample_id` | Unique numeric identifier |

Generate from folder structure:

```powershell
python -m datasets.prepare_metadata `
    --root ./datasets/LAV-DF `
    --output ./data/lavdf_compact_metadata.csv `
    --dataset LAV-DF
```

For frame-level supervision (TFLH), place the annotation JSON at:

```
./datasets/LAV-DF/metadata.min.json
```

---

## End-to-End Workflow

### 1. Pre-compute GFC features (CPU-only, run once)

```powershell
python scripts/precompute_gfc.py --config configs/lavdf_talvad_v5_5.yaml
```

### 2. Build MFCC + CNN feature cache

```powershell
python -m features.build_cache --config configs/lavdf_talvad_v5_5.yaml
```

### 3. Train TALVAD

```powershell
python -m training.train --config configs/lavdf_talvad_v5_5.yaml
```

Optional K-fold training:

```powershell
python -m training.kfold_train --config configs/lavdf_talvad_v5_5.yaml
```

### 4. Evaluate on held-out test set

```powershell
python -m evaluation.evaluate `
    --config configs/lavdf_talvad_v5_5.yaml `
    --checkpoint checkpoints/talvad_v5_5/best_model.pt
```

### 5. Temporal localisation evaluation (video + frame + segment AP)

```powershell
python scripts/eval_temporal.py --config configs/lavdf_talvad_v5_5.yaml
```

With on-demand GFC computation:

```powershell
python scripts/eval_temporal.py --config configs/lavdf_talvad_v5_5.yaml --compute_gfc
```

### 6. Single-video inference

```powershell
python -m inference.predict `
    --config configs/lavdf_talvad_v5_5.yaml `
    --checkpoint checkpoints/talvad_v5_5/best_model.pt `
    --input path/to/video.mp4
```

### 7. Batch inference

```powershell
python -m inference.predict `
    --config configs/lavdf_talvad_v5_5.yaml `
    --checkpoint checkpoints/talvad_v5_5/best_model.pt `
    --input path/to/video_dir/
```

---

## Config Versions

| Config | Status | Key Differences |
|--------|--------|-----------------|
| `lavdf_talvad_v4.yaml` | Baseline | MFCC + ResNet-18, plain BiGRU, no attention, no GFC |
| `lavdf_talvad_v5.yaml` | Prior | + GFC, cross-modal attention, A/V agreement loss |
| `lavdf_talvad_v5_5.yaml` | **Stable** | + TDFS, TFLH with fixed frame labels, dropout 0.45 |
| `lavdf_talvad_v6.yaml` | Experimental | Dropout 0.4, TFLH weight 0.15, patience 20 |

---

## Multi-Task Loss

```
L_total = L_BCE                          (video-level classification, main task)
        + λ_tflh × L_TFLH               (frame-level localisation)
        + λ_tdfs × L_TDFS_L1            (TDFS sparsity regularisation)
        + λ_av   × L_AV_Agreement       (audio-visual sync supervision)
```

| Weight | v5.5 | v6 |
|--------|------|----|
| `λ_tflh` | 0.10 | 0.15 |
| `λ_tdfs` | 0.05 | 0.05 |
| `λ_av` | 0.10 | 0.10 |

---

## TensorBoard

```powershell
tensorboard --logdir logs
```

---

## Visualisation Outputs

`visualizations/plots.py` generates:

- Training / validation loss and accuracy curves
- ROC curves with AUC
- Confusion matrices
- Temporal attention weight heatmaps
- TDFS soft-mask averages (feature importance)

---

## Reproducibility

- `seed: 42` in YAML controls all randomness (NumPy, PyTorch, CUDA)
- Feature caches versioned by directory signature (`cache_v3`, `cache_v5_gfc`)
- Training state persisted to `logs/<experiment>/training_status.json` after every epoch
- Best checkpoint selected by AUC on validation set
#   T A L V A D - D F - D E T E C T I O N  
 