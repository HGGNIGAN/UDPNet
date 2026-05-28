# UDPNet Detection Evaluation Pipeline

## Overview

Two-stage pipeline:

1. **Image Restoration** — UDPNet-based dehazing with optional depth guidance
2. **Detection & Evaluation** — YOLO object detection on original and restored images, with dual-detection visualization and detailed metrics

## Key Features

- **Dual-stage evaluation**: Compare baseline (original) vs. restored detection performance
- **Multi-dataset support**: OTS, ITS, DAWN, FoggyCityscape
- **Per-image metrics**: PSNR, SSIM, IoU, per-dataset and aggregated statistics
- **DAWN hazard grouping**: Automatic grouping by hazard type (dust, fog, rain, snow, etc.)
- **Flexible visuals**: Separate folders for original, restored, and dual detection overlays
- **Per-dataset quotas**: Control visual output count per dataset or per hazard type

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Pipeline Workflow

```bash
# 1. Scan and pair datasets
python scripts/step1_scan_datasets.py

# 2. Normalize annotations to YOLO format
python scripts/step2_normalize_gt.py

# 3. Generate depth maps (optional, for depth-guided restoration)
python scripts/step3_generate_depthmaps.py

# 4. Run evaluation
python scripts/step4_evaluate_pipeline.py
```

**Special case — FoggyCityscape:**

```bash
# Extract polygon annotations to bounding boxes
python scripts/extract_foggycityscape.py

# Then proceed with steps 3–4 above
```

## Configuration

See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed config reference, metrics, and troubleshooting.

Key config file: `configs/pipeline.yaml`

## Output Structure

Evaluation outputs organized by run under `outputs/`:

```
outputs/<run>/
├── visuals/
│   ├── <dataset>/
│   │   ├── original/              # Original images
│   │   ├── restored/              # Restored images (if restoration enabled)
│   │   ├── original_detection/    # YOLO detections on original
│   │   └── restored_detection/    # YOLO detections on restored
│   │
│   └── DAWN/
│       └── <hazard>/              # (dusttornado, foggy, haze, mist, rain_storm, sand_storm, snow_storm)
│           ├── original/
│           ├── restored/
│           ├── original_detection/
│           └── restored_detection/
│
└── metrics/
    ├── <dataset>_per_image.csv    # Per-image metrics
    ├── <dataset>_summary.json     # Dataset-level aggregates
    │
    ├── DAWN/
    │   ├── <hazard>_per_image.csv
    │   └── <hazard>_summary.json
    │
    └── all_datasets_summary.json   # Overall metrics
```

## Supported Datasets

- **OTS** — Outdoor Training Set (dehazing)
- **ITS** — Indoor Training Set (dehazing)
- **DAWN** — Diverse Adverse Weather Network (multi-hazard)
- **FoggyCityscape** — Foggy Cityscapes (polygon-based, requires extraction)
- **RTTS** — Rainy Cityscapes

## Metrics

Per-image:
- **PSNR** — Peak Signal-to-Noise Ratio (restoration quality)
- **SSIM** — Structural Similarity Index (restoration quality)
- **Mean IoU** — Per-image best-match IoU (detection performance)
- **Num GT / Num Predictions** — Detection counts

Aggregated (per-dataset):
- Mean, std, min, max of per-image metrics
- Image count

For DAWN, same metrics computed per hazard type separately.

## Acknowledgements

This repository is built upon [UDPNet](https://github.com/Harbinzzy/UDPNet), extending the original repository to provide an evaluation pipeline that measures object detection performance before and after image restoration.

## License

See [LICENSE](LICENSE).