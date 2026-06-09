# UDPNet Detection Evaluation Pipeline Documentation

This document explains the complete two-stage pipeline in this repository:

1. Stage 1: Image restoration (UDPNet-based dehazing with optional depth prior)
2. Stage 2: Object detection (Ultralytics YOLO)
3. Evaluation: Per-image metrics (PSNR, SSIM, explicit IoU metrics) + per-dataset aggregates (mAP, IoU) comparing baseline and restored

It also explains every YAML config field, every script/module, the end-to-end run order, and how to improve scores.

## 1. Pipeline At a Glance

The high-level flow is:

1. Scan datasets and pair images + raw annotations
2. Normalize annotations to YOLO format
3. Generate depth maps for organized images
4. Run evaluation pipeline:
   - Load image + depth + normalized GT
   - Run YOLO detection on original image file paths by default
   - Run restoration model (or bypass if disabled)
   - Save restored images at original resolution
   - Run YOLO detection on saved restored image file paths
   - Compute metrics (`mean_iou_all_gt`, `mean_iou_matched`, `mean_iou_tp50`, and mAP)
   - Save dual-detection visuals and per-image/per-dataset metrics

Recommended script order:

1. `scripts/step1_scan_datasets.py`
2. `scripts/step2_normalize_gt.py`
3. `scripts/step3_generate_depthmaps.py`
4. `scripts/step4_evaluate_pipeline.py`

FoggyCityscape uses one extra handoff step before the pipeline:

1. `scripts/extract_foggycityscape.py`
2. `scripts/step1_scan_datasets.py`
3. `scripts/step2_normalize_gt.py`
4. `scripts/step3_generate_depthmaps.py`
5. `scripts/step4_evaluate_pipeline.py`

The extractor is standalone. It reads every raw FoggyCityscape `val/` foggy image, converts allowed object polygons to VOC XML boxes, drops images with no allowed objects, and writes a flat extracted tree under `Datasets/extracted/FoggyCityscape`. The main pipeline then consumes that extracted tree only.

## 2. Configuration File (`configs/pipeline.yaml`)

### 2.1 `project`

- `name`: Project label (informational).
- `colab_mode`:
  - `false`: use `local_root`
  - `true`: use `colab_root`
- `local_root`: Root path used when running locally.
- `colab_root`: Root path used when `colab_mode=true`.

How it is used:

- `pipeline/common/paths.py` resolves all relative paths from either `local_root` or `colab_root`.

Score impact:

- Indirect. Wrong root paths lead to wrong files, missing data, and invalid scoring runs.

### 2.2 `runtime`

- `device`: Runtime device (`cuda:0` or `cpu`).
- `num_workers`: DataLoader worker processes.
- `seed`: Intended reproducibility seed value.
- `gpu_ids`: Optional list of GPU ids for multi-GPU execution, for example `[0, 1]`.
- `enable_data_parallel`: If `true`, restoration model wraps with `torch.nn.DataParallel` when multiple GPUs are available.

Multi-GPU behavior in the current pipeline:

- Restoration stage can use multiple GPUs through `DataParallel` when `gpu_ids` has more than one entry.
- Detection stage receives a YOLO-compatible device string, so it can target one GPU or a comma-separated multi-GPU selection.
- Depth generation accepts the same runtime device config, but currently runs on the first available GPU only. This keeps the pipeline backward-compatible, but it does not shard depth inference across multiple GPUs yet.

Recommended values:

```yaml
# Single GPU
runtime:
    device: "cuda:0"

# Multi GPU
runtime:
    device: "cuda:0,1"
    gpu_ids: [0, 1]
    enable_data_parallel: true
```

Score impact:

- `device`: mostly speed/throughput impact, not metric formula impact.
- `num_workers`: throughput/stability impact only.

### 2.3 `vram`

- `batch_size`: Batch size for evaluation DataLoader.
- `image_resolution`: `[height, width]` resize resolution used by the evaluator dataset.

Score impact (high):

- `image_resolution` has major impact on detection quality and therefore IoU/mAP.
- `batch_size` usually affects speed and memory more than final score, but OOM can force unstable runs.

### 2.4 `paths`

- `datasets_root`: Root datasets folder (informational in current flow).
- `organized_root`: Output folder for paired/normalized/depth-organized data.
- `outputs_root`: Folder where evaluation run outputs are created.

Score impact:

- Indirect. Wrong path -> wrong dataset subset or empty inputs.

### 2.5 `restoration`

- `enabled`: If `true`, run restoration model before detection; if `false`, restoration stage is bypassed.
- `model_module`: Python module containing model builder (e.g., `Dehazing.OTS.models.FSNet_UDPNet`).
- `build_function`: Builder function name in module (usually `build_net`).
- `checkpoint`: Restoration checkpoint path.
- `use_depth`: If `true`, model input is RGB+Depth; if `false`, RGB only.
- `output_index`: If model returns list/tuple, which tensor to use as restored output.
- `strict_state_dict`: Strict checkpoint loading behavior.

Score impact (high):

- `enabled`: determines if you are evaluating restoration effect at all.
- `model_module` + `checkpoint`: biggest restoration quality control.
- `use_depth`: strong effect if model architecture expects depth guidance.
- `output_index`: wrong index can destroy restoration output quality.

Important compatibility note:

- `model_module` should match checkpoint architecture family (e.g., OTS module with OTS checkpoint).

Pairs of `model_module` and `checkpoint`:

Use these pairs in restoration config:

```yaml
1.
model_module: Dehazing.OTS.models.FSNet_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/FSNet_UDPNet_OTS.ckpt

2.
model_module: Dehazing.ITS.models.FSNet_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/FSNet_UDPNet_ITS.ckpt

3.
model_module: Dehazing.OTS.models.FSNet_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/FSNet_UDPNet_NHR.ckpt

4.
model_module: Dehazing.ITS.models.FSNet_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/FSNet_UDPNet_NHR.ckpt

5.
model_module: Dehazing.OTS.models.FSNet_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/FSNet_UDPNet_haze4k.ckpt

6.
model_module: Dehazing.ITS.models.FSNet_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/FSNet_UDPNet_haze4k.ckpt

7.
model_module: Dehazing.OTS.models.ConvIR_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/ConvIR_UDPNet_OTS.ckpt

8.
model_module: Dehazing.ITS.models.ConvIR_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/ConvIR_UDPNet_ITS.ckpt

9.
model_module: Dehazing.OTS.models.ConvIR_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/ConvIR_UDPNet_NHR.ckpt

10.
model_module: Dehazing.ITS.models.ConvIR_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/ConvIR_UDPNet_NHR.ckpt

11.
model_module: Dehazing.OTS.models.ConvIR_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/ConvIR_UDPNet_haze4k.ckpt

12.
model_module: Dehazing.ITS.models.ConvIR_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/ConvIR_UDPNet_haze4k.ckpt

13.
model_module: Dehazing.OTS.models.ConvIR_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/ConvIR_UDPNet_GTA5.ckpt

14.
model_module: Dehazing.ITS.models.ConvIR_UDPNet
checkpoint: UDPNet_pretrained_(checkpoints)/ConvIR_UDPNet_GTA5.ckpt
```

Important:

PoolNet checkpoint exists: `UDPNet*pretrained*(checkpoints)/PoolNet_UDPNet_GTA5.ckpt`.

However, there's no PoolNet restoration module in current Dehazing modules, so PoolNet is not valid now (unless added PoolNet module + build_net()).

Best-safe picks:

```text
FSNet_UDPNet_OTS.ckpt ↔ Dehazing.OTS.models.FSNet_UDPNet
ConvIR_UDPNet_OTS.ckpt ↔ Dehazing.OTS.models.ConvIR_UDPNet
```

These have the same domain tag OTS, so there's less mismatch risk.

### 2.6 `detection`

- `weights`: YOLO weights path.
- `imgsz`: YOLO inference image size. File-path mode passes this directly to Ultralytics.
- `conf_threshold`: Detection confidence filter.
- `nms_iou_threshold`: NMS IoU threshold.
- `max_det`: Max detections per image.

Score impact (very high):

- `weights`: primary detection baseline quality.
- `conf_threshold`:
  - Higher -> fewer false positives, but may lose true positives.
  - Lower -> more recall, potentially lower precision.
- `nms_iou_threshold`:
  - Too low can suppress valid boxes.
  - Too high can keep duplicates.
- `max_det`: can cap detections and affect recall on crowded scenes.

### 2.7 `evaluation`

- `max_images`: limit number of evaluated samples (`0` means full dataset).
- `max_visuals`: controls saved visual count. Can be:
  - Integer: applied per-dataset (e.g., `max_visuals: 100` saves up to 100 images per dataset)
  - Dict: per-dataset mapping (e.g., `{DAWN: 50, RTTS: 20}`)
  - `0`: unlimited (save all)
- `detector_input_mode`: `file_path` (default final-report mode) or `tensor` (legacy fast mode).
- `save_restored_full_resolution`: saves restored outputs resized back to original image dimensions. File-path mode forces this behavior so GT coordinates stay aligned.
- `restored_image_format`: image extension for `restored_images/` artifacts, e.g. `png`.
- `restore_output_interpolation`: resize interpolation for restored output (`lanczos`, `linear`, `cubic`, `area`, `nearest`).
- `map_class_set`: `gt_only` or `gt_union_pred`. `gt_union_pred` matches the RTTS parity script.
- `iou_primary`: primary IoU display choice: `all_gt`, `matched`, or `tp50`.
- `map_iou_thresholds`: IoU thresholds used for mAP aggregation.

Score impact:

- `map_iou_thresholds` directly defines reported `map` value.
- `max_images` changes statistical stability of results (small sample variance can be large).
- `metrics.use_multichannel_ssim`: boolean flag (default `false`) to compute SSIM on all channels vs. luminance only.
- `mean_iou_all_gt`: one value per GT; unmatched GT contributes `0.0`.
- `mean_iou_matched`: only GT boxes with best class-matched IoU > 0.
- `mean_iou_tp50`: true-positive matches with IoU >= 0.50.

### 2.7b Visual Output Structure

Evaluation saves four types of visuals per image:

- **`original/`**: Original (non-restored) images in RGB
- **`restored/`**: Restored images after dehazing (if restoration enabled)
- **`original_detection/`**: YOLO predictions drawn on original images (baseline detection)
- **`restored_detection/`**: YOLO predictions drawn on restored images (restored detection)

Directory layout:

```
outputs/<run>/visuals/<dataset>/
├── original/
├── restored/
├── original_detection/
└── restored_detection/
```

File-path mode also writes durable restored images:

```
outputs/<run>/restored_images/<dataset>/<image_id>.png
```

DAWN uses the same layout as every other dataset. Its scanner may filter filenames to haze/fog prefixes, but evaluation does not create hazard subfolders.

### 2.8 `normalization`

- `output_label_dir_name`: folder name for normalized YOLO labels.
- `skip_unknown_classes`: skip objects not in `class_map`.
- `clamp_boxes`: clamp normalized box values to `[0,1]`.
- `yolo_txt_assume_normalized`:
  - If true, YOLO labels expected normalized unless values suggest pixel scale.

Score impact (high):

- Label normalization quality is critical for trustworthy IoU/mAP.
- `skip_unknown_classes` can reduce GT count and alter class-wise AP.
- Bad class mapping produces invalid evaluation.

### 2.9 `datasets`

- `active`: active dataset names.
- `scan`:
  - `link_mode`: `symlink|hardlink|copy` for organized files.
  - `overwrite`: overwrite organized outputs.
- `entries.<DatasetName>`:
  - `annotation_format`: `yolo_txt` or `voc_xml`
  - `root`, `image_dir`, `label_dir`
  - `split` and optional `split_file`
  - `image_extensions`, `label_extensions`
  - `class_map` (maps class names or numeric label ids into detector class ids)

FoggyCityscape handoff:

- `scripts/extract_foggycityscape.py` writes the extracted sample tree to `Datasets/extracted/FoggyCityscape`.
- The pipeline config should point `datasets.entries.FoggyCityscape.root` at that extracted tree.
- The pipeline itself should not perform city-balanced filtering or beta selection.

Score impact (high):

- Wrong `annotation_format`, `split_file`, or `class_map` invalidates GT alignment and scoring.

### 2.10 FoggyCityscape extraction workflow

Use this workflow when you only have the raw FoggyCityscape dataset and not this repository's source code setup:

```bash
python scripts/extract_foggycityscape.py \
    --source-root Datasets/FoggyCityscape-raw \
    --output-root Datasets/extracted/FoggyCityscape \
    --split-filter val
```

The extractor:

- reads `leftImg8bit_foggy/val/<city>/` and `gtFine/val/<city>/`
- keeps only driving-object classes
- converts polygon objects into VOC-style bounding boxes
- includes every fog beta variant present
- optionally filters fog beta variants with `--fog-levels 0.01 0.02`
- drops images with no allowed object boxes
- writes `images/`, `labels_raw/`, and `manifests/`

Then run the existing pipeline stages against `Datasets/extracted/FoggyCityscape`.

## 3. Script-by-Script Guide (`scripts/`)

### 3.1 `scripts/step1_scan_datasets.py`

Purpose:

- Scans raw datasets from config and creates organized paired structure:
  - `Datasets/organized/<Dataset>/images`
  - `Datasets/organized/<Dataset>/labels_raw`
  - `Datasets/organized/<Dataset>/manifests/pairs.csv`
  - `Datasets/organized/<Dataset>/manifests/summary.json`

CLI:

- `--config`
- `--datasets`
- `--link-mode`
- `--overwrite`
- `--dry-run`

### 3.2 `scripts/step2_normalize_gt.py`

Purpose:

- Converts raw labels (`yolo_txt` or `voc_xml`) into normalized YOLO `.txt` labels.
- Writes:
  - `labels_yolo/`
  - `manifests/pairs_yolo.csv`
  - `manifests/normalization_summary.json`

CLI:

- `--config`
- `--datasets`
- `--overwrite`

### 3.3 `scripts/step3_generate_depthmaps.py`

Purpose:

- Generates depth maps for organized images by invoking `scripts/depthmap-create.py`.
- Writes:
  - `Datasets/organized/<Dataset>/DepthMaps/`

CLI:

- `--config`
- `--datasets`
- `--overwrite`
- `--dry-run`

### 3.4 `scripts/depthmap-create.py`

Purpose:

- Runs DepthAnythingV2 (`vits`) inference over an input directory and saves 8-bit normalized depth maps.
- Accepts `--device` for `auto`, `cpu`, `cuda`, `cuda:N`, or `cuda:N,M`.
- If a multi-GPU string is passed, the first GPU is used for inference.

Inputs:

- `input_dir`
- `output_dir`

Outputs:

- Depth images mirrored by relative path into output dir.
- Prints summary: `processed`, `skipped`, `failed`.

### 3.5 `scripts/step4_evaluate_pipeline.py`

Purpose:

- Runs full evaluation workflow (restoration + detection + metrics + reports + visuals).

CLI:

- `--config`
- `--datasets`
- `--max-images`
- `--run-name`
- `--dry-run`
- `--detector-input-mode file_path|tensor`
- `--detection-imgsz 1280`
- `--save-restored-full-resolution`
- `--metrics-parity rtts_script`

Outputs:

- Under `outputs/<checkpoint_name>_<yolo_weight>/`:
  - `restored_images/<dataset>/<image_id>.png`
  - `labels/yolo_txt/<dataset>/{original,restored}/`
  - `labels/voc_xml/<dataset>/{original,restored}/`
  - `metrics.json`
  - `metrics.csv`
  - `visuals/*.jpg`
  - `metrics/<dataset>_per_image.csv`
  - `metrics/<dataset>_summary.json`
  - `metrics/all_datasets_summary.json`

If the target directory already exists, the evaluator creates the next available suffix such as `_2`, `_3`, and so on.

### 3.6 `scripts/extract_foggycityscape.py`

Purpose:

- Standalone extractor for FoggyCityscape.
- Converts Cityscapes polygon JSON into VOC XML bounding boxes.
- Extracts all valid `val/` foggy images.
- Includes every fog beta variant present.
- Drops images with no allowed object boxes.

CLI:

- `--source-root`
- `--output-root`
- `--fog-levels`
- `--split-filter`
- `--overwrite`
- `--dry-run`

Outputs:

- `Datasets/extracted/FoggyCityscape/images/`
- `Datasets/extracted/FoggyCityscape/labels_raw/`
- `Datasets/extracted/FoggyCityscape/manifests/extracted_pairs.csv`
- `Datasets/extracted/FoggyCityscape/manifests/summary.json`

### 3.7 `scripts/compare_panels.py`

Purpose:

- Build side-by-side contact-sheet panels from existing run visuals for report inclusion.
- Combines `original`, `restored`, `original_detection`, and `restored_detection` into a 3-row × 4-column panel (rows = runs, cols = visual types).

Inputs / assumptions:

- Expects the standard visual layout under `outputs/<run>/visuals/...` described in section 2.7b.
- Selects one shared filename per dataset to build a representative panel.
- Filenames must be present in each run's `original/` folder (shared across runs) for deterministic comparison.

Key features:

- Deterministic sampling via `--seed` (default `42`).
- Explicit filename overrides via repeated `--sample` arguments, or a file of overrides (e.g., `samples.txt`).
  - Inline override syntax: `DATASET=FILENAME`.
  - File format: plain lines with `DATASET=FILENAME`, comments allowed starting with `#`.
- Writes panels into `figs/report_comparisons/by_dataset/...` and a manifest `figs/report_comparisons/panels.json`.

CLI examples:

```bash
# Use deterministic random selection (default seed)
python scripts/compare_panels.py

# Override specific samples inline
python scripts/compare_panels.py \
  --sample RTTS=FogDr_Google_348.jpg \
  --sample FoggyCityscape=aachen_000032_000019_leftImg8bit_foggy_beta_0.01.jpg

# Read overrides from a file (samples.txt)
python scripts/compare_panels.py --sample samples.txt
```

Notes:

- If an override filename is not shared across all runs, the script raises an error to avoid mismatched comparisons.
- Use `--output-dir` to change where panels are written. Default: `figs/report_comparisons`.

## 4. Module-by-Module Guide (`pipeline/`)

### 4.1 `pipeline/common/`

- `config.py`:
  - `load_yaml_config(path)` loads and validates YAML root type.
- `paths.py`:
  - `get_project_root(config)` handles local/colab root switch.
  - `resolve_from_project_root(config, rel_or_abs_path)`.
  - `resolve_under(base_dir, rel_or_abs_path)`.

### 4.2 `pipeline/data/scan_and_pair.py`

Main function:

- `scan_and_pair_datasets(...)`

What it does:

- For each active dataset:
  - Index images and labels by stem.
  - Respect split file if present.
  - Pair image-label ids.
  - Materialize links/copies in organized folders.
  - Produce `pairs.csv` + `summary.json`.

Supported annotation formats:

- `yolo_txt`
- `voc_xml`

### 4.3 `pipeline/data/gt_normalize.py`

Main function:

- `normalize_ground_truth(...)`

What it does:

- Reads `pairs.csv`.
- Converts each raw label to YOLO normalized format.
- Writes normalized labels and `pairs_yolo.csv`.
- Writes per-dataset normalization stats.

Key converters:

- `_convert_yolo_txt(...)`
- `_convert_voc_xml(...)`

### 4.4 `pipeline/data/yolo_dataset.py`

Class:

- `OrganizedYoloDataset`

What it loads per sample:

- Original image path and original image dimensions
- RGB image from organized `images/` for restoration input
- Depth map from organized `DepthMaps/` (falls back to zero map if missing)
- YOLO normalized GT from `normalized_label_path`
- GT boxes in original-resolution XYXY and restoration-resolution XYXY

Transforms:

- Resizes restoration input to `vram.image_resolution`
- Leaves detector file-path input at original resolution in `file_path` mode

Collate helper:

- `collate_eval_batch(...)`

### 4.5 `pipeline/models/restoration_loader.py`

Classes/functions:

- `RestorationModelLoader`
- `RestorationRuntime.restore(rgb, depth)`

What it does:

- Dynamically imports restoration module and builder function.
- Loads checkpoint (with optional shape-compatible relaxed loading).
- Returns runtime object that restores batch tensors.

### 4.6 `pipeline/models/detection_loader.py`

Classes/functions:

- `DetectionModelLoader`
- `load_detection_inference_settings(config)`

What it does:

- Loads YOLO model from `detection.weights`.
- Builds predict kwargs (`conf`, `iou`, `max_det`, `imgsz`, `device`).
- Auto-fallbacks CUDA device to CPU if unavailable.

### 4.7 `pipeline/depth/generate_depthmaps.py`

Main function:

- `generate_depthmaps_for_datasets(...)`

What it does:

- Resolves organized dataset image folders.
- Auto-locates depth script at `scripts/depthmap-create.py`.
- Calls script via subprocess for each dataset.
- Returns per-dataset status and output paths.

### 4.8 `pipeline/eval/metrics.py`

Key functions:

- `box_iou_xyxy(...)`
- `extract_predictions_from_ultralytics(...)`
- `mean_iou_all_gt(...)`
- `mean_iou_matched(...)`
- `mean_iou_tp50(...)`
- `compute_map(...)`

Metric behavior:

- `mean_iou_all_gt`: average best class-matched IoU for every GT, unmatched as zero.
- `mean_iou_matched`: average best class-matched IoU for GT with overlap > 0.
- `mean_iou_tp50`: average true-positive IoU for matches >= 0.50.
- `compute_map`: AP per class and IoU threshold (101-point interpolation), then mean across classes and thresholds. `map_class_set` controls GT-only vs. GT + prediction classes.

### 4.9 `pipeline/eval/visualize.py`

Functions:

- `save_separate_visuals(...)`
- `save_visual_batch(...)`

Visual output (4 separate image types per sample):

- **`original/`**: Original image (no restoration)
- **`restored/`**: Restored image after dehazing
- **`original_detection/`**: Baseline YOLO predictions drawn on original
- **`restored_detection/`**: Restored YOLO predictions drawn on restored

DAWN uses the same visual layout as other datasets.

Supports per-dataset visual quotas via `max_visuals` config.

### 4.10 `pipeline/run/metrics_utils.py`

Utility functions:

- `compute_psnr(img_ref, img_rest)` — Peak Signal-to-Noise Ratio
- `compute_ssim(img_ref, img_rest, use_multichannel=False)` — Structural Similarity Index

Uses scikit-image if available; fallback implementations provided.

Default: luminance-only SSIM for perceptual alignment. Set `metrics.use_multichannel_ssim: true` in config for full RGB SSIM.

### 4.11 `pipeline/eval/report.py`

Function:

- `write_metrics_report(output_dir, report)`

Writes per-dataset and aggregated metrics (legacy structure, preserved for compatibility).

### 4.12 `pipeline/run/evaluate_pipeline.py`

Main function:

- `evaluate_pipeline(...)`

End-to-end workflow:

- Build output run dir (`outputs/<run_name>/`)
- Build dataset/dataloader
- Load restoration and detection models
- For each batch:
  - In `file_path` mode, run YOLO on original image paths
  - Restore images (if `restoration.enabled`) and save restored full-resolution images
  - Run YOLO on restored image paths
  - In `tensor` mode, run YOLO on resized numpy arrays for backward compatibility
  - Collect predictions/GT and compute per-image metrics:
    - PSNR, SSIM (restoration quality)
    - IoU all-GT, matched IoU, TP50 IoU, AP50, precision/recall/F1
    - Num GT / Num predictions (detection counts)
  - Save 4-type visuals with per-dataset quotas
  - Accumulate per-image rows
- Write per-dataset metrics:
  - `metrics/<dataset>_per_image.csv` — one row per image
  - `metrics/<dataset>_summary.json` — aggregated stats plus dataset-level baseline/restored/improvement detection metrics
- Write global summary:
  - `metrics/all_datasets_summary.json` — aggregated across all datasets, plus an `__overall__` run-level rollup
- Compute run-level quality metrics:
  - `quality.psnr` and `quality.ssim` are averaged from the dataset means first, then equally averaged across datasets
- Compute and report overall baseline/restored metrics
- Return summary dict with `processed_images`, `visuals_saved`, `metrics_dir`, etc.

## 5. Complete Execution Playbook

### 5.1 Minimal full run sequence

1. Scan and pair:

```bash
python scripts/step1_scan_datasets.py --config configs/pipeline.yaml --datasets DAWN RTTS
```

2. Normalize labels:

```bash
python scripts/step2_normalize_gt.py --config configs/pipeline.yaml --datasets DAWN RTTS
```

3. Generate depth maps:

```bash
python scripts/step3_generate_depthmaps.py --config configs/pipeline.yaml --datasets DAWN RTTS
```

4. Evaluate full pipeline:

```bash
python scripts/step4_evaluate_pipeline.py --config configs/pipeline.yaml --datasets DAWN RTTS --run-name eval-rtts-dawn
```

### 5.2 Dry-run checks

Step 3 dry-run:

```bash
python scripts/step1_scan_datasets.py --config configs/pipeline.yaml --datasets DAWN RTTS --dry-run
```

Step 5 dry-run:

```bash
python scripts/step3_generate_depthmaps.py --config configs/pipeline.yaml --datasets DAWN RTTS --dry-run
```

Step 7 dry-run:

```bash
python scripts/step4_evaluate_pipeline.py --config configs/pipeline.yaml --datasets DAWN RTTS --max-images 8 --dry-run
```

## 6. Expected Outputs and Their Meaning

### 6.1 Data preparation outputs

- `Datasets/organized/<Dataset>/images/`
  - Organized image files used by evaluator.
- `Datasets/organized/<Dataset>/labels_raw/`
  - Original labels copied/linked into organized tree.
- `Datasets/organized/<Dataset>/manifests/pairs.csv`
  - Raw pairing manifest.
- `Datasets/organized/<Dataset>/labels_yolo/`
  - Normalized YOLO GT labels.
- `Datasets/organized/<Dataset>/manifests/pairs_yolo.csv`
  - Evaluator-ready manifest (includes normalized label path).
- `Datasets/organized/<Dataset>/DepthMaps/`
  - Depth maps for restoration input.
- `Datasets/extracted/FoggyCityscape/`
  - Standalone extracted input tree created by `scripts/extract_foggycityscape.py`.
  - Contains `images/`, `labels_raw/`, and extractor manifests.

### 6.2 Evaluation outputs

Directory structure:

```
outputs/<run_name>/
├── restored_images/
│   └── <dataset>/<image_id>.png
├── labels/
│   ├── yolo_txt/<dataset>/{original,restored}/
│   └── voc_xml/<dataset>/{original,restored}/
├── visuals/
│   ├── <dataset>/
│   │   ├── original/
│   │   ├── restored/
│   │   ├── original_detection/
│   │   └── restored_detection/
│
└── metrics/
    ├── <dataset>_per_image.csv
    ├── <dataset>_summary.json
    ├── all_datasets_summary.json
    ├── metrics.json
    └── metrics.csv
```

Metrics content:

**Per-image CSV** (`<dataset>_per_image.csv`):

- Stable columns include: `dataset`, `image_id`, `image_path`, `restored_image_path`, `original_width`, `original_height`, `num_gt`, `num_pred_original`, `num_pred_restored`, `original_iou_all_gt`, `restored_iou_all_gt`, `original_iou_matched`, `restored_iou_matched`, `original_iou_tp50`, `restored_iou_tp50`, `original_ap50`, `restored_ap50`, `precision_original`, `recall_original`, `f1_original`, `precision_restored`, `recall_restored`, and `f1_restored`
- Legacy aliases such as `baseline_mean_iou_tp50`, `restored_mean_iou_tp50`, and `mean_iou` remain for compatibility.
- One row per image

**Summary JSON** (`<dataset>_summary.json`):

- Aggregated statistics include `psnr`, `ssim`, `mean_iou_tp50`, `mean_iou_tp50_mean`, and explicit IoU fields.
- Detection comparisons: `baseline`, `restored`, `improvement`
- `baseline` and `restored` carry dataset-level `iou` (`all_gt`, `matched`, `tp50`, `primary`), legacy `mean_iou`, mAP, `map_by_iou`, and `map_class_set`.
- `improvement` stores restored-minus-baseline deltas for `iou_all_gt`, `iou_matched`, `iou_tp50`, legacy `mean_iou`, and mAP.

**All datasets summary** (`all_datasets_summary.json`):

- Aggregated statistics across all evaluated datasets
- Includes an `__overall__` section with the run-level quality summary and detection summary.

**Legacy outputs** (preserved for compatibility):

- `metrics.json` — Full metrics payload with `baseline.mean_iou_tp50`, `baseline.mean_iou`, `baseline.map`, `baseline.map_by_iou`, `restored.*`, `improvement.*`, `quality.psnr`, `quality.ssim`, detection config, `processed_images`, `visuals_saved`
- `metrics.csv` — Compact table with `category,metric,baseline,restored,delta,value`; detection rows fill the comparison columns, while quality rows use `value` for `psnr` and `ssim`

## 7. How to Improve Scorings

Below are practical tuning strategies mapped to observed output behavior.

### 7.1 If both baseline and restored mAP are low

Actions:

1. Upgrade `detection.weights` to stronger detector.
2. Increase `detection.imgsz` in file-path mode, or `vram.image_resolution` in tensor mode, if memory allows.
3. Tune `detection.conf_threshold` and `detection.nms_iou_threshold`.
4. Verify `datasets.entries.<name>.class_map` correctness.

Why:

- Detector quality and preprocessing scale dominate absolute mAP.

### 7.2 If baseline is good but restored is worse

Actions:

1. Verify `restoration.model_module` matches `restoration.checkpoint` architecture family.
2. Check `restoration.output_index` picks the correct final restored tensor.
3. Ensure depth maps exist and are valid (not all zeros).
4. Try `restoration.enabled=false` as ablation sanity check.

Why:

- Mismatch in restoration stage can inject artifacts that hurt detection.

### 7.3 If `improvement.map` is near zero

Actions:

1. Expand `evaluation.max_images` to full dataset for stable estimate.
2. Inspect `visuals/` for qualitative restoration gains/losses.
3. Test alternate restoration checkpoints in `restoration.checkpoint`.
4. Tune detector thresholds specifically for restored images.

Why:

- Small sample variance and threshold mismatch can hide real gains.

### 7.4 If class-wise behavior is uneven

Actions:

1. Check class distribution and rare classes in GT.
2. Verify `class_map` and normalization conversion stats.
3. Consider class-specific detector retraining/fine-tuning.

Why:

- mAP is average across classes; class mismatch heavily distorts final score.

### 7.5 If runs are unstable (OOM / slow)

Actions:

1. Reduce `vram.batch_size`.
2. Reduce `vram.image_resolution`.
3. Set `runtime.device: cpu` for correctness run when GPU unavailable.
4. Reduce `evaluation.max_images` for quick iteration.

Why:

- Throughput stability is required for reproducible scoring.

### 7.6 If you want multi-GPU speedup

Actions:

1. Set `runtime.gpu_ids` to the GPU list you want, for example `[0, 1]`.
2. Set `runtime.enable_data_parallel: true`.
3. Set `runtime.device` to the first CUDA device you want to anchor on, for example `cuda:0`.
4. Keep `vram.batch_size` high enough to benefit from multiple GPUs, but low enough to avoid OOM.

What happens now:

- Restoration model uses `DataParallel` across the listed GPUs.
- YOLO detection gets a multi-GPU device string.
- Depth generation stays single-GPU on the first device, so it remains compatible with existing runs.

Limit:

- Depth stage is not yet sharded across GPUs. To parallelize it, the code would need dataset splitting and per-GPU workers or a DDP-style worker launch.

## 8. Scoring-Sensitive Configs (Quick Reference)

Highest impact:

- `detection.weights`
- `detection.conf_threshold`
- `detection.nms_iou_threshold`
- `vram.image_resolution`
- `restoration.model_module`
- `restoration.checkpoint`
- `restoration.output_index`
- `datasets.entries.*.class_map`
- `normalization.skip_unknown_classes`
- `evaluation.map_iou_thresholds`

Medium impact:

- `restoration.use_depth`
- `detection.max_det`
- `evaluation.max_images` (stability, confidence in score)

Low/direct score impact (mostly speed/ops):

- `runtime.num_workers`
- `vram.batch_size` (unless OOM causes run failures)

## 9. Typical Troubleshooting

1. Missing `pairs_yolo.csv`:

- Run Step 4 after Step 3.

2. Depth map warnings / missing depth files:

- Run Step 5.
- Current dataset loader falls back to zero depth if missing.

3. Checkpoint load mismatch:

- Ensure `restoration.model_module` and `restoration.checkpoint` pair correctly.
- Keep `strict_state_dict=false` for relaxed compatible loading if needed.

4. CUDA errors:

- Set `runtime.device: cpu` or reduce resolution/batch.

6. Multi-GPU not used by one stage:

- Restoration and detection can scale with the current config.
- Depth generation remains single-GPU by design in the current implementation.
- If you need full multi-GPU depth throughput, that stage needs a separate sharding implementation.

## 10. Suggested Reproducible Workflow

For experiment tracking:

1. Keep one config copy per experiment (e.g., duplicate YAML with variant name).
2. Use `--run-name` only as a label if you want one recorded in `metrics.json`.
3. Archive:
   - config file snapshot
   - `metrics.json`
   - representative `visuals/`
4. Compare baseline/restored deltas across runs using `metrics.csv`.

---

If you add a new dataset format in the future, implement a new converter/scanner entry in:

- `pipeline/data/scan_and_pair.py` (if format needs custom scan behavior)
- `pipeline/data/gt_normalize.py` (if annotation conversion differs from YOLO/VOC)

Then add a dataset block under `datasets.entries` in YAML.
