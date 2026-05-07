# Comparative Analysis of YOLO Models for Indian Sign Language Detection

  

## Overview

This project performs a systematic comparative analysis of YOLOv5, YOLOv8, and YOLOv10 object detection models on an Indian Sign Language (ISL) gesture dataset. All model variants (nano, small, medium, large, extra-large) are trained and evaluated under identical conditions. A Min-Max normalization scoring framework ranks models by balancing detection accuracy and model efficiency.

---

## Dataset

- **Source:** Roboflow — Indian Sign Language 2024
- **Total Images:** 2,424 annotated images
- **Classes:** 29 ISL hand gestures
- **Split:** 88% train / 8% validation / 4% test
- **Format:** YOLOv7 (compatible with all YOLO versions used)
- **Input Resolution:** 640×640

---

## Models Trained

| Version | Variants |
|---------|----------|
| YOLOv5  | n, s, m, l, x |
| YOLOv8  | n, s, m, l, x |
| YOLOv10 | n, s, m, l, x |

**Total:** 15 models trained and compared.

---

## Key Results

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Size (MB) | Score |
|-------|-----------|--------|---------|--------------|-----------|-------|
| **YOLOv8n** | 0.9190 | 0.9330 | 0.9638 | 0.5897 | 6.25 | **0.9688** |
| YOLOv8s | 0.9322 | 0.9627 | 0.9712 | 0.5845 | 22.53 | 0.9663 |
| YOLOv8m | 0.9116 | 0.9591 | 0.9734 | 0.5901 | 52.05 | 0.9239 |
| YOLOv8l | 0.9269 | 0.9548 | 0.9741 | 0.5987 | 87.68 | 0.8920 |
| YOLOv8x | 0.9035 | 0.9751 | 0.9727 | 0.5966 | 136.74 | 0.8281 |
| YOLOv10l | 0.8089 | 0.8419 | 0.9028 | 0.5363 | 52.24 | 0.7884 |
| YOLOv10x | 0.7992 | 0.8874 | 0.8916 | 0.5326 | 64.17 | 0.7790 |
| YOLOv10m | 0.7212 | 0.7993 | 0.8443 | 0.4760 | 33.51 | 0.7074 |
| YOLOv10s | 0.6656 | 0.7855 | 0.8287 | 0.5048 | 16.54 | 0.7051 |
| YOLOv10n | 0.6896 | 0.5499 | 0.7061 | 0.4063 | 5.75 | 0.5663 |
| YOLOv5l | 0.8590 | 0.7642 | 0.8411 | 0.4470 | 93.06 | 0.6764 |
| YOLOv5x | 0.8643 | 0.8737 | 0.9101 | 0.4789 | 173.40 | 0.6572 |
| YOLOv5m | 0.7318 | 0.7128 | 0.7891 | 0.3961 | 42.36 | 0.6180 |
| YOLOv5s | 0.5607 | 0.5144 | 0.4975 | 0.2525 | 14.50 | 0.3421 |
| YOLOv5n | 0.5112 | 0.3562 | 0.3619 | 0.1775 | 3.89 | 0.2000 |

**Winner: YOLOv8n** — best balance of accuracy and model size (6.25 MB), achieving a normalized score of 0.9688.

---

## Project Structure

```
ISL_yolo/
│
├── yolov5s.ipynb               # Training notebook (YOLOv5 on Google Colab)
├── isl_yolo.ipynb              # Comparative analysis notebook (local)
│
├── yolov5n_results/            # YOLOv5 nano training output
├── yolov5s_results/            # YOLOv5 small training output
├── yolov5m_results/            # YOLOv5 medium training output
├── yolov5l_results/            # YOLOv5 large training output
├── yolov5x_results/            # YOLOv5 extra-large training output
│
├── yolov8n_results/            # YOLOv8 nano training output
├── yolov8s_results/            # ... (same pattern for s/m/l/x)
│
├── yolov10n_results/           # YOLOv10 nano training output
│   └── content/runs/detect/
│       └── yolov10n_isl_custom3/
│           ├── results.csv
│           └── weights/
│               ├── best.pt
│               └── last.pt
│
├── precision_recall_yolo_comparison.png    # Saved P-R curve plot
└── model_size_vs_map50.png                 # Saved size vs accuracy plot
```

Each `*_results/` folder follows the Colab export path and contains:
- `results.csv` — per-epoch training/validation metrics
- `weights/best.pt` — best checkpoint
- `weights/last.pt` — last checkpoint
- Confusion matrix, P/R/F1/PR curve plots, training batch images

---

## Link for the dataset used:
[Link](https://drive.google.com/file/d/1q3UkZ3Q0BjEfPHqn4ilntLVriOB6r5-0/view?usp=sharing)


## Training Configuration

| Parameter | Value |
|-----------|-------|
| Input size | 640×640 |
| Batch size | 16 (reduced for large models) |
| Epochs | 100 (early stopping, patience=10) |
| Optimizer | AdamW |
| Learning rate | 0.01 with cosine annealing + warmup |
| Weight decay | 5×10⁻⁴ |
| Platform | Google Colab (NVIDIA Tesla T4 GPU) |
| Framework | PyTorch + Ultralytics |

**Augmentations:** Mosaic, MixUp, horizontal flip, scale jittering, rotation ±15°, color jitter.

---

## Scoring Methodology

All models are ranked using **Min-Max normalization** across five dimensions:

- Precision (higher is better)
- Recall (higher is better)
- mAP@0.5 (higher is better)
- mAP@0.5:0.95 (higher is better)
- Model size in MB (smaller is better — score is inverted)

```
normalized = (value - min) / (max - min)
final_score = mean(precision_norm, recall_norm, mAP50_norm, mAP5095_norm, size_norm_inverted)
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `yolov5s.ipynb` | Trains all YOLOv5 variants (n/s/m/l/x) on Google Colab using the Roboflow ISL dataset |
| `isl_yolo.ipynb` | Loads saved result CSVs locally, generates comparative plots, computes normalized scores and final ranking |

---

## Requirements

See `Instruction to run.txt` for full setup and execution details.

- Python 3.9+
- PyTorch
- Ultralytics (`pip install ultralytics`)
- Roboflow (`pip install roboflow`)
- pandas, matplotlib, numpy, opencv-python

---

## Architecture Highlights

| Feature | YOLOv5 | YOLOv8 | YOLOv10 |
|---------|--------|--------|---------|
| Detection head | Anchor-based | Anchor-free | NMS-free (dual-head) |
| Backbone | CSPDarknet | C2f | CSPNet variant |
| Post-processing | NMS | NMS | End-to-end |
| Best model here | YOLOv5x | YOLOv8n | YOLOv10l |
