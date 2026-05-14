# Positional Encodings Anchor Spatial Structure in Vision Transformers: A Geometric Perspective on Robustness

**Mahmoud Mannes** — Independent Researcher  
[[Paper]](paper/main.pdf)

---

## Overview

This repository contains the paper, pretrained models, intervention code, and figures for our investigation into how positional encodings shape representational geometry and corruption robustness in Vision Transformers (ViTs).

The central question motivating this work is not whether positional encodings improve performance — they do — but *how* they do so at the level of internal representations. We argue that positional encodings function less as simple position injectors and more as mechanisms that stabilize a spatial reference frame within the model's residual stream. Models lacking a stable reference frame learn nontrivial spatial structure, but that structure is fragile: it collapses under positional perturbation and correlates poorly with robustness to image corruptions.

---

## Key Contributions

- **SSDC (Spatial Similarity Distance Correlation):** a permutation-sensitivity probe measuring the Spearman rank correlation between pairwise spatial distances and representational similarities among unit-normalized residual stream tokens. SSDC quantifies how faithfully a model's internal geometry reflects the spatial layout of the input.

- **RPI (Random Permutation at Inference):** an inference-time intervention that randomly permutes patch tokens, severing index-based positional signals while preserving content. Used to isolate whether spatial structure is anchored to position indices or encoded in content representations.

- **RPT (Random Permutation Training):** a training-time variant that removes positional reference entirely during training. The RPT inference result (restoring ordered patches at inference time recovers medium-to-high SSDC) reveals latent content-based spatial structure. It serves as a key dissociation experiment showing that the mere presence of positional encodings is not enough for preserving robustness to content-based distributional shifts.

- **Depth-wise analysis across PE schemes:** we track SSDC across transformer depth for APE, Sinusoidal PEs, RoPE, an ablated (no-PE) model, and RPT, revealing qualitatively distinct organizational trajectories. RoPE accumulates spatial structure progressively across depth; APE and Sinusoidal PEs inject it sharply at early layers.

---

## Main Findings

- ViTs trained without positional encodings learn non-trivial spatial structure, but it collapses under RPI, indicating it is index-anchored rather than content-based.
- Models with stable positional reference frames (particularly RoPE) retain substantially stronger spatial organization under perturbation.
- Robustness to image corruptions tracks representational geometry more closely than clean accuracy, implicating reference frame stability as a driver of robustness.
- RoPE and APE/SinuPE show sharply different depth-wise SSDC trajectories, suggesting they instantiate qualitatively different mechanisms for encoding position.
- The RPT inference result (removing permutation at inference after training under permutation partially recovers spatial structure) establishes that content-based spatial organization emerges even in the absence of a positional reference frame, but requires that frame for robustness.

---

## Repository Structure

```
vit-encoding-paper/
├── paper/
│   ├── main.pdf                         # Main paper
│   ├── main.tex                         # LaTeX source
│   ├── checklist.tex                    # LaTeX source for checklist
    ├── neurips_2026.sty                 # Style file
    └── references.bib                   # Bibliography
    
├── code/
│   └── notebooks/
│       ├── interventions.ipynb          # RPI and RPT intervention pipeline
│       ├── ssdc_analysis.ipynb          # SSDC computation and depth-wise analysis
│       └── attention_maps.ipynb         # Attention map extraction and visualization
├── models/                              # See pretrained models below
└── figures/                             # All figures as used in the paper
```

---

## Pretrained Models

Trained checkpoints for all five PE variants (APE, SinuPE, RoPE, Ablated, RPT) are available via Google Drive:

**[[Download Pretrained Models]](https://drive.google.com/drive/folders/1L3yuTNPnaxNyNfqxRMPMk2xVnJNHCHgc?usp=drive_link)**

All models are ViT-S scale (patch size 16) trained on ImageNet-100. For each model condition, there are 4 different models trained on different random seeds.

To load a checkpoint:

```python
import torch

checkpoint = torch.load("path/to/checkpoint.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
```

---

## Experimental Setup

| Component | Details |
|---|---|
| Model | ViT-Small (patch size 16) |
| Dataset | ImageNet-100 |
| Robustness benchmarks | JPEG Compression and Gaussian Blur |
| PE variants | APE, SinuPE, RoPE, Ablated (no PE), RPT |
| Training hardware | TPU v5-1 |
| Framework | PyTorch / torch_xla |

---

## Reproducing the Analysis

**Dependencies:**

```bash
pip install torch torchvision timm numpy matplotlib scipy
```

Download the pretrained models from the link above and place them in the `models/` directory. Then open the notebooks in order:

1. `interventions.ipynb` — applies RPI and RPT to each model, computes SSDC under intervention
2. `ssdc_analysis.ipynb` — depth-wise SSDC plots across all PE variants, robustness correlation analysis
3. `attention_maps.ipynb` — attention map extraction and spatial structure visualization

Each notebook is self-contained with markdown explanation of the methodology at each step.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{mannes2025positional,
  title={Positional Encodings Anchor Spatial Structure in Vision Transformers: A Geometric Perspective on Robustness},
  author={Mannes, Mahmoud},
  year={2025}
}
```

---

## Notes on Scope and Limitations

This work studies a single model scale (ViT-S) and a single dataset (ImageNet-100). SSDC is a proxy metric for spatial organization and does not directly measure all aspects of representational geometry. Causal claims are made cautiously throughout: the experimental evidence strongly implicates reference frame stability as a driver of robustness, but the full mechanistic picture remains open. We discuss limitations and directions for future work in §5 of the paper.

---

*Conducted completely independently.*
