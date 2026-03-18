# GAN Training Stability Under Data Scarcity
### Evaluating DCGAN and WGAN-GP Across Data Size and Density Regimes

**MSc Dissertation Project | Distinction**
**MSc Artificial Intelligence and Data Science | January 2026**

---

## Overview

This repository contains the full experimental code for my MSc dissertation:

> *"Evaluating GAN Training Stability Across Data Size and Density 
>  Regimes: How Little Data Is Too Little?"*

The research investigates a fundamental problem in scientific imaging: 
GANs perform well on large natural image datasets but their behaviour 
under extreme data scarcity is poorly understood. Scientific domains 
such as cell biology and colloidal physics routinely produce datasets 
of only a few hundred to a few thousand images. This study asks — 
at what point does a GAN stop learning, and does architecture choice 
change that threshold?

---

## Research Question

> How much data is too little for stable GAN training, and how do 
> dataset density and GAN architecture interact to influence training 
> stability under limited scientific imaging data?

---

## Experimental Design

**16 controlled experiments** comparing two architectures across two 
dataset types and four data availability levels.

| Variable | Values |
|---|---|
| Architecture | DCGAN, WGAN-GP |
| Dataset Type | Sparse (idr0088), Dense (idr0139) |
| Data Availability | 100%, 50%, 25%, 10% |

---

## Datasets

RGB fluorescence microscopy datasets sourced from the 
**Image Data Resource (IDR)**:

- **idr0088** — Sparse configuration. ~4,380 images at 733 × 616px. 
  Fewer objects per frame, large background regions.
- **idr0139** — Dense configuration. ~4,925 images at 616 × 616px. 
  Overlapping structures, higher visual complexity.

**Preprocessing pipeline:**
- 128 × 128 patch extraction (stride = 128) to maximise sample count
- Normalisation to [-1, 1] to align with Tanh generator output
- 80:20 train/validation split with strict file-level separation
- TensorFlow tf.data pipeline with batching (batch size = 64) 
  and prefetching

---

## Model Architectures

### DCGAN
- Baseline architecture using Jensen-Shannon divergence
- Four-layer deep convolutional structure
- Known to suffer mode collapse in low-data regimes
- Learning rate: 0.0002 | Adam betas: (0.5, 0.999)

### WGAN-GP
- Robust alternative using Wasserstein loss and gradient penalty
- Enforces Lipschitz continuity for stable gradients
- Gradient penalty coefficient λ = 10
- Critic updated 4 times per generator update (N_CRITIC = 4)
- Learning rate: 0.0001 | Adam betas: (0.0, 0.9)

Both architectures use: latent dimension 128, image size 128×128×3, 
batch size 64, 100 epochs, 500 steps per epoch.

---

## Evaluation Metrics

- **FID (Frechet Inception Distance)** — distributional similarity 
  between real and generated images. Lower is better.
- **Precision** — sample fidelity
- **Recall** — mode coverage and distributional diversity
- **Qualitative visual inspection** — 4×4 image grids at best FID epoch

> **Key finding on metrics:** FID alone is insufficient in sparse, 
> patch-based regimes. DCGAN was found to artificially lower its FID 
> score by generating blank patches matching background statistics 
> while recall had collapsed — a finding that exposes a significant 
> limitation in single-metric GAN evaluation.

---

## Results Summary

| Model | Dataset | Data Size | Best FID | Precision | Recall |
|---|---|---|---|---|---|
| DCGAN | Sparse | 100% | 42.01 | 0.9543 | 0.8048 |
| DCGAN | Sparse | 10% | 138.67 | 0.8636 | 0.2500 |
| WGAN-GP | Sparse | 100% | 35.99 | 0.9612 | 0.8201 |
| WGAN-GP | Sparse | 10% | 85.66 | 0.9545 | 0.8750 |
| DCGAN | Dense | 10% | 174.75 | 0.7980 | 0.1919 |
| WGAN-GP | Dense | 10% | 102.30 | 0.8181 | 0.4747 |

**WGAN-GP outperformed DCGAN in 100% of low-data conditions tested.**

At 10% sparse data: WGAN-GP maintained recall of 0.875 while DCGAN 
collapsed to recall of 0.25 — a 3.5x difference in distributional 
coverage despite DCGAN appearing competitive on FID alone.

---

## Key Findings

1. **WGAN-GP is significantly more robust** than DCGAN under data 
   scarcity, maintaining stable training dynamics even at 10% 
   data availability.

2. **The FID Trap** — DCGAN games the FID metric in sparse regimes 
   by generating blank, low-information patches that match background 
   statistics. Recall exposes this failure while FID conceals it.

3. **No universal minimum dataset size exists** — data sufficiency 
   depends on the interaction between architecture, dataset density, 
   and patch-level information content.

4. **Dataset density matters independently of size** — dense datasets 
   support stable critic learning but increase modelling difficulty, 
   producing higher FID even when training is stable.

5. **WGAN-GP is a viable tool** for data augmentation in scientific 
   domains where acquisition is technically constrained.

---

## Repository Structure
```
├── notebooks/
│   ├── DCGAN_Sparse_Dataset.ipynb
│   ├── DCGAN_Dense_Dataset.ipynb
│   ├── WGAN_GP_Sparse_Dataset.ipynb
│   └── WGAN_GP_Dense_Dataset.ipynb
├── results and images/
│   ├── sparse_generated_samples/
│   └── dense_generated_samples/
├── requirements.txt
└── README.md
```

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/oluwamuyiwajaiyeola/gan-stability-data-scarcity
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download datasets from the Image Data Resource:
   - Sparse: https://idr.openmicroscopy.org/search/?query=Name:idr0088
   - Dense: https://idr.openmicroscopy.org/search/?query=Name:idr0139

4. Open and run notebooks in order within Google Colab or Jupyter

---

## Technologies

Python | TensorFlow | NumPy | Pandas | Matplotlib | Seaborn | 
Scikit-learn | Google Colab

---

## References

- Arjovsky et al. (2017) — Wasserstein GAN
- Goodfellow et al. (2014) — Generative Adversarial Networks
- Gulrajani et al. (2017) — Improved Training of WGANs
- Karras et al. (2019, 2020) — StyleGAN
- Radford et al. (2015) — DCGAN

---

## Author

**Oluwamuyiwa Peter Jaiyeola**
> github.com/oluwamuyiwajaiyeola
> 
> linkedin.com/in/oluwamuyiwa-jaiyeola
