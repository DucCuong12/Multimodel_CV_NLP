# FLUID: Flow-Latent Unified Integration via Token Distillation for Expert Specialization in Multimodal Learning

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](./SoICT__2025_Toán_rời_rạc_69.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## 📌 Overview
**FLUID** is a robust multimodal classification architecture that enhances the integration of visual and textual information through:
- **Q-Transforms** for token-level feature distillation
- **Contrastive alignment** for cross-modal consistency
- **Adaptive gated fusion** to dynamically balance modalities
- **Q-Bottleneck** for compact, task-specific representation
- **Lightweight Mixture-of-Experts (MoE)** for specialized output prediction

FLUID achieves **91% accuracy** on the challenging **GLAMI-1M** benchmark, outperforming strong multimodal baselines while demonstrating resilience to noise, imbalance, and semantic diversity.

---

## 📜 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage](#-usage)
- [Results](#-results)
- [Ablation Study](#-ablation-study)
- [Citations](#-citations)
- [License](#-license)

---

## 🚀 Key Features
- **Q-Transform:** Learnable query tokens to retain only salient token-level features.
- **Contrastive Fusion:** Enforces cross-modal alignment before fusion.
- **Gated Mechanism:** Learns adaptive weights for modality contributions.
- **Q-Bottleneck:** Selectively compresses fused features for better noise suppression.
- **Mixture-of-Experts:** Enhances prediction robustness through expert specialization.

---

## 🏗 Architecture
<p align="center">
  <img src="./assets/fluid_architecture.png" alt="FLUID Architecture" width="800">
</p>

FLUID follows a **two-stage fusion**:
1. **Feature Extraction:** ViT for images, mBERT for text → Q-Transform blocks → Contrastive alignment
2. **Fusion & Prediction:** Gated fusion → Q-Bottleneck → MoE classifier

---

## ⚙ Installation
```bash
# Clone this repository
git clone https://github.com/your-username/FLUID.git
cd FLUID

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
