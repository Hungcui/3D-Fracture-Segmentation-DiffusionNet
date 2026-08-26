# Surface Segmentation for 3D Cultural-Heritage Fracture Objects using DiffusionNet

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Bachelor Thesis / Research Project**  
> **Department:** Department of Information and Communication Technology, University of Science and Technology of Hanoi (USTH)  
> **Supervisor:** Mr. Huynh Vinh Nam (ICT Lab, USTH)

---

## 📌 Project Overview

Restoring fragmented cultural heritage artifacts (such as ancient pottery and ceramic vessels) is a critical task in digital heritage preservation. Reassembling broken fragments manually by hand is slow and risks damaging fragile artifacts. 

This project develops an autonomous **3D surface learning pipeline** using **DiffusionNet** to classify each vertex on a 3D mesh into binary semantic classes:
- **`0` = Original Surface** (smooth, intact outer/inner vessel surface)
- **`1` = Fracture Surface** (rough, unglazed break site)

Isolating the fracture regions dramatically narrows the search space for virtual reassembly algorithms, accelerating automatic 3D fragment matching.

---

## 🚀 Key Features & Contributions

1. **Spectral Geometric Representation (LBO):** Computes discrete Laplace-Beltrami Operators using cotangent weights ($k=128$ spectral basis) to achieve rotation- and pose-invariant surface features.
2. **20-Channel Joint Feature Descriptor:** Combines spatial coordinates (3 ch), multi-scale Heat Kernel Signatures (HKS, 16 ch), and local mean curvature (1 ch). Log-compression and Z-score standardization prevent feature explosion at sharp corners.
3. **Class-Weighted & Soft Dice Hybrid Loss:** Combines class-weighted Cross-Entropy ($w=[1.0, 3.0]$) and Soft Dice Loss to overcome severe class imbalance (fracture boundaries represent only 5%–15% of mesh vertices).
4. **Synthetic-to-Real Transfer Learning Pipeline:** 
   - *Stage 1:* Pre-training on 1,640 synthetic fracture meshes from the **Breaking Bad** dataset (Artifact subset).
   - *Stage 2:* Two-phase fine-tuning on real archaeological 3D scans from the **Fantastic Breaks** dataset.
5. **Interactive 3D Visualization Demo:** Interactive PyVista GUI allowing instant selection, rendering, and fracture prediction on sample 3D broken objects.

---

## 📊 Evaluation & Results

| Configuration / Dataset | Acc | Precision | Recall | F1-Score | Fracture IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Breaking Bad (Synthetic Pre-training)** | 0.892 | 0.524 | 0.744 | 0.615 | **0.444** |
| **Breaking Bad (Decision Threshold 0.6)** | 0.901 | 0.582 | 0.687 | 0.631 | **0.461** |
| **Fantastic Breaks (Zero-Shot)** | 0.520 | 0.007 | 0.820 | 0.014 | 0.007 |
| **Fantastic Breaks (From Scratch)** | 0.841 | 0.380 | 0.710 | 0.492 | **0.326** |
| **Fantastic Breaks (Two-Phase Fine-Tuned)** | 0.850 | 0.375 | 0.690 | 0.485 | **0.320** |

*Note: Pre-training acts as a strong geometric regularizer during real-world transfer learning, ensuring smooth and stable optimization.*

---

## 📂 Repository Structure

```
3D-Fracture-Segmentation-DiffusionNet/
├── README.md                           # Project documentation
├── requirements.txt                    # Dependencies
├── .gitignore                          # Git ignore configuration
│
├── demo/                               # Interactive 3D GUI Demo
│   ├── demo.py                         # PyVista 3D interactive viewer script
│   ├── build_demo.py                   # Standalone demo package generator
│   ├── model_bb.pt                     # Checkpoint: Pre-trained Breaking Bad model
│   ├── model_fb.pt                     # Checkpoint: Fine-tuned Fantastic Breaks model
│   └── data/                           # Sample .npz 3D meshes (00_bb..05_fb)
│
├── notebooks/                          # Training & Evaluation Notebooks
│   ├── diffnet_starter.ipynb           # Basic pipeline setup & verification
│   ├── diffnet_training.ipynb          # Stage 1 Synthetic pre-training notebook
│   ├── fantastic_breaks_transfer.ipynb # Stage 2 Real-world transfer fine-tuning
│   ├── test_models.ipynb               # Evaluation & benchmark scripts
│   ├── bb_decompress.ipynb             # Data decompression utilities
│   └── bb_decompress_kaggle.ipynb      # Kaggle dataset preprocessing
│
├── src/                                # Core Python Source Modules
│   ├── breaking_bad_loader.py          # Dataset loader & auto-labeler
│   ├── synthetic_fracture.py           # Synthetic fracture generator
│   └── visualize.py                    # PyVista / Matplotlib visualizer
│
└── data_samples/                       # Sample 3D OBJ & NPY data files
```

---

## 💻 Quickstart & Interactive Demo

### 1. Prerequisites & Installation

Clone this repository and install requirements:

```bash
git clone https://github.com/<your-username>/3D-Fracture-Segmentation-DiffusionNet.git
cd 3D-Fracture-Segmentation-DiffusionNet

pip install -r requirements.txt
```

### 2. Running the Interactive 3D Demo

Launch the PyVista interactive 3D visualizer to inspect sample meshes and predict fracture regions in real time:

```bash
cd demo
python demo.py
```

*Inside the interactive demo:*
- Select samples (e.g. `00_bb` synthetic vessel or `03_fb` real ceramic scan).
- Toggle between **Ground Truth** and **Model Predictions**.
- View fracture surfaces highlighted in red.

---

## 🛠 Tech Stack

- **Deep Learning Framework:** PyTorch
- **Surface Spectral Learning:** [DiffusionNet](https://github.com/nmwsharp/diffusion-net)
- **3D Geometry Processing:** `potpourri3d`, `robust_laplacian`, `scipy`
- **3D Rendering & Visualization:** `pyvista`, `matplotlib`

---

## 📜 License

This repository is licensed under the [MIT License](LICENSE).
