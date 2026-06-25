# Semantic Image Segmentation on COVID-19 CT Scans

**Student Name:** Athina Andri 
**Student ID (Α.Μ.):** 25001 
**Course:** Deep Learning 
**Program:** MSc in Artificial Intelligence & Visual Computing

---

## Project Overview
This project focuses on applying Deep Learning techniques for automated semantic image segmentation of chest Computed Tomography (CT) scans. 
Utilizing the **COVID-19 CT Lung and Infection Segmentation Dataset**, the objective is to automatically identify and delineate lung fields as well as regions affected by COVID-19 associated infections.

## Installation & Execution Guide
1. **Clone the repository:**
   ```bash
   git clone (https://github.com/andriathina/COVID19-CT-Semantic-Segmentation)

---

## Performance Comparison, Capabilities & Limitations

### 1. Quantitative Evaluation
The quantitative evaluation of both architectures on the validation dataset over 5 training epochs demonstrates the explicit advantages of utilizing transfer learning methodologies:

| Performance Metric | Model 1: Standard U-Net (ResNet34) | Model 2: Transfer Learning (MobileNetV2) |
| :--- | :---: | :---: |
| **Optimal Training Epoch** | 4 | 4 |
| **Validation Loss** | 0.3520 | **0.2906** |
| **Validation Accuracy** | 98.72% | **99.10%** |
| **Dice Coefficient (Macro)** | 97.44% | **98.20%** |
| **Intersection over Union (IoU)** | 95.01% | **96.45%** |

---

### 2. Capabilities & Strengths

#### Model 2: U-Net with MobileNetV2 (Transfer Learning)
* **Rapid Convergence:** Initializing the encoder with weights pre-trained on ImageNet allowed the network to achieve an outstanding macro Dice Score of 97.39% from the very first epoch, proving that low-level features (edges, textures) translate effectively to medical imaging domains.
* **Stable Optimization:** The validation loss decreased smoothly across consecutive epochs (0.8212 → 0.4750 → 0.3510 → 0.2906) without showing any signs of overfitting or learning fluctuations.
* **Excellent Spatial Alignment:** The model recorded a top-tier Intersection over Union (IoU) of 96.45%, indicating highly accurate pixel-level overlapping boundaries with the expert ground truth masks.

#### Model 1: Standard U-Net (From Scratch)
* **Autonomous Adaptability:** Despite starting with completely randomized weights (`encoder_weights=None`), the model reached a very competitive 97.44% Dice Score by Epoch 4, demonstrating that the hybrid Cross-Entropy and Multiclass Dice Loss function is remarkably robust at guiding a model from scratch.

---

### 3. Limitations & Weaknesses

#### Pixel-Level Error Analysis (Infection Class Challenge)
According to the computed Normalized Confusion Matrix for our optimal MobileNetV2 U-Net, clear clinical and statistical limitations are identified:
* **The Infection Class Bottleneck:** While the model exhibits near-perfect segmentation for the Background (1.00), Left Lung (0.96), and Right Lung (0.95) pixels, its ability to perfectly isolate the COVID-19 Infection class is limited to a true positive rate of **0.60 (60%)**.
* **Partial Volume Effects & Ill-Defined Boundaries:** The matrix shows that 18% of true infection pixels are misclassified as Left Lung tissue and 15% as Right Lung tissue. In chest CT scans, COVID-19 manifestations (such as Ground-Glass Opacities) display highly diffuse, ambiguous, and non-geometric boundaries that naturally blend into surrounding healthy lung tissue.
* **Severe Class Imbalance:** The actual infection pixels occupy a tiny fraction of the total spatial volume compared to the lung cavities and the dominant background, rendering it difficult for the network to refine sub-visual pathogen features despite using a targeted Dice Loss.

#### Architectural & Data Limitations
* **Channel Redundancy:** To maintain strict structural compatibility with ImageNet pre-trained encoders, the 1-channel grayscale CT slices had to be duplicated into 3 channels via `image_tensor.repeat(3, 1, 1)`, artificially inflating memory consumption.
* **Loss of 3D Contextual Continuity:** Slicing the native 3D NIfTI patient volumes into independent 2D arrays ignores the volumetric continuity and spatial dependency that exists between consecutive vertical slices of the chest cavity.

---

### Conclusion
The integration of Transfer Learning via the MobileNetV2 backbone provides the most reliable approach for this clinical classification task. It minimizes computational latency and drastically accelerates optimization stability, delivering a lower validation loss (0.2906 vs 0.3520) which is critical when dealing with limited medical datasets.
