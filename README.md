# HandScript: Deep Learning Handwriting Generation System

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hosted_on_Render-success)](https://handscript-project.onrender.com/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/ahmaadtalal/HandScript-Project)

**Institution:** Military College of Signals, NUST[cite: 1]  
**Course:** Deep Learning (CS-405) Lab Project[cite: 1]  
**Developers:** Rafay Ali Shah, Mohad Saeed Hashmi, Ahmed Talal Sajid[cite: 1]

---

## 📖 Project Overview
HandScript is a deep learning-based handwriting generation system capable of creating photorealistic handwritten word images from typed text input[cite: 1]. The system was trained to handle the high variability of human handwriting by capturing style, content, and physical realism (such as ink strokes on paper)[cite: 1]. 

The final deployment is packaged as a fully functional Flask web application that renders full A4 pages of handwritten text in navy blue ink[cite: 1].

---

## ✨ Key Features
*   **Style-Conditioned Synthesis:** Generates text conditioned on a reference handwriting style[cite: 1].
*   **A4 Document Rendering:** Automatically formats input text into full A4 pages at 300 DPI (2480 x 3508 pixels) with proper word-wrapping, margins, and multi-paragraph support[cite: 1].
*   **Realistic Ink & Texture:** Simulates navy blue ink mimicking a ballpoint pen on paper, complete with Gaussian Blur and Unsharp Mask filtering for crisp, anti-aliased strokes[cite: 1].
*   **Natural Human Variation:** Implements variable character spacing (kerning), random rotation jitter (±1.2 degrees), and correct baseline alignment for ascenders and descenders[cite: 1].
*   **Configurable Slant:** Allows users to adjust the italic slant of the generated handwriting via a web interface[cite: 1].

---

## 🧠 Model Architecture
The core system is built on a Generative Adversarial Network (GAN) architecture consisting of five cooperating modules trained jointly[cite: 1]:

1.  **Style Encoder:** Extracts a 256-dimensional style vector (writer fingerprint) from a reference image using Convolutional layers[cite: 1].
2.  **Sequence Generator:** A 2-layer LSTM that combines character embeddings with the style vector to produce a stroke sequence[cite: 1].
3.  **Generator:** A ConvTranspose2d network that upsamples the LSTM output into a 32x128 grayscale image[cite: 1].
4.  **Discriminator:** Provides adversarial feedback by outputting real/fake probabilities per image[cite: 1].
5.  **Recogniser (OCR):** A CNN + BiLSTM network trained with CTC loss to provide legibility supervision to the generator[cite: 1].

---

## 📊 Datasets & Training Curriculum
### Datasets
*   **IAM Handwriting Word Database:** Used for GAN training. Consists of 38,305 labelled word images across 657 unique writers[cite: 1]. 
*   **EMNIST ByClass Split:** Used for the final high-fidelity rendering engine. A curated "Golden Index" dictionary was built from 124,800 characters to map specific, clean samples to the generation pipeline[cite: 1].

### Training Phases
The GAN was trained over 180 epochs on Kaggle using dual NVIDIA T4 GPUs[cite: 1]:
*   **Phase 1 (Epochs 1-60):** Standard GAN training with full OCR supervision to learn basic stroke shapes and legible letter groupings[cite: 1].
*   **Phase 2 (Epochs 61-150):** Annealed OCR weight to balance legibility with visual realism[cite: 1].
*   **Phase 3 (Epochs 151-180 - "Beauty School"):** OCR weight dropped to near-zero, allowing the generator to focus entirely on ink texture, stroke weight variation, and naturalistic pen motion[cite: 1].

---

## 🛠️ Tech Stack
*   **Backend:** Python, Flask, Gunicorn[cite: 1]
*   **Deep Learning:** PyTorch[cite: 1]
*   **Image Processing:** OpenCV, NumPy, Pillow (PIL)[cite: 1]
*   **Frontend:** HTML/CSS, JavaScript[cite: 1]
*   **Deployment:** Render (Cloud Hosting)

---

## 🚀 Running Locally
To run this project on your local machine:

**1. Clone the repository:**
```bash
git clone [https://github.com/ahmaadtalal/HandScript-Project.git](https://github.com/ahmaadtalal/HandScript-Project.git)
cd HandScript-Project
