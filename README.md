# Pneumonia X-Ray Classifier
**SciEncephalon AI · Summer Intern Series 2026**

A deep-learning image classifier that analyzes chest X-ray images and predicts **Normal vs. Pneumonia** — built with PyTorch transfer learning and deployed via a Gradio web UI.

> **Education Only.** This project is for learning purposes. Real clinical AI requires FDA clearance, prospective validation, and significantly more data.

---

## Overview

This project walks through a full machine learning pipeline:
- Loading and preprocessing real medical imaging data (Kaggle Chest X-Ray dataset)
- Training a ResNet18 model using transfer learning
- Evaluating with healthcare-aware metrics (sensitivity, specificity, AUC)
- Serving predictions through a drag-and-drop Gradio web interface

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 89% |
| Sensitivity (pneumonia recall) | 97.4% |
| Specificity (normal recall) | 75.2% |
| AUC | 0.951 |

Trained on 5,216 chest X-ray images over 5 epochs on CPU using transfer learning from a pretrained ResNet18.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| Modeling | PyTorch + pretrained ResNet18 |
| Data | Kaggle "Chest X-Ray Images (Pneumonia)" by Paul Mooney |
| Visualization | PyEcharts (training curves, confusion matrix, sensitivity gauge) |
| UI | Gradio |
| Environment | Python virtual environment (.venv) |

---

## Project Structure

```
pneumonia-xray-classifier/
    02_pneumonia_xray_classifier.ipynb   # main notebook — training pipeline
    app.py                               # Gradio inference UI
    model.pt                             # saved model weights (not tracked in git)
    chest_xray/                          # Kaggle dataset (not tracked in git)
    .venv/                               # virtual environment (not tracked in git)
```

---

## Setup & Usage

### 1. Clone the repo
```
git clone https://github.com/tiffaning/pneumonia-xray-classifier.git
cd pneumonia-xray-classifier
```

### 2. Create a virtual environment and install dependencies
```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision pyecharts numpy pillow scikit-learn gradio
```

### 3. Add the dataset
Download the [Kaggle Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it in the project folder:
```
chest_xray/
    train/NORMAL/ and train/PNEUMONIA/
    val/NORMAL/   and val/PNEUMONIA/
    test/NORMAL/  and test/PNEUMONIA/
```

### 4. Run the notebook
Open `02_pneumonia_xray_classifier.ipynb` in VS Code and run all cells top to bottom. The final cell saves `model.pt`.

### 5. Launch the Gradio UI
```
python app.py
```
Open `http://127.0.0.1:7860` in your browser. Drag and drop any chest X-ray image to get a prediction.

---

## Key Concepts

**Transfer Learning** — ResNet18 was pretrained on 1.2M ImageNet photos. Its early layers already detect edges, textures, and shapes useful for any image task. Only the final classification layer was retrained for NORMAL vs. PNEUMONIA — making this fast and effective even on a small dataset.

**Why sensitivity over accuracy** — In medical screening, missing a sick patient (false negative) is far more dangerous than a false alarm. Sensitivity (97.4%) measures exactly this: of all real pneumonia cases, how many did the model catch?

---

## Stretch Goals
- Grad-CAM: visualize which region of the X-ray the model focused on
- WeightedRandomSampler: address the ~75% pneumonia class imbalance in training data
- Multi-class: add a COVID-19 class from the COVID-19 Radiography Database
- Model card: document failure modes before broader sharing

---

## Ethics Note
- This model is trained on pediatric X-rays and may not generalize to adult films
- Always report sensitivity/specificity, not just accuracy, for medical AI
- Synthetic data ≠ clinical data — results should not inform real diagnoses
