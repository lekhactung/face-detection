#  Face Detection

A real-time face detection system built with **TensorFlow**, **VGG16** (transfer learning), and **OpenCV**. The model detects and localizes faces in live webcam footage using a custom dual-head neural network that simultaneously performs **classification** (face / no face) and **bounding box regression**.

---

##  Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Setup & Usage](#setup--usage)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)

---

## Overview

This project implements an end-to-end face detection pipeline:

1. **Collect** images via webcam using OpenCV
2. **Annotate** bounding boxes using LabelMe
3. **Augment** data with Albumentations (60× per image)
4. **Train** a custom dual-head VGG16 model using TensorFlow
5. **Detect** faces in real-time via live webcam feed

---

## Project Structure

```
FaceDetection/
│
├── FaceDetection.ipynb     # Main notebook – full pipeline
├── requirements.txt        # Python dependencies
│
├── data/
│   ├── images/             # Raw collected images
│   ├── label/              # LabelMe JSON annotations
│   ├── train/
│   │   ├── images/
│   │   └── label/
│   ├── test/
│   │   ├── images/
│   │   └── label/
│   └── val/
│       ├── images/
│       └── label/
│
├── aug_data/               # Augmented dataset (auto-generated)
│   ├── train/
│   ├── test/
│   └── val/
│
├── models/
│   └── facetracker.h5      # Saved trained model
│
└── logs/                   # TensorBoard training logs
```

---

## Pipeline

```
Webcam Capture → LabelMe Annotation → Train/Test/Val Split
       ↓
Albumentations Augmentation (60× per image)
       ↓
TF Data Pipeline (load, resize 120×120, normalize)
       ↓
VGG16 Backbone (pretrained, frozen)
       ↓
┌──────────────────────┐
│  Classification Head │ → sigmoid → face / no face
│  Regression Head     │ → sigmoid → [x1, y1, x2, y2]
└──────────────────────┘
       ↓
Real-time Webcam Detection
```

---

## Model Architecture

The model uses **VGG16** as a feature extractor (without top layers) with two separate output heads:

| Head | Layers | Output |
|------|--------|--------|
| **Classification** | GlobalMaxPool → Dense(2048, ReLU) → Dense(1, Sigmoid) | 0 or 1 (face probability) |
| **Regression** | GlobalMaxPool → Dense(2048, ReLU) → Dense(4, Sigmoid) | [x1, y1, x2, y2] normalized coords |

**Loss Functions:**
- Classification: `BinaryCrossentropy`
- Localization: Custom loss = `Δcoord² + Δsize²`
- Total: `localization_loss + 0.5 × classification_loss`

**Optimizer:** Adam (lr=0.0001, with learning rate decay)

---

## Requirements

```txt
tensorflow
opencv-python
matplotlib
albumentations
labelme
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** GPU is recommended for training. The notebook auto-configures GPU memory growth to avoid OOM errors.

---

## Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/lekhactung/face-detection.git
cd FaceDetection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `FaceDetection.ipynb` in Jupyter or VS Code and run cells step by step.

---

## Dataset

### Step 1 – Collect Images

The notebook uses OpenCV to capture **30 images** from your webcam automatically:

```python
cap = cv2.VideoCapture(0)
# Captures frames every 0.5s, saves as UUID-named .jpg files
```

Images are saved to: `data/images/`

### Step 2 – Annotate with LabelMe

Run LabelMe from the notebook:
```python
!labelme
```

- Open the images directory in LabelMe
- Draw bounding boxes around faces
- Save annotations as `.json` files in `data/label/`

### Step 3 – Split Dataset

Manually split images into three folders:

```
data/
├── train/images/   # ~70% of images
├── test/images/    # ~15% of images
└── val/images/     # ~15% of images
```

The notebook will automatically move the corresponding label files.

### Step 4 – Data Augmentation

Each image is augmented **60 times** with:

| Transform | Probability |
|-----------|-------------|
| Horizontal Flip | 50% |
| Random Brightness/Contrast | 20% |
| Random Gamma | 20% |
| RGB Shift | 20% |

Output saved to: `aug_data/`

---

## Training

Train the model for **10 epochs** with TensorBoard logging:

```python
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs/
```

Training tracks three metrics:
- `total_loss`
- `class_loss`
- `regress_loss`

---

## Results

After training, the model is saved to `models/facetracker.h5` and used for **real-time detection**:

```python
facetracker = load_model("models/facetracker.h5")

# Real-time detection loop
while cap.isOpened():
    ret, frame = cap.read()
    yhat = facetracker.predict(frame_preprocessed)
    if yhat[0][0] > 0.5:   # confidence threshold
        # Draw bounding box + "face" label
```

The detection draws a blue bounding box with a **"face"** label when confidence > 0.5.

---

## Notes

- Images are resized to **120×120** for model input
- Bounding box coordinates are **normalized** to [0, 1] range
- The model was trained/tested on **Windows** with local webcam input
- LabelMe annotation format uses `[x_min, y_min]` and `[x_max, y_max]` point pairs

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Model building & training |
| VGG16 | Pretrained CNN backbone |
| OpenCV | Image capture & real-time display |
| LabelMe | Bounding box annotation |
| Albumentations | Data augmentation |
| Matplotlib | Visualization |
| TensorBoard | Training monitoring |