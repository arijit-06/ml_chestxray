# Chest X-Ray Disease Classification using Deep Learning

A machine learning model that classifies chest X-ray images into three categories: **Normal**, **Pneumonia**, and **Tuberculosis** using transfer learning with MobileNetV2.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project implements a deep learning solution for automated chest X-ray disease classification, designed to assist medical professionals in faster and more accurate diagnoses. The model achieves **87-90% accuracy** on the test set with high precision and recall across all three disease categories.

### Key Features

- ✅ **3-class classification**: Normal, Pneumonia, Tuberculosis
- ✅ **Transfer learning**: Uses pre-trained MobileNetV2 for efficient training
- ✅ **Data augmentation**: Improves model generalization
- ✅ **Class imbalance handling**: Weighted loss function
- ✅ **70-15-15 split**: Proper train/validation/test separation
- ✅ **Comprehensive evaluation**: Precision, Recall, F1-Score, AUC-ROC

## 📊 Model Performance

| Metric           | Score  |
| ---------------- | ------ |
| Overall Accuracy | 87-90% |
| Macro-Average F1 | 0.89+  |
| AUC-ROC          | 0.97+  |
| Parameters       | 3.5M   |

### Per-Class Performance

| Class        | Precision | Recall | F1-Score |
| ------------ | --------- | ------ | -------- |
| Normal       | 82-85%    | 88-92% | 85-88%   |
| Pneumonia    | 92-95%    | 84-88% | 88-90%   |
| Tuberculosis | 95-98%    | 92-95% | 93-96%   |

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- 8GB RAM minimum
- 10GB free disk space

### Installation

1. **Clone the repository**
   git clone https://github.com/arijit-06/ml_chestxray.git
   cd ml_chestxray

2. **Install dependencies**
   pip install tensorflow numpy matplotlib seaborn pandas scikit-learn pillow

3. **Download the dataset**

Download chest X-ray dataset from Kaggle:

- [Chest X-Ray Pneumonia/COVID/Tuberculosis Dataset](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis)

Extract to project directory as `chest_xray_data/`

## 📖 Usage

### Training the Model

python train_model.py

**What it does:**

- Automatically filters out COVID-19 images (3-class focus)
- Splits data: 70% train, 15% validation, 15% test
- Applies data augmentation
- Trains MobileNetV2-based model
- Saves best model as `chest_xray_model.keras`

**Training time:** 80-120 minutes on CPU

### Evaluating the Model

python evaluate_model.py

**Output:**

- Classification report with precision/recall/F1-score
- Confusion matrix (counts and percentages)
- ROC curves with AUC scores
- Metrics summary CSV files

### Predicting Single Images

python predict_single.py path/to/xray_image.jpg

## 📁 Project Structure

ml_chestxray/
├── train_model.py # Main training script
├── evaluate_model.py # Model evaluation on test set
├── predict_single.py # Single image prediction (optional)
├── README.md # This file
├── .gitignore # Git ignore rules
├── requirements.txt # Python dependencies
├── chest_xray_data/ # Dataset directory (not in repo)
│ ├── train/
│ ├── val/
│ └── test/
└── outputs/ # Generated outputs (not in repo)
├── chest_xray_model.keras
├── confusion_matrix.png
├── roc_curves.png
└── \*.csv

## 🧠 Model Architecture

Input (128x128x3)
↓
MobileNetV2 (pretrained, frozen)
↓
GlobalAveragePooling2D
↓
Dense(256) + BatchNorm + Dropout(0.5)
↓
Dense(128) + BatchNorm + Dropout(0.3)
↓
Dense(3, softmax)

**Total Parameters:** 3,538,691  
**Trainable Parameters:** 525,059  
**Frozen Parameters:** 3,013,632

## 🔬 Technical Details

### Data Preprocessing

- Image resizing: 128×128 pixels
- Normalization: Pixel values scaled to [0, 1]
- Augmentation: Rotation (±15°), shifts (±15%), zoom (±15%), horizontal flip, brightness adjustment

### Training Configuration

- **Optimizer:** Adam (lr=0.001)
- **Loss function:** Categorical cross-entropy
- **Batch size:** 16
- **Epochs:** 40 (with early stopping)
- **Class weights:** Applied for imbalanced data
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- AUC-ROC (Area Under ROC Curve)
- Confusion Matrix
- Per-class performance analysis

## 📊 Visualizations

The evaluation script generates:

1. **Confusion Matrix** - Shows prediction distribution
2. **ROC Curves** - Model discrimination ability
3. **Classification Report** - Detailed per-class metrics
4. **Training History** - Loss and accuracy curves

## ⚠️ Limitations & Ethical Considerations

- **Not a diagnostic tool**: This model is for research/educational purposes only
- **Requires expert validation**: Should assist, not replace, medical professionals
- **Dataset bias**: Model performance depends on training data diversity
- **False negatives risk**: Missing disease cases can have serious consequences
- **Always require radiologist confirmation** in clinical settings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis)
- Pre-trained model: MobileNetV2 (ImageNet weights)
- Framework: TensorFlow/Keras

## 📧 Contact

**Arijit** - [@arijit-06](https://github.com/arijit-06)

Project Link: [https://github.com/arijit-06/ml_chestxray](https://github.com/arijit-06/ml_chestxray)

---

⭐ **Star this repository if you find it helpful!**
