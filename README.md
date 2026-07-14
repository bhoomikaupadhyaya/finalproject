# 🌿 Leaf Vein Detection using Image Processing & Machine Learning

An AI-powered computer vision project for the **early detection of nutrient deficiencies in Rosa-Sinensis (Hibiscus)** by analyzing **leaf vein architecture, color, texture, and shape**.

Unlike traditional plant disease detection systems that rely only on visible symptoms, this project focuses on **early-stage nutrient deficiency detection** through advanced image preprocessing, vein extraction, feature engineering, and machine learning.

---

## 📌 Features

- 📷 Leaf image preprocessing
- 🔍 Image quality assessment
- 📏 Automatic image resizing
- 🧹 Noise removal using Bilateral Filter
- 🎨 Color space conversion (RGB → LAB)
- ✨ Contrast enhancement using CLAHE
- 🍃 Leaf segmentation
- 🧬 Morphological processing
- 🌱 Leaf vein enhancement
- 🕸 Skeletonization
- 📊 Feature extraction
  - Color Features
  - Texture Features
  - Shape Features
  - Vein Features
- 🤖 Machine Learning Classification
- 📈 Nutrient Deficiency Prediction

---

# 🏗 Project Pipeline

```
Raw Leaf Image
        │
        ▼
Image Quality Check
        │
        ▼
Resize
        │
        ▼
Noise Removal
        │
        ▼
Color Space Conversion
        │
        ▼
CLAHE Enhancement
        │
        ▼
Leaf Segmentation
        │
        ▼
Morphological Operations
        │
        ▼
Leaf Vein Enhancement
        │
        ▼
Skeletonization
        │
        ▼
Feature Extraction
        │
        ▼
Machine Learning
        │
        ▼
Prediction
```

---

# 📂 Project Structure

```
Leaf-Vein-Detection/
│
├── dataset/
│   ├── healthy/
│   ├── iron/
│   ├── nitrogen/
│   └── phosphorus/
│
├── output/
│   ├── resized/
│   ├── denoised/
│   ├── color_space/
│   ├── enhanced/
│   ├── segmented/
│   ├── morphology/
│   ├── veins/
│   └── skeleton/
│
├── preprocessing/
│   ├── image_quality.py
│   ├── resize.py
│   ├── denoise.py
│   ├── color_space.py
│   ├── clahe.py
│   ├── segmentation.py
│   ├── morphology.py
│   └── preprocess.py
│
├── vein/
│   ├── vein_enhancement.py
│   ├── skeleton.py
│   └── vein_pipeline.py
│
├── features/
│   ├── color_features.py
│   ├── shape_features.py
│   ├── texture_features.py
│   ├── vein_features.py
│   ├── feature_pipeline.py
│   └── features.csv
│
├── models/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── model.pkl
│
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-image
- Scikit-learn
- Matplotlib
- Pillow
- Joblib

---

# 🧠 Image Processing Techniques

- Image Quality Assessment
- Bilateral Filtering
- LAB Color Space Conversion
- CLAHE
- HSV-Based Segmentation
- Morphological Operations
- Adaptive Thresholding
- Top-Hat Filtering
- Skeletonization

---

# 📊 Extracted Features

## 🍃 Color Features

- Mean Red
- Mean Green
- Mean Blue
- Mean Hue
- Mean Saturation
- Mean Value
- Mean L
- Mean A
- Mean B

---

## 🌿 Shape Features

- Leaf Area
- Perimeter
- Aspect Ratio
- Extent
- Solidity
- Circularity

---

## 🧬 Texture Features

- GLCM Contrast
- Dissimilarity
- Homogeneity
- Energy
- Correlation
- ASM
- Entropy
- Local Binary Pattern (LBP)

---

## 🌱 Vein Features

- Vein Density
- Vein Pixels
- Vein Length
- Branch Count

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Leaf-Vein-Detection.git
```

Go inside the project

```bash
cd Leaf-Vein-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Image Preprocessing

```bash
cd preprocessing

python preprocess.py
```

---

# ▶️ Run Vein Extraction

```bash
cd vein

python vein_pipeline.py
```

---

# ▶️ Run Feature Extraction

```bash
cd features

python feature_pipeline.py
```

---

# 🤖 Machine Learning (Coming Soon)

- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- CNN (Future Enhancement)

---

# 📈 Future Enhancements

- Deep Learning (CNN)
- Mobile Application
- Flask Web Application
- Real-Time Leaf Detection
- Multi-Plant Support
- Fertilizer Recommendation System
- Explainable AI (XAI)
- Cloud Deployment

---

# 🎯 Project Goal

To develop a simple, low-cost, and non-invasive AI system capable of detecting nutrient deficiencies in **Rosa-Sinensis** by analyzing leaf vein architecture and visual characteristics, enabling early diagnosis and timely intervention.

---

# 📄 License

This project is developed for academic and research purposes.

---
