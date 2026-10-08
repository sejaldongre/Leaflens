# LeafLens 🌿

## Medicinal Plant Classification & Identification Using Deep Learning

LeafLens is an end-to-end computer vision project for identifying medicinal plants from leaf images. It compares a classical machine-learning baseline with a CNN-based deep-learning approach and deploys the final model as an interactive Streamlit application.

The final model uses **MobileNetV3Small transfer learning and fine-tuning** and achieved **90.25% accuracy on an untouched test set** containing 595 images across 40 medicinal plant classes.

## 🌐 Project Links

- **Live Demo:** https://sejaldongre-leaflens.streamlit.app/
- **GitHub:** https://github.com/sejaldongre/Leaflens
- **Dataset:** https://huggingface.co/datasets/Project-AgML/DIMPSAR_medicinal_plant_classification

## 📌 Overview

Medicinal plants can have visually similar leaves, making image-based identification a challenging computer-vision problem.

LeafLens addresses this task using a dataset of **5,945 images from 40 medicinal plant classes**.

The project follows two main approaches:

1. **Classical Computer Vision + Machine Learning**
2. **CNN-based Deep Learning with Transfer Learning**

The deep-learning approach significantly outperformed the engineered-feature baseline and became the final deployed model.

## 📊 Dataset

LeafLens uses the **DIMPSAR Medicinal Plant Classification Dataset** from Hugging Face.

| Property          |           Value |
| ----------------- | --------------: |
| Total images      |           5,945 |
| Classes           |              40 |
| Training images   |           4,756 |
| Validation images |             594 |
| Test images       |             595 |
| Split             | 80% / 10% / 10% |

A stratified split was used to maintain class distribution across the training, validation, and test sets. The final evaluation was performed on the **untouched test set**.

### Plant Classes

```text
Aloevera
Amla
Amruta_Balli
Arali
Ashoka
Ashwagandha
Avacado
Bamboo
Basale
Betel
Betel_Nut
Brahmi
Castor
Curry_Leaf
Doddapatre
Ekka
Ganike
Gauva
Geranium
Henna
Hibiscus
Honge
Insulin
Jasmine
Lemon
Lemon_grass
Mango
Mint
Nagadali
Neem
Nithyapushpa
Nooni
Pappaya
Pepper
Pomegranate
Raktachandini
Rose
Sapota
Tulasi
Wood_sorel
```

# 🧠 Methodology

## 1. Classical Machine Learning Baseline

A traditional computer-vision pipeline was developed as a baseline.

### Image Preprocessing

- Image resizing
- HSV / Excess Green based masking
- GrabCut segmentation
- Morphological operations
- Largest connected-component extraction

### Feature Engineering

The system extracted **55 engineered features** using:

- Local Binary Patterns (LBP)
- Gray-Level Co-occurrence Matrix (GLCM)
- Gabor features
- Color moments
- Gradient-based texture features

### Model Pipeline

```text
Leaf Image
    ↓
Preprocessing & Segmentation
    ↓
Feature Extraction
    ↓
PCA
    ↓
Random Forest
    ↓
Plant Classification
```

### Baseline Results

| Metric      |  Score |
| ----------- | -----: |
| Accuracy    | 66.55% |
| Macro F1    | 0.6604 |
| Weighted F1 | 0.6613 |

## 2. Deep Learning Approach

The final system uses **MobileNetV3Small** with transfer learning.

### Architecture

```text
Input Leaf Image
       ↓
RGB Conversion
       ↓
224 × 224 Resize
       ↓
MobileNetV3Small
       ↓
Classification Head
       ↓
40 Plant Classes
```

MobileNetV3Small provides a useful balance between classification performance and computational efficiency for lightweight deployment.

### Transfer Learning

The model was trained in stages.

**Stage 1 — Frozen Backbone**

The MobileNetV3Small backbone was initially frozen while the classification layers were trained.

- Test accuracy: **87.73%**

**Stage 2 — Fine-Tuning**

The best frozen model was fine-tuned by unfreezing the final 30 layers.

Training configuration:

- MobileNetV3Small pretrained on ImageNet
- Final 30 layers unfrozen
- Batch Normalization layers kept frozen
- Adam optimizer
- Learning rate: `1e-5`
- Input resolution: `224 × 224`

# 📈 Final Model Performance

The final fine-tuned model was evaluated on the untouched test set.

| Metric                |        Result |
| --------------------- | ------------: |
| **Test Accuracy**     |    **90.25%** |
| **Macro F1**          |    **0.8993** |
| **Weighted F1**       |    **0.9005** |
| Correct predictions   | **537 / 595** |
| Incorrect predictions |      58 / 595 |

### Final Model

```text
MobileNetV3Small
Transfer Learning + Fine-Tuning
```

Model file:

```text
data/processed/cnn/mobilenetv3small_finetuned_best.keras
```

## 🖼️ Inference Preprocessing

The deployed application uses the same core preprocessing expected by the trained CNN:

```python
image = image.convert("RGB")
image = image.resize((224, 224))

image_array = np.array(image, dtype=np.float32)
image_array = (image_array / 127.5) - 1.0

image_array = np.expand_dims(image_array, axis=0)
```

The model produces probabilities for all 40 plant classes and selects the class with the highest predicted probability.

# 🌐 Streamlit Application

LeafLens is deployed using **Streamlit Community Cloud**.

The application allows users to:

1. Upload a leaf image
2. Process the image automatically
3. Run the trained MobileNetV3Small model
4. Predict the plant class
5. Display plant information

### Deployment Flow

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
LeafLens Web Application
```

**Live Application:** https://sejaldongre-leaflens.streamlit.app/

# 🛠️ Technology Stack

### Programming

- Python

### Deep Learning

- TensorFlow
- Keras
- MobileNetV3Small
- Transfer Learning
- CNN

### Machine Learning

- Scikit-learn
- Random Forest
- PCA

### Computer Vision

- OpenCV
- scikit-image
- Pillow

### Data Processing

- NumPy
- Pandas

### Deployment

- Streamlit
- Streamlit Community Cloud

### Development

- Visual Studio Code
- Jupyter Notebook
- Google Colab
- Git
- GitHub

# 📁 Project Structure

```text
Leaflens/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── cnn/
│           └── mobilenetv3small_finetuned_best.keras
│
├── src/
│   ├── app.py
│   ├── models/
│   └── ...
│
├── requirements.txt
├── README.md
└── ...
```

The exact contents may vary depending on the local development and experiment files.

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sejaldongre/Leaflens.git
cd Leaflens
```

Create a virtual environment:

```bash
python -m venv cnn_env
```

### Windows

```powershell
cnn_env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

# ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run src/app.py
```

Streamlit will provide a local URL in the terminal.

# 🔬 Robustness Experiment

A separate robustness experiment was conducted using:

- Random horizontal flipping
- Random rotation
- Random translation
- Random zoom
- Random contrast
- Random brightness

The robustness model was evaluated separately from the final original model. It did **not** improve the untouched test-set performance or external-image experiments, so the original fine-tuned model remains the final portfolio/deployment model.

# ⚠️ Limitations

The reported **90.25% accuracy** represents performance on the project's untouched test set.

Real-world images may differ from the training data because of:

- Lighting conditions
- Backgrounds
- Camera quality
- Leaf orientation
- Image composition
- Leaf growth stage
- Domain differences between dataset images and user photographs

These differences can cause incorrect predictions even when test-set performance is high.

Therefore, LeafLens should be considered an **image-classification prototype and portfolio project**, rather than a definitive botanical identification system.

# 🎯 Key Learning Outcomes

This project provided hands-on experience with:

- End-to-end image classification
- Dataset preparation
- Stratified train/validation/test splitting
- Computer-vision preprocessing
- Image segmentation
- Feature engineering
- Classical machine learning
- CNN image classification
- Transfer learning
- Model fine-tuning
- Model evaluation
- F1-score based evaluation
- Domain-shift analysis
- Streamlit deployment
- Git and GitHub
- Cloud deployment

# 📌 Results Summary

```text
Dataset
5,945 images
40 plant classes
        ↓
Classical ML Baseline
Random Forest + PCA
        ↓
66.55% Accuracy
        ↓
Deep Learning
MobileNetV3Small Transfer Learning
        ↓
87.73% Accuracy
        ↓
Fine-Tuning
        ↓
90.25% Test Accuracy
```

# 👩‍💻 Author

**Sejal Dongre**

AI & Data Science Engineer

Interested in:

- Artificial Intelligence
- Machine Learning
- Data Science
- Computer Vision
- Generative AI
- LLMs
- Applied AI

## ⭐ Links

- **Live Demo:** https://sejaldongre-leaflens.streamlit.app/
- **GitHub Repository:** https://github.com/sejaldongre/Leaflens
- **Dataset:** https://huggingface.co/datasets/Project-AgML/DIMPSAR_medicinal_plant_classification
