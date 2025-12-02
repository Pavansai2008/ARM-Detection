# 🩺 Pediatric Ultrasound ARM Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-0.84+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**AI-Powered Medical Imaging Tool for Early Detection of Anorectal Malformations**

[🚀 Live Demo](#) | [📖 Documentation](#features) | [💻 GitHub](https://github.com/Pavansai2008/ARM) | [📧 Contact](mailto:pavansaibudur2008@gmail.com)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Screenshots](#screenshots)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Contact](#contact)

---

## 🎯 Overview

This project is an **AI-powered medical imaging application** designed to assist clinicians and researchers in screening pediatric ultrasound images for **Anorectal Malformations (ARMs)**. The system uses deep learning models to classify ultrasound images and provides visual explanations through Grad-CAM heatmaps, making it a valuable tool for early diagnosis and medical research.

### Key Highlights

- ✅ **Two-Stage Classification**: First detects if the image is an ultrasound, then classifies ARM vs Normal
- ✅ **Explainable AI**: Grad-CAM visualizations show which regions influenced the prediction
- ✅ **Professional UI**: Clean, medical-themed interface built with Streamlit
- ✅ **PDF Reports**: Generate downloadable diagnostic reports
- ✅ **Production-Ready**: Optimized for deployment and real-world use

---

## ✨ Features

### 🔬 Core Functionality

1. **Ultrasound Detection**
   - Automatically identifies if uploaded image is an ultrasound scan
   - High accuracy classification with confidence scores

2. **ARM Classification**
   - Binary classification: ARM vs Normal
   - Probability scores for both classes
   - Confidence thresholding for uncertain cases

3. **Visual Explanations**
   - Grad-CAM heatmaps highlight important regions
   - Side-by-side comparison of original and heatmap
   - Helps clinicians understand model decisions

4. **Report Generation**
   - Downloadable PDF reports with:
     - Original image
     - Grad-CAM visualization
     - Prediction results and probabilities
     - Medical disclaimers

### 🎨 User Interface

- **Professional Medical Theme**: Clean, modern design optimized for medical professionals
- **Responsive Layout**: Works on desktop and tablet devices
- **Real-time Feedback**: Loading indicators and progress updates
- **Educational Content**: ARM insights and medical information sidebar

---

## 🛠️ Tech Stack

### Core Technologies

- **Python 3.8+**: Primary programming language
- **PyTorch 1.9+**: Deep learning framework
- **Streamlit**: Web application framework
- **OpenCV**: Image processing
- **Pillow**: Image manipulation

### Key Libraries

- `torch` & `torchvision`: Deep learning models (ResNet18)
- `streamlit`: Web UI framework
- `pytorch-grad-cam`: Model interpretability
- `fpdf`: PDF report generation
- `opencv-python`: Computer vision operations
- `numpy`, `pandas`: Data processing

### Model Architecture

- **Stage 1**: ResNet18-based Ultrasound Classifier
- **Stage 2**: ResNet18-based ARM Classifier
- **Transfer Learning**: Pre-trained on ImageNet
- **Data Augmentation**: Rotation, flipping, color jitter

---

## 📁 Project Structure

```
Arm_detection-main/
│
├── Arm-Detectiom/
│   ├── frontend/
│   │   ├── app.py                 # Main Streamlit application
│   │   └── test_frontend.py       # Frontend tests
│   │
│   └── backend/
│       ├── inference.py           # Model inference functions
│       ├── requirements.txt       # Backend dependencies
│       ├── test_backend.py        # Backend tests
│       │
│       └── model/
│           ├── train/
│           │   ├── stage1.py      # Ultrasound classifier training
│           │   └── stage2.py      # ARM classifier training
│           │
│           └── output/            # Trained model weights
│               ├── best_ultrasound_classifier.pth
│               └── best_arm_classifier.pth
│
├── README.md                      # This file
└── requirements.txt               # Root dependencies (for deployment)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Pavansai2008/ARM-Detection-.git
cd ARM-Detection

```

### Step 2: Download Model Weights

Download the pre-trained model weights from Google Drive:

**[📥 Download Models](https://drive.google.com/drive/folders/1O9V8wVQQ20nWwya30ONJlJsZqIq9GMZQ?usp=sharing)**

Place the model files in: `backend/output/`

### Step 3: Install Dependencies

```bash
# Install backend dependencies
cd Arm-Detectiom/backend
pip install -r requirements.txt

# Or install from root
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
# Navigate to frontend directory
cd Arm-Detectiom/frontend

# Run Streamlit app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 💻 Usage

### For End Users

1. **Upload Image**: Click "Upload Ultrasound Image" and select a `.jpg`, `.jpeg`, or `.png` file
2. **Analyze**: Click the "Diagnose" button
3. **View Results**: 
   - See classification results with confidence scores
   - View Grad-CAM heatmap visualization
   - Download PDF report if needed

### For Developers

#### Training Models

```bash
# Train Stage 1: Ultrasound Classifier
cd backend/model/train
python stage1.py

# Train Stage 2: ARM Classifier
python stage2.py
```

#### Running Tests

```bash
# Test backend
cd backend
python test_backend.py

# Test frontend
cd frontend
python test_frontend.py
```

---

## 🧠 Model Architecture

### Two-Stage Pipeline

1. **Stage 1 - Ultrasound Detection**
   - Input: Medical image
   - Output: Ultrasound / Non-Ultrasound
   - Model: ResNet18 (ImageNet pre-trained)
   - Classes: 2 (ultrasound, non_ultrasound)

2. **Stage 2 - ARM Classification**
   - Input: Confirmed ultrasound image
   - Output: ARM / Normal
   - Model: ResNet18 (ImageNet pre-trained)
   - Classes: 2 (ARM, Normal)

### Training Details

- **Optimizer**: Adam (lr=0.0001, weight_decay=1e-4)
- **Loss Function**: CrossEntropyLoss
- **Data Augmentation**: Random rotation, flipping, color jitter
- **Early Stopping**: Patience=3 epochs
- **Dropout**: 0.5 for regularization

---

## 📸 Screenshots

### Main Interface
![Main Interface](https://via.placeholder.com/800x400?text=Main+Interface)

### Results Display
![Results](https://via.placeholder.com/800x400?text=Results+with+Grad-CAM)

### PDF Report
![PDF Report](https://via.placeholder.com/800x400?text=PDF+Report)

> **Note**: Add your actual screenshots here for a complete portfolio showcase

---

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. **Push to GitHub** – include `streamlit_app.py`, `.streamlit/secrets_template.toml`, and (optionally) the `backend/output/*.pth` weights.
2. **Create the app** at [share.streamlit.io](https://share.streamlit.io):
   - Repository: this project
   - Branch: `main` (or your deploy branch)
   - Main file: `streamlit_app.py`
3. **Provide model URLs**  
   - In Streamlit Cloud, open *App → Settings → Secrets*.
   - Paste the contents of `.streamlit/secrets_template.toml` and replace the placeholder URLs with real links to your `.pth` files.
   - The backend auto-downloads each model if it is missing locally.
4. **Deploy** – Streamlit installs from `requirements.txt` and runs the UI automatically.

> **Tip:** If you prefer to bundle the `.pth` files directly in the repo (they are ≈43 MB each), place them under `backend/output/`. In that case you can leave the secret URLs empty.

### Other Platforms

- **Heroku**: Use Procfile and requirements.txt
- **AWS EC2**: Deploy with Docker
- **Google Cloud Run**: Container-based deployment

---

## 📊 Performance Metrics

- **Ultrasound Detection Accuracy**: ~95% (on test set)
- **ARM Classification Accuracy**: ~92% (on test set)
- **Inference Time**: < 2 seconds per image
- **Model Size**: ~45 MB per model

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

This tool is designed for **educational and screening purposes only**. It is **NOT a substitute for professional medical diagnosis**. Always consult qualified healthcare professionals for medical decisions.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Contact & Links

### Developer Information

- **Name**: Nitish Kumar
- **Email**: [knitishk102@gmail.com](mailto:knitishk102@gmail.com)
- **GitHub**: [@Nitish-kumar7](https://github.com/Nitish-kumar7)
- **Project Repository**: [ARM Detection](https://github.com/Nitish-kumar7/ARM)

### Project Links

- 🔗 **Live Demo**: [Add your Streamlit Cloud link here]
- 📦 **Model Download**: [Google Drive](https://drive.google.com/drive/folders/1O9V8wVQQ20nWwya30ONJlJsZqIq9GMZQ?usp=sharing)
- 💻 **Source Code**: [GitHub Repository](https://github.com/Nitish-kumar7/ARM)
- 📧 **Contact**: [knitishk102@gmail.com](mailto:knitishk102@gmail.com)

---

## 🙏 Acknowledgments

- Medical imaging community for datasets and research
- PyTorch team for the excellent deep learning framework
- Streamlit for the intuitive web framework
- Grad-CAM authors for model interpretability tools

---

<div align="center">

**Made with ❤️ for Pediatric Healthcare**

⭐ Star this repo if you find it helpful!

</div>
#   A R M - D e t e c t i o n  
 #   A R M - D e t e c t i o n  
 