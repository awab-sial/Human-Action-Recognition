
# 🤖 Human Action Recognition System

![Python](https://img.shields.io/badge/Language-Python-blue?logo=python)
![Keras](https://img.shields.io/badge/Framework-Keras-orange?logo=keras)
![Tensorflow](https://img.shields.io/badge/Library-TensorFlow-yellow?logo=tensorflow)
![Semester Project](https://img.shields.io/badge/SE%20Semester%20Project-informational)

---

## 📌 Overview

**Human Action Recognition System** is an AI-powered deep learning project that identifies and classifies human activities from image and video data. Developed as a Software Engineering semester project at **Bahria University Islamabad Campus**, the system uses a convolutional neural network (CNN) architecture based on MobileNetV2, offering real-time action recognition through an intuitive PyQt-based interface.

---

## 🎯 Objectives

- Recognize 15 predefined human actions using computer vision and deep learning.
- Provide a **user-friendly GUI** for image/video input and real-time prediction.
- Ensure compatibility with PC and smartphone cameras.
- Support exporting results for further analysis.

---

## ⚙️ Features

- 📷 **Real-Time Detection** from live video streams
- 🖼️ **Image and Video Upload Support**
- 🧠 **Deep Learning with MobileNetV2**
- 📊 **Action Visualization** with performance stats
- 📁 **Result Exporting** (planned CSV/JSON support)
- 🖥️ **Clean GUI Interface** using PyQt5
- 🔄 **Integrated Data Augmentation Pipeline**

---

## 🧪 Supported Action Classes

1. Running
2. Sitting
3. Fighting
4. Dancing
5. Calling
6. Clapping
7. Cycling
8. Drinking
9. Eating
10. Hugging
11. Laughing
12. Listening to Music
13. Sleeping
14. Texting
15. Using Laptop

---

## 🗂️ Repository Structure

```

Human-Action-Recognition-System/
├── Activity Diagram.png              # Activity workflow visualization
├── Architecture Diagram.jpg          # System architecture overview
├── Class Diagram.png                 # Class relationships and structure
├── Front End.jpg                     # Frontend UI screenshot
├── Sequence Diagram.png              # Sequence of system interactions
├── human_action_model_improved.h5   # Trained AI model
├── Report.pdf                        # Final report document
├── README.md                         # Project documentation
├── train.py                          # Model training script
├── test.py                           # Model evaluation script
├── test_gui.py                       # GUI interaction testing script


````

---

## 📊 System Diagrams

### 🔁 Activity Diagram
<img src="Activity Diagram.png" alt="Activity Diagram" width="600"/>

### 📐 Architecture Diagram
<img src="Architecture Diagram.jpg" alt="Architecture Diagram" width="600"/>

### 🧩 Class Diagram
<img src="Class Diagram.png" alt="Class Diagram" width="600"/>

### 🧭 Sequence Diagram
<img src="Sequence Diagram.png" alt="Sequence Diagram" width="600"/>

### 🖥️ Front-End UI Design
<img src="Front End.jpg" alt="Front End UI" width="600"/>



---

## 🧠 Technologies Used

| Technology       | Purpose                          |
|------------------|----------------------------------|
| **Python**       | Backend development              |
| **Keras / TensorFlow** | Deep learning model architecture |
| **OpenCV**       | Image/video frame handling       |
| **NumPy**        | Data processing                  |
| **PyQt5**        | Desktop GUI development          |

---

## 🔧 Dataset Details

- **Name:** Human Action Recognition Dataset
- **Source:** Kaggle (by Shashank Rapolu)
- **Total Images:** 12,600 (Train: 10,710 | Test: 1,890)
- **Classes:** 15 human actions (listed above)
- **Formats:** `.jpg` images in class-wise directories
- **Augmentation:** Flip, zoom, brightness, rotation, etc.

---

## 📦 Installation & Usage

### 📥 Dependencies

Install via pip:

```bash
pip install -r requirements.txt
````

### 🚀 Running the System

```bash
python main.py
```

* Click `Upload` to load image/video
* Click `Detect` to identify the action
* View results in the GUI and visual preview

---

## 📢 Project Showcase

🚀 See live demo and project walkthrough on our LinkedIn:
[🔗 LinkedIn Showcase](https://shorturl.at/qhb75)

---

## 📜 License

This project is submitted for academic evaluation only under **Bahria University, Islamabad**'s **SE Semester Project** requirements.

---

## 🙌 Acknowledgments

**Instructor:** Ms. Sara Durrani

**Team Members:**

* Muhammad Awab Sial
* Syed Amber Ali Shah
* Fawad Naveed

