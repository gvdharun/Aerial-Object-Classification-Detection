# 🕊️🚁 Aerial Object Classification & Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-darkred?logo=streamlit)](https://streamlit.io)

---

## 🚀 Overview

Classifies aerial images as **Bird** or **Drone** using machine learning and a fine-tuned MobileNetV2 model. Includes a modern Streamlit web app for fast, easy, and visual prediction.

---

## 🛠️ Features

- 🧠 **Transfer Learning:** Uses [MobileNetV2](https://keras.io/api/applications/) for efficient and accurate results.
- 📊 **Model Evaluation:** View performance with confusion matrix, accuracy/loss plots, and classification report.
- ✨ **Streamlit UI:** Upload an image, hit **Predict**, and see instant results.
- 🐦🚁 **Practical Use-Case:** Security, wildlife monitoring, airspace safety.

---

## 📦 Installation

  `git clone https://github.com/yourusername/aerial-drone-bird-classifier.git`

  `cd aerial-drone-bird-classifier`

  `pip install -r requirements.txt`


---

## 💻 Usage
- In cmd
  
    `streamlit run Aerial_Object_Prediction_app.py`

- Model weights (`best_mobilenetv2.h5`) must be present inside the `/models` directory.

---

## 📝 Example

- Upload your aerial `.jpg` image  
- Click **Predict**  
- View label (**Bird** / **Drone**) & confidence

---

## 🏅 Results

- **Test Accuracy:** ~98.14%
- **Balanced Precision/Recall:** No bias between bird and drone classes

---

## 📁 Project Structure

```
├── Aerial_Object_Prediction_app.py # Streamlit app
├── models/ # Trained model weights
├── Aerial Object Prediction.ipynb / # jupyter notebook
├── dataset/ # Data 
├── requirements.txt
└── README.md
```

## ✅ Conclusion

- 🎯 **End-to-end solution:** This project delivers a complete aerial **Bird vs Drone** classification pipeline, from data preprocessing and model training to a user-friendly Streamlit web app.
  
- 🧠 **Strong models:** Transfer learning with MobileNetV2 (and other backbones) achieves robust accuracy with balanced precision/recall, making predictions reliable for real-world surveillance scenarios.
  
- 🌐 **Easy to use:** Anyone can upload an image through the web interface and instantly see the predicted class and confidence score, without needing ML or coding knowledge.
  
- 🚀 **Ready to extend:** The current setup is a solid base for future work such as YOLOv8 object detection, model comparison dashboards, and cloud deployment for real-time monitoring.
  
- 🤝 **Open for collaboration:** Contributions, ideas, and improvements are welcome—feel free to fork, experiment with new architectures, or integrate additional aerial datasets.

---
