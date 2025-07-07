# 🧠 CNN Project: Image Classification (Optional Object Detection)

This project implements a **modular, configurable Convolutional Neural Network (CNN)** from scratch using PyTorch. It supports:
- 🖼️ Image classification using CIFAR-10 or custom datasets
- 📦 Optional extension to object detection
- 🔧 Configurable CNN blocks, batch norm, dropout, learning rate, etc.

---

## 📁 Folder Structure

# CNN_image_classification

cnn_project/
├── config/ # YAML configs
├── data/ # Dataset loader scripts
├── models/ # CNN blocks, classifier and detector
├── train/ # Training scripts
├── inference/ # Prediction/inference code
├── utils/ # Logging, metrics
├── main.py # Entrypoint CLI
├── requirements.txt # Python dependencies
└── README.md # You're here

python main.py --config config/default.yaml
