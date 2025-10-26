🧠 Age & Gender Prediction

This project predicts a person’s age and gender from a face image using PyTorch and a ResNet-based multi-task neural network. The application is integrated with Streamlit for a simple web interface.

🛠 Features

Predicts age (regression) and gender (binary classification).

Handles images from file upload or camera capture.

Draws a bounding box around the detected face with predicted age and gender.

Supports GPU acceleration if available.

📁 Project Structure
age_gender_project/
│
├── dataset/                # UTKFace dataset
├── dataset_loader.py       # Custom PyTorch Dataset
├── model.py                # Multi-task ResNet model
├── train.py                # Training script
├── inference.py            # Prediction utilities
├── streamlit_app.py        # Streamlit web app
├── age_gender_model.pth    # Trained model (after training)
└── README.md

⚡ Requirements

Python >= 3.10

PyTorch >= 2.0

Torchvision

Streamlit

OpenCV

Pillow

NumPy

Install dependencies:

pip install torch torchvision streamlit opencv-python pillow numpy tqdm


Optional: GPU recommended for faster training and inference.

🚀 Usage
1. Training the Model
python train.py


Trains the model on the UTKFace dataset.

Adjustable hyperparameters: EPOCHS, BATCH_SIZE, IMG_SIZE, LR.

Saves the trained model as age_gender_model.pth.

2. Running the Web App
streamlit run streamlit_app.py


Open your browser.

Upload an image or use the camera.

See predicted age and gender instantly.

🧩 Notes

Image size during training: 128x128.

Age normalization: scaled to [0,1].

Gender threshold: 0.5 by default; can adjust for better accuracy.

Sometimes predictions may vary slightly; fine-tuning the model improves performance.

📈 Performance

Gender Accuracy: ~86%

Age MAE (Mean Absolute Error): ~9 years

Accuracy may vary depending on the dataset and preprocessing.

🔧 Troubleshooting

CUDA/GPU not detected: Check torch.cuda.is_available().

Face not detected: The app will resize the full image.

Slow predictions: Ensure IMG_SIZE matches training size and GPU is used.

📌 References

UTKFace Dataset: https://susanqq.github.io/UTKFace/

PyTorch: https://pytorch.org

Streamlit: https://streamlit.io