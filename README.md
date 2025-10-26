# 🧠 Age & Gender Prediction

A deep learning project to predict **age** and **gender** from facial images using PyTorch and ResNet-based multi-task learning. It also provides a **Streamlit app** for real-time predictions via upload or webcam.

---

## 🔹 Features

- Predict **age** (regression) and **gender** (binary classification) simultaneously.
- Uses **ResNet18** backbone with a shared fully connected layer.
- Streamlit interface for **live predictions**.
- Preprocessing and augmentation included for better performance.

---

## 📂 Repository Structure

```
├── dataset/                  # UTKFace dataset (not included, download separately)
├── dataset_loader.py         # PyTorch Dataset for UTKFace
├── model.py                  # MultiTaskResNet architecture
├── train.py                  # Training script
├── inference.py              # Model loading & prediction functions
├── streamlit_app.py          # Streamlit web app
├── evaluate.py               # Evaluation script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/Chaitanya1462k4/age_gender_prediction.git
cd age_gender_prediction
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training

Update `train.py` with your **dataset path**:

```python
DATA_DIR = "dataset"  # Path to UTKFace dataset
```

Run training:

```bash
python train.py
```

**Settings:**

- `IMG_SIZE = 128` (resize images to 128x128)  
- `BATCH_SIZE = 16`  
- `EPOCHS = 12`  
- `FREEZE_BACKBONE = True` (set False to fine-tune entire network)  

After training, the model weights will be saved as:

```
age_gender_model.pth
```

---

## 🧪 Evaluation

Use the evaluation script to check model performance:

```bash
python evaluate.py
```

- **Gender Accuracy** – Percentage of correct gender predictions  
- **Age MAE** – Mean Absolute Error for age predictions in years

---

## 🌐 Streamlit App

Run the interactive web app:

```bash
streamlit run streamlit_app.py
```

- Choose **Upload Image** or **Camera**  
- The app will display **predicted age, gender, and confidence**, along with a **detected face box**.  

---

## ⚠️ Notes

- The UTKFace dataset is **not included**. Download from [UTKFace Dataset](https://susanqq.github.io/UTKFace/)  
- Large files like datasets or model weights are not uploaded to GitHub.  
- Accuracy may vary for **low-quality images** or **extreme lighting conditions**.  

---

## 🔧 Requirements

- Python >= 3.10  
- PyTorch >= 2.0  
- Torchvision  
- OpenCV  
- Pillow  
- Streamlit  
- Numpy  
- tqdm  

Install all via:

```bash
pip install -r requirements.txt
```

---

## 📊 Results (Example)

| Metric           | Value        |
|-----------------|-------------|
| Gender Accuracy  | ~86%        |
| Age MAE         | ~9 years    |

---

## 📌 License

MIT License

