# 🛰️ Klasifikasi Landscape Aerial dengan Deep Learning

> Sistem klasifikasi otomatis untuk mengenali 15 jenis landscape dari foto udara menggunakan CNN dan Transfer Learning

## 🌍 Overview Dataset

**Skyview Multi-Landscape Aerial Imagery Dataset** - Koleksi 12,000 gambar aerial berkualitas tinggi untuk riset computer vision.

### 📊 Spesifikasi Dataset

- **Total Kategori**: 15 jenis landscape
- **Gambar per Kategori**: 800 (seimbang sempurna)
- **Resolusi**: 256×256 pixels → diresize ke 128×128
- **Total Gambar**: 12,000
- **Sumber**: [Kaggle - Skyview Dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)

### 🏞️ Kategori Landscape

| Agriculture |  Airport  |  Beach  |    City     |  Desert  |
| :---------: | :-------: | :-----: | :---------: | :------: |
|   Forest    | Grassland | Highway |    Lake     | Mountain |
|   Parking   |   Port    | Railway | Residential |  River   |

## 🧠 Arsitektur Model

### 🔧 Desain Inti

```
MobileNetV2 (Base Model - Frozen)
    ↓
Conv2D (128 filters) + BatchNorm + MaxPool
    ↓
Conv2D (64 filters) + BatchNorm
    ↓
GlobalAveragePooling2D + Dropout
    ↓
Dense (128) + Dropout
    ↓
Dense (15) - Softmax Output
```

### 🎯 Strategi Training

- **Fase 1**: Transfer Learning (30 epochs) - Freeze base model
- **Fase 2**: Fine-tuning (15 epochs) - Unfreeze 20 layer teratas
- **Pembagian Data**: 70% Train | 15% Val | 15% Test

## 📈 Hasil Performa

### 🏆 Performa Model

- **Test Accuracy**: 90.89%
- **Test Precision**: 91.24%
- **Test Recall**: 90.89%
- **Test F1-Score**: 90.91%

### 🥇 Kelas Terbaik

- **Forest**: 99.3% akurasi
- **Parking**: 99.3% akurasi
- **Port**: 99.3% akurasi
- **Agriculture**: 99.2% akurasi

### 📉 Kelas Menantang

- **Airport**: 97.9% akurasi
- **River**: 97.9% akurasi
- **Railway**: 98.3% akurasi

## 🚀 Fitur Teknis

### 🔥 Training Lanjutan

- **Data Augmentation**: Rotasi, shift, zoom, flip, brightness (training saja)
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Optimizer**: Adam dengan adaptive learning rate
- **Regularisasi**: Dropout + BatchNormalization

### 📱 Format Deployment

| Format     | Kegunaan    | File                       |
| ---------- | ----------- | -------------------------- |
| Keras      | Development | `finish_model.keras`       |
| SavedModel | Production  | `finish_model_savedmodel/` |
| TFLite     | Mobile/Edge | `finish_model.tflite`      |

## 🛠️ Cara Penggunaan

### 1. Setup Environment

```bash
TensorFlow version: 2.18.0
NumPy version: 2.0.2
Pandas version: 2.2.2
KaggleHub version: 0.3.12
Matplotlib version: 3.10.0
Seaborn version: 0.13.2
PIL (Pillow) version: 11.2.1
```

### 2. Load Model

```python
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('finish_model.keras')

# Predict single image
import numpy as np
from PIL import Image

# Load dan preprocessing gambar
img = Image.open('aerial_image.jpg').resize((128, 128))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediksi
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

class_names = ['Agriculture', 'Airport', 'Beach', 'City', 'Desert',
               'Forest', 'Grassland', 'Highway', 'Lake', 'Mountain',
               'Parking', 'Port', 'Railway', 'Residential', 'River']

print(f"Prediksi: {class_names[predicted_class]}")
print(f"Confidence: {predictions[0][predicted_class]:.2%}")
```

### 3. Training Custom Model

```python
# Download dataset
dataset_path = kagglehub.dataset_download("ankit1743/skyview-an-aerial-landscape-dataset")

# Setup data generators dengan augmentation
train_gen, val_gen, test_gen = setup_data_generators(dataset_path)

# Buat model
model = create_optimized_cnn_model()

# Training
history = model.fit(train_gen, validation_data=val_gen, epochs=30)
```

## 📊 Struktur Project

```
submission
├── finish_model.keras
├───tfjs_model(finish_model_tfjs/)
| ├───group1-shard1of1.bin
| └───model.json
├───tflite(finish_model_tflite/)
| ├───model.tflite
| └───label.txt
├───saved_model(finish_model_savedmodel/)
| ├───saved_model.pb
| └───variables
├───notebook.ipynb
├───README.md
└───requirements.txt
```

## 🎯 Hasil Visualisasi

Model menghasilkan beberapa visualisasi penting:

- **Training History**: Kurva accuracy dan loss
- **Confusion Matrix**: Analisis kesalahan prediksi
- **Top Predictions**: Contoh prediksi terbaik per kelas

## 🚀 Deployment

Model siap untuk deployment dalam berbagai format:

- **Web Application**: Gunakan TensorFlow.js
- **Mobile App**: Gunakan TensorFlow Lite
- **Server API**: Gunakan SavedModel format
- **Research**: Gunakan Keras format

## 📝 Credits

- **Dataset**: [Skyview Aerial Dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)
- **Original Sources**: AID Dataset & NWPU-Resisc45 Dataset
- **Architecture**: MobileNetV2 + Custom Layers
