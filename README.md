# 🛰️ Klasifikasi Landscape Aerial dengan Deep Learning

<div align="center">

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-Transfer%20Learning-green?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-90.89%25-success?style=for-the-badge)

**Proyek Akhir Kelas Belajar Machine Learning untuk Pemula — Dicoding**

*oleh Silmi Azdkiatul Athqia (silmiathqia)*

</div>

> Sistem klasifikasi otomatis untuk mengenali 15 jenis landscape dari foto udara menggunakan CNN dan Transfer Learning

---

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

---

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

---

## 📈 Hasil Performa

### 🏆 Performa Model

| Metrik | Nilai |
|---|---|
| Test Accuracy | **90.89%** |
| Test Precision | **91.24%** |
| Test Recall | **90.89%** |
| Test F1-Score | **90.91%** |

### 🥇 Kelas Terbaik

- **Forest**: 99.3% akurasi
- **Parking**: 99.3% akurasi
- **Port**: 99.3% akurasi
- **Agriculture**: 99.2% akurasi

### 📉 Kelas Menantang

- **Airport**: 97.9% akurasi
- **River**: 97.9% akurasi
- **Railway**: 98.3% akurasi

---

## 🚀 Fitur Teknis

### 🔥 Training Lanjutan

- **Data Augmentation**: Rotasi, shift, zoom, flip, brightness (training saja)
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Optimizer**: Adam dengan adaptive learning rate
- **Regularisasi**: Dropout + BatchNormalization

### 📱 Format Deployment

| Format | Kegunaan | File |
|---|---|---|
| Keras | Development | `finish_model.keras` |
| SavedModel | Production | `finish_model_savedmodel/` |
| TFLite | Mobile/Edge | `finish_model.tflite` |
| TFJS | Web Browser | `finish_model_tfjs/` |

---

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
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model('finish_model.keras')

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

---

## 📊 Struktur Project

```
klasifigambar/
├── finish_model.keras
├── Submission_Akhir.ipynb
├── requirements.txt
├── README.md
├── finish_model_savedmodel/
│   ├── saved_model.pb
│   └── variables/
├── finish_model_tfjs/
│   ├── group1-shard1of4.bin
│   ├── group1-shard2of4.bin
│   ├── group1-shard3of4.bin
│   ├── group1-shard4of4.bin
│   └── model.json
└── finish_model_tflite/
    ├── finish_model.tflite
    └── labels.txt
```

---

## 🚀 Deployment

Model siap untuk deployment dalam berbagai format:

- **Web Application**: Gunakan TensorFlow.js
- **Mobile App**: Gunakan TensorFlow Lite
- **Server API**: Gunakan SavedModel format
- **Research**: Gunakan Keras format

---

## 🎓 Sertifikat

<div align="center">

> 🏅 **Belajar Machine Learning untuk Pemula** — Dicoding Indonesia
>
> Diperoleh oleh **Silmi Azdkiatul Athqia**

[![Lihat & Verifikasi Sertifikat](https://img.shields.io/badge/🎓%20Lihat%20Sertifikat-Dicoding-06b6d4?style=for-the-badge)](https://www.dicoding.com/certificates/81P2LEW9NZOY)

</div>

---

## 📝 Credits

- **Dataset**: [Skyview Aerial Dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)
- **Original Sources**: AID Dataset & NWPU-Resisc45 Dataset
- **Architecture**: MobileNetV2 + Custom Layers

---

## 👩‍💻 Author

<div align="center">

**Silmi Azdkiatul Athqia**

[![Dicoding](https://img.shields.io/badge/Dicoding-silmiathqia-blue?style=flat-square)](https://www.dicoding.com/users/silmiathqia)
[![GitHub](https://img.shields.io/badge/GitHub-silmiaathqia-black?style=flat-square&logo=github)](https://github.com/silmiaathqia)

🎓 Laskar AI 2025 Cohort — Mahasiswa & Fresh Graduate

</div>

---

<div align="center">
<i>Proyek Klasifikasi Gambar — Belajar Machine Learning untuk Pemula — Dicoding 2025</i>
<br/>
<sub>Made with ❤️ by Silmi Azdkiatul Athqia</sub>
</div>
