# Eksperimen_SML_Silmi-Azdkiatul-Athqia

## Deskripsi
Repository ini berisi eksperimen machine learning untuk analisis produktivitas remote worker menggunakan dataset produktivitas karyawan. Implementasi mencakup pipeline preprocessing otomatis yang komprehensif untuk mempersiapkan data siap training.

## Sumber Dataset

**Sumber:** Kaggle - Remote Worker Productivity Dataset  
**URL:** https://www.kaggle.com/datasets/ziya07/remote-worker-productivity-dataset  
**Penyedia:** Ziya07  
**Jenis Data:** Dataset simulasi dengan representasi yang realistis dari kondisi kerja remote

## Karakteristik Utama Dataset

Dataset ini mencakup informasi demografis, pola kerja harian, penggunaan alat bantu AI, metrik penyelesaian tugas, dan skor produktivitas yang telah dihitung. Variabel target (`productivity_label`) mengkategorikan produktivitas ke dalam tiga kelas: Tinggi (High), Sedang (Medium), dan Rendah (Low).

### Fitur-Fitur Unggulan:
1. **Keberagaman Geografis**: Data simulasi dari pekerja remote di daerah perkotaan, semi-perkotaan, dan pedesaan
2. **Cakupan Multi-Industri**: Meliputi sektor IT, Kesehatan, Pendidikan, Keuangan, dan Ritel
3. **Metrik Penggunaan AI/ML**: Melacak frekuensi penggunaan dan dampak dari alat-alat cerdas
4. **Indikator Manajemen Waktu**: Jam kerja, penjadwalan tugas, pola istirahat
5. **Variabel Target**: `productivity_label` dengan kategori High, Medium, Low

## Struktur Repository

```
Eksperimen_SML_Silmi-Azdkiatul-Athqia/
├── .github/
│   └── workflows/
│       └── preprocessing_workflow.yml
├── remote_worker_productivity_raw.csv
├── preprocessing/
│   ├── Eksperimen_Silmi-Azdkiatul-Athqia.ipynb
│   ├── automate_silmi-azdkiatul-athqia.py
│   └── preprocessed_data/
│       ├── data_train.csv
│       ├── data_validation.csv
│       ├── data_test.csv
│       ├── label_encoder.pkl
│       ├── feature_scaler.pkl
│       ├── label_mapping.csv
│       ├── feature_names.csv
│       └── preprocessing_summary.json
├── README.md
└── requirements.txt
```

## Cara Penggunaan

### 1. Manual Experiment
```bash
# Buka Jupyter notebook untuk eksperimen manual
jupyter notebook preprocessing/Eksperimen_Silmi-Azdkiatul-Athqia.ipynb
```

### 2. Automatic Preprocessing Pipeline
```python
from preprocessing.automate_silmi_azdkiatul_athqia import RemoteWorkerDataPreprocessor

# Inisialisasi preprocessor
preprocessor = RemoteWorkerDataPreprocessor(random_state=42)

# Jalankan pipeline preprocessing lengkap
result = preprocessor.preprocess(
    file_path='remote_worker_productivity_raw.csv',
    output_dir='preprocessing/preprocessed_data'
)

# Akses data yang sudah diproses
X_train = result['X_train']
X_val = result['X_val'] 
X_test = result['X_test']
y_train = result['y_train']
y_val = result['y_val']
y_test = result['y_test']
```

### 3. Load Preprocessed Data
```python
from preprocessing.automate_silmi_azdkiatul_athqia import load_preprocessed_data

# Muat data yang sudah diproses sebelumnya
data = load_preprocessed_data('preprocessing/preprocessed_data')
X_train = data['X_train']
y_train = data['y_train']
label_encoder = data['label_encoder']
feature_scaler = data['feature_scaler']
```

## Pipeline Preprocessing

Pipeline preprocessing otomatis mencakup tahapan berikut:

1. **Data Loading & Validation**
   - Memuat dataset dari file CSV
   - Validasi integritas data
   - Penanganan missing values dan duplikasi

2. **Data Cleaning**
   - Menghapus kolom yang tidak diperlukan (`worker_id`, `productivity_score`)
   - Pembersihan data yang tidak konsisten

3. **Feature Engineering**
   - One-hot encoding untuk fitur kategorikal (`location_type`, `industry_sector`)
   - Label encoding untuk target variable (`productivity_label`)

4. **Data Splitting**
   - Pembagian data: 60% training, 20% validation, 20% testing
   - Stratified sampling untuk menjaga distribusi kelas

5. **Feature Scaling**
   - StandardScaler untuk normalisasi fitur numerik
   - Mencegah bias akibat perbedaan skala fitur

6. **Artifact Saving**
   - Menyimpan semua model preprocessing (encoders, scalers)
   - Dokumentasi lengkap proses preprocessing

## Hasil Preprocessing

Setelah preprocessing selesai, Anda akan mendapatkan:

### Dataset Terproses:
- `data_train.csv`: Dataset training siap untuk model training
- `data_validation.csv`: Dataset validasi untuk hyperparameter tuning
- `data_test.csv`: Dataset testing untuk evaluasi final

### Model Artifacts:
- `label_encoder.pkl`: Encoder untuk decode prediksi kembali ke label asli
- `feature_scaler.pkl`: Scaler untuk normalisasi fitur pada data baru
- `label_mapping.csv`: Mapping antara label asli dan encoded
- `feature_names.csv`: Daftar nama fitur dan indeksnya
- `preprocessing_summary.json`: Informasi detail lengkap proses preprocessing

## Informasi Preprocessing

File `preprocessing_summary.json` berisi informasi detail seperti:
- Bentuk dataset asli dan setelah preprocessing
- Jumlah missing values dan duplikasi yang dihapus
- Informasi pembagian dataset
- Mapping label dan fitur kategorikal
- Parameter preprocessing yang digunakan

## Penggunaan dengan GitHub Actions

Repository ini dilengkapi dengan GitHub Actions workflow untuk otomatisasi preprocessing:

```yaml
# .github/workflows/preprocessing_workflow.yml
# Otomatis menjalankan preprocessing saat ada push ke main branch
```

## License

Proyek ini dibuat untuk keperluan pembelajaran kelas Membangun Sistem Machine Learning.

## Kontribusi

Silakan buat issue atau pull request untuk perbaikan dan pengembangan lebih lanjut.

---
*Dibuat dengan ❤️ untuk Machine Learning Specialization*
