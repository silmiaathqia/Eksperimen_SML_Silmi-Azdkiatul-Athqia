# 🔬 Eksperimen & Preprocessing Dataset
### Worker Productivity Classification — Kriteria 1 MSML

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12.7-blue?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.3.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Automated-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

<br/>

**Kriteria 1 — Proyek Akhir Kelas Membangun Sistem Machine Learning — Dicoding**

*oleh Silmi Azdkiatul Athqia (silmiathqia)*

</div>

---

## 📋 Deskripsi

Repository ini berisi tahapan **eksperimen dan preprocessing dataset** untuk klasifikasi produktivitas pekerja remote. Mencakup EDA (Exploratory Data Analysis), preprocessing pipeline, dan otomasi menggunakan GitHub Actions.

---

## 📂 Struktur Repository

```
Eksperimen_SML_Silmi-Azdkiatul-Athqia/
│
├── 📄 remote_worker_productivity_raw.csv       # Dataset mentah
│
├── 📁 .github/workflows/
│   └── ⚙️ preprocessing_workflow.yml           # GitHub Actions automation
│
└── 📁 preprocessing/
    ├── 📓 Eksperimen_Silmi_Azdkiatul_Athqia.ipynb  # Notebook eksperimen
    ├── 🐍 automate_silmi_azdkiatul_athqia.py        # Script otomasi preprocessing
    └── 📁 preprocessed_data/                        # Output dataset
        ├── data_train.csv
        ├── data_validation.csv
        ├── data_test.csv
        ├── feature_scaler.pkl
        ├── label_encoder.pkl
        ├── label_mapping.csv
        ├── feature_names.csv
        └── preprocessing_summary.json
```

---

## 🔄 Pipeline Preprocessing

```
Raw Dataset
    │
    ▼
Data Loading
    │
    ▼
Exploratory Data Analysis (EDA)
    │
    ▼
Data Cleaning & Handling Missing Values
    │
    ▼
Feature Encoding (Label Encoding)
    │
    ▼
Train / Validation / Test Split
    │
    ▼
Feature Scaling (StandardScaler)
    │
    ▼
Save Preprocessed Data ✅
```

---

## 📊 Dataset

**Remote Worker Productivity**

| Info | Detail |
|---|---|
| Target | `productivity_label` (High / Medium / Low) |
| Train set | 640 sampel |
| Validation set | 160 sampel |
| Test set | 200 sampel |
| Total fitur | 20 fitur |

---

## ⚙️ GitHub Actions Workflow

Workflow **Automated Data Preprocessing** berjalan otomatis setiap:
- Push ke branch `main`
- Scheduled (terjadwal bulanan)
- Manual trigger (`workflow_dispatch`)

```yaml
Trigger → Setup Python 3.12.7 → Install Dependencies
       → Run automate_silmi.py → Save Preprocessed Data ✅
```

![Workflow Status](https://github.com/silmiaathqia/Eksperimen_SML_Silmi-Azdkiatul-Athqia/actions/workflows/preprocessing_workflow.yml/badge.svg)

---

## 🚀 Cara Menjalankan

**Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Jalankan preprocessing otomatis:**
```bash
python preprocessing/automate_silmi_azdkiatul_athqia.py
```

**Atau jalankan notebook:**
```bash
jupyter notebook preprocessing/Eksperimen_Silmi_Azdkiatul_Athqia.ipynb
```

---

## 🎓 Sertifikat

<div align="center">

> 🏅 **Membangun Sistem Machine Learning** — Dicoding Indonesia
>
> Diperoleh oleh **Silmi Azdkiatul Athqia**

[![Lihat & Verifikasi Sertifikat](https://img.shields.io/badge/🎓%20Lihat%20Sertifikat-Dicoding-06b6d4?style=for-the-badge)](https://www.dicoding.com/certificates/JMZVOJ9O3XN9)

</div>

---

## 🔗 Repo Terkait

| Repository | Deskripsi |
|---|---|
| [SMSML_Silmi-Azdkiatul-Athqia](https://github.com/silmiaathqia/SMSML_Silmi-Azdkiatul-Athqia) | Submission utama |
| [Workflow-CI](https://github.com/silmiaathqia/Workflow-CI) | CI/CD Pipeline (K3) |

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
<i>Kriteria 1 — Membangun Sistem Machine Learning (MSML) — Dicoding 2025</i>
<br/>
<sub>Made with ❤️ by Silmi Azdkiatul Athqia</sub>
</div>
