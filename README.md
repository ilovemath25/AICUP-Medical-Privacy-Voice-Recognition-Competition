# Doctor-Patient Speech Recognition and SHI Tagging

This project focuses on **automatic speech recognition (ASR)** and **sensitive health information (SHI) tagging** in doctor-patient conversations. It includes:

- Speech-to-text transcription using `Whisper` model.
- Named Entity Recognition (NER) tagging for SHI using fine-tuned `BERT` model.

---

## 📁 Project Structure

```
├───audio_folder/
│   ├───Training_Dataset_01/
│   │   ├───audio/
│   │   │   └───*.wav
│   │   ├───task_answer1.txt
│   │   └───task_answer2.txt
│   ├───Training_Dataset_02/
│   │   ├───audio/
│   │   │   └───*.wav
│   │   ├───task_answer1.txt
│   │   └───task_answer2.txt
│   └───Validation_Dataset/
│       ├───audio/
│       │   └───*.wav
│       ├───task_answer1.txt
│       └───task_answer2.txt
├───model/
│   ├───ner_shi_model/
│   └───whisper/
├───NER_Shi/
│   │   combineData.py
│   │   combined_data.json
│   │   ner_shi.py
│   │   tag.txt
│   │   train.ipynb
│   └───__init__.py
├───Speech2Text/
│   │   speech2text.py
│   └───__init__.py
├── app.py
├── requirement.txt
└── README.md

````

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ilovemath25/AICUP-Medical-Privacy-Voice-Recognition-Competition.git
cd AICUP-Medical-Privacy-Voice-Recognition-Competition
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔽 Download Dataset & Model

### Dataset

Download the training and validation datasets from this Google Drive folder:

📁 **[Download Dataset](https://drive.google.com/drive/folders/1TAglXUvOtkoJSZcOF2CexcIhqqOwvtFD?usp=sharing)**

### Model

Download the fine-tuned ASR model from:

📁 **[Download Model](https://drive.google.com/drive/folders/1qQwq-Kedm0BBPTajaE8x0_w56yu3rQhh?usp=drive_link)**

---

## 🚀 Run the App

To transcribe audio and detect SHI tags:

```bash
python app.py
```

---
