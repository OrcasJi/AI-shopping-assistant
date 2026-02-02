# 🛍️ AI Shopping Assistant (NLP-based Product Recommendation System)

An end-to-end **AI-powered shopping assistant** that understands natural language shopping queries and returns ranked product recommendations.  
The system combines **Intent Classification**, **Style Named Entity Recognition (NER)**, and **Fuzzy Product Search** to simulate an intelligent conversational shopping experience.

This project is developed as part of an **MSc Computer Science final project** and is designed with **modularity, extensibility, and research clarity** in mind.

---

## ✨ Key Features

- 🧠 **Intent Classification (BERT-based)**
  - Understands user intent such as product search, style inquiry, price constraints, greetings, etc.
- 🎨 **Style NER (BIO tagging with BERT)**
  - Extracts fashion styles from free-text queries (e.g. *sporty*, *vintage*, *formal*)
- 💰 **Price Range Extraction**
  - Supports expressions like `under $100`, `between 50 and 120`, `around 200`
- 🔍 **Fuzzy Product Search & Ranking**
  - Combines style similarity, name matching, and price closeness
- 🧪 **Training, Evaluation & Testing**
  - Early stopping, class weighting, confusion matrices, TensorBoard support
- 🔌 **API-ready Architecture**
  - Designed to be wrapped as a backend service (future extension)

---

## 📁 Project Structure
├── api/ # API layer (planned / partial)
├── config/ # Configuration files (e.g. search weights)
├── data/ # Training data, validation data, product catalog
├── models/ # Trained models and model definitions
├── recommend/ # Recommendation logic
├── scripts/ # Demo & utility scripts
├── test/ # Unit tests for each module
├── text/ # Text-related resources
├── train/ # Model training scripts
├── utils/ # Shared utilities (data loading, preprocessing)
├── web/ # Web or UI-related code (future extension)
├── config.py
├── requirement.txt
└── README.md

---

## 🧠 Models Overview

### 1️⃣ Intent Classifier
- **Model**: `bert-base-uncased`
- **Task**: Multi-class text classification
- **Loss**: Cross-Entropy / Focal Loss (configurable)
- **Metrics**: Accuracy, Macro-F1
- **Output**: User intent label

### 2️⃣ Style NER
- **Model**: `bert-base-uncased`
- **Tagging Scheme**: BIO (`O`, `B-STYLE`, `I-STYLE`)
- **Purpose**: Extract fashion-related style entities from user queries
- **Evaluation**: Token-level F1 score, confusion matrix

---

## 🚀 How to Run

### 1️⃣ Install dependencies

pip install -r requirement.txt

### 2️⃣ Prepare data
Intent data
data/intent_training_data_expanded.csv

### 2️⃣ Prepare data

Intent data

data/intent_training_data_expanded.csv


Product catalog

data/Shopping_product_catalog_expanded.csv

### 3️⃣ Train models

Intent classifier

python train/train_intent_classifier.py


Style NER

python train/train_style_ner.py

### 4️⃣ Run demo
python scripts/smoke_demo.py


Example input:

Looking for a sporty jacket under $100

###🧪 Testing

Run all unit tests with:
pytest test/


## Tests cover:

Data preprocessing

Intent dataset loading & splitting

Product catalog parsing

Style NER extraction

End-to-end model loading & inference

## 📊 Research & Design Considerations

Modular architecture for easy replacement of models

Explicit separation of NLP understanding and recommendation logic

Designed to support future experiments:

Model comparison (rule-based vs ML)

Confidence-aware dialogue flow

API vs local inference performance

## 🛠️ Tech Stack

Python

PyTorch

HuggingFace Transformers

scikit-learn

RapidFuzz

Pandas / NumPy

Matplotlib

PyTest

## 📌 Future Work

RESTful API with FastAPI

Multi-turn dialogue management

Transformer-based joint intent + slot model

User feedback loop & ranking optimisation

Frontend demo (Web / Chat UI)

## 👤 Author

Orcas Ji
MSc Computer Science
Queen Mary University of London

This project is intended for academic research, learning, and portfolio demonstration.
