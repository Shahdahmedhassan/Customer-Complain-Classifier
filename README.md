# 🗂️ Customer Complain Classifier

A deep learning NLP project that automatically classifies customer complaint narratives into the correct category, using a fine-tuned Transformer model deployed as an interactive web app.

## 🚀 Live Demo

**[Customer Complain Classifier - a Hugging Face Space by ShahdCoder](https://huggingface.co/spaces/ShahdCoder/Customer_Complain_Classifier)**

Try it out — type in a complaint narrative and get back the predicted category with a confidence score.

## 📋 About the Project

Companies receive thousands of customer complaints every day, and manually sorting them into categories (credit cards, debt collection, mortgages, etc.) is slow and costly. This project builds an end-to-end NLP pipeline that automates that process:

1. Clean and preprocess raw complaint text
2. Train and compare several deep learning architectures
3. Fine-tune a pretrained Transformer for the same task
4. Pick the best-performing model and deploy it as a live web app

## 🧠 What Was Used

**Models trained and compared:**
- SimpleRNN
- LSTM
- GRU
- Fine-tuned Transformer (DistilBERT, via HuggingFace)

**Text preprocessing:**
- Lowercasing
- Removing punctuation, numbers, and special characters
- Removing stopwords
- Lemmatization

**Evaluation metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

**Tools & libraries:**
- Python
- TensorFlow / Keras (for SimpleRNN, LSTM, GRU)
- HuggingFace Transformers & Datasets (for the fine-tuned model)
- Scikit-learn (metrics, preprocessing)
- Gradio (interactive web app)
- Kaggle Notebooks (training environment, GPU)
- Hugging Face Spaces (deployment)

## 📁 Repository Contents

- `app.py` — Gradio web app used for deployment
- `requirements.txt` — Python dependencies for the app
- `transformer_model/` — Fine-tuned Transformer weights and tokenizer
- Notebook — full training pipeline (preprocessing, training, evaluation, comparison)

## 🏷️ Complaint Categories

The model classifies complaints into categories such as:
- Credit card
- Credit reporting
- Debt collection
- Mortgages and loans
- Retail banking

## ✨ App Features

- Single complaint classification with confidence scores
- Batch classification (classify multiple complaints at once)
- Downloadable session report
- Clean, easy-to-use interface
