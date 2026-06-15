# Hate Speech & Offensive Language Detection

A machine learning + NLP based moderation system that detects hate speech and offensive language from social media text and provides explainable predictions with real-time analysis.

## Overview

This project was built to explore real-world NLP classification on noisy social media data.
The system classifies input text into:

* Hate Speech
* Offensive Language
* Neutral Content

Along with prediction, the system also provides:

* Confidence score
* Severity level (Low / Medium / High)
* Toxic word highlighting
* Explanation generation
* AI-based neutral text rewriting

The project combines traditional NLP techniques with a lightweight Generative AI layer to create a more practical moderation workflow.

---

## Features

### NLP Classification

* TF-IDF based feature extraction
* Logistic Regression text classification
* 3-class prediction system

### Moderation Outputs

* Toxic word extraction
* Severity analysis
* Confidence scoring
* Prediction explanations

### Real-Time System

* FastAPI backend
* Streamlit frontend
* API-based workflow

### GenAI Neutralization

* Gemini API integration
* Rewrites toxic text into respectful language
* Triggered only for toxic/offensive predictions

---

## Dataset

The dataset contains around 24,781 annotated English tweets collected for hate speech detection tasks.

### Classes

* `0` → Hate Speech
* `1` → Offensive Language
* `2` → Neutral Content

### Dataset Challenges

* Highly imbalanced classes
* Noisy Twitter text
* Slang and abbreviations
* Offensive vocabulary
* Mentions, URLs, punctuation noise

---

## Preprocessing

The following preprocessing steps were applied:

* URL removal
* HTML entity removal
* Mention replacement (`@user → user`)
* Lowercasing
* Punctuation removal
* Number removal
* Stopword removal

The cleaned text was stored as `processed_text`.

---

## Model Training

### Vectorization

TF-IDF vectorization was used to convert text into numerical features.

### Models Compared

* Logistic Regression
* Class-weighted Logistic Regression
* Support Vector Machine (SVM)

The final model selected was class-weighted Logistic Regression due to better minority-class detection performance.

---

## System Architecture

User
↓
Streamlit Frontend
↓
FastAPI Backend
↓
Input Validation
↓
Preprocessing Layer
↓
TF-IDF Vectorizer
↓
ML Model
↓
Post-processing
↓
Gemini Neutralization Layer
↓
JSON Response
↓
Frontend Display

---

## Tech Stack

### Backend

* FastAPI
* Python

### Frontend

* Streamlit

### ML / NLP

* Scikit-learn
* TF-IDF
* Logistic Regression
* NLTK

### GenAI

* Gemini API

---

## Additional Engineering Features

### Caching

Repeated toxic inputs are cached to avoid unnecessary Gemini API calls.

### Logging

Predictions and outputs are logged for monitoring and debugging.

### Error Handling

* Empty input handling
* Gemini API fallback handling
* Validation checks

---

## Future Improvements

* Transformer-based models (BERT/DistilBERT)
* Multilingual moderation
* Better contextual toxicity understanding
* Real-time deployment optimization
* Database-backed moderation logs
* Advanced ensemble methods

---

## Learning Outcomes

This project helped in understanding:

* NLP preprocessing
* Text vectorization
* ML-based text classification
* Handling imbalanced datasets
* API development
* Frontend-backend integration
* System design for ML applications
* Combining ML with Generative AI

---

## Author

Manaswini Reddy
