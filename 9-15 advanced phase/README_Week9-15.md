# Sentiment Analysis on Twitter Data – Weekly Progress (Weeks 8–15)

**Course:** Data Science – AI (Practical, Project-Oriented Course)  
**Project:** Sentiment Analysis on Twitter Data  
**Duration:** Weeks 8–16 (Advanced Phase)  
**Dataset:** [Twitter Text Mining Dataset](https://raw.githubusercontent.com/HarshalSanap/Twitter-Text-Mining/master/twittersubset.csv)  
**Deployment:** *Streamlit App (Link will be added upon publication)*  

---

## 📘 Overview

This document outlines project development from **Week 8 to Week 16**, focusing on the integration of unsupervised learning, artificial neural networks, NLP preprocessing, model deployment, and explainability. This phase transformed the baseline sentiment model into a fully functional, deployed AI application.

---

## 🗓️ Week 8 – Unsupervised Learning

**Objectives:**
- Explore unsupervised methods such as clustering and dimensionality reduction.
- Visualize data structure and relationships between tweets.

**Steps Taken:**
- Applied **K-Means clustering** on TF-IDF vectors to identify latent sentiment groups.
- Used **PCA (Principal Component Analysis)** for 2D visualization.

**Outcome:**
- Confirmed that sentiment clusters align with model predictions.
- Strengthened understanding of feature separability.

---

## 🗓️ Week 9 – Neural Network (ANN)

**Objectives:**
- Introduce a feedforward neural network (ANN) for sentiment prediction.
- Compare its performance with classical models.

**Steps Taken:**
- Implemented a **Dense-layer ANN** using Keras with ReLU activations.
- Used **binary_crossentropy** loss and **Adam optimizer**.
- Trained on TF-IDF features.

**Sample Code:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(256, activation='relu', input_dim=X_train.shape[1]),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**Outcome:**
- Achieved performance comparable to Logistic Regression.
- Demonstrated ability to implement and train neural networks.

---

## 🗓️ Week 10 – Advanced Deep Learning

**Objectives:**
- Evaluate the use of advanced models (CNN/RNN) for potential improvement.
- Understand architectures for text data.

**Steps Taken:**
- Studied LSTM and RNN architectures for NLP.
- Experimented with simple sequence models on tokenized text (optional).

**Outcome:**
- Determined that TF-IDF + Logistic Regression provided best balance of performance and simplicity.

---

## 🗓️ Week 11 – Natural Language Processing (NLP)

**Objectives:**
- Apply core NLP preprocessing steps.
- Enhance text representation for sentiment classification.

**Steps Taken:**
- Implemented **tokenization**, **stopword removal**, and **TF-IDF weighting**.
- Used `nltk` for text preprocessing.
- Visualized frequent positive and negative words.

**Code Snippet:**
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

tokens = df['clean_text'].apply(word_tokenize)
```

**Outcome:**
- Created a clean and semantically rich text corpus ready for modeling.
- Enhanced model interpretability.

---

## 🗓️ Week 12 – AI in Data Science (Case Study)

**Objectives:**
- Connect project with real-world applications.
- Discuss ethical and industrial implications.

**Steps Taken:**
- Researched sentiment analysis use cases in marketing and customer service.
- Documented industry relevance in the report.

**Outcome:**
- Demonstrated understanding of real-world deployment scenarios and potential biases.

---

## 🗓️ Week 13 – Model Deployment (Streamlit)

**Objectives:**
- Deploy the final Logistic Regression model using Streamlit.
- Enable real-time user input for sentiment prediction.

**Steps Taken:**
- Saved trained model with `joblib`.
- Created `app.py` Streamlit script to load model and accept user input.

**Code Example:**
```python
import streamlit as st
import joblib

model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("Twitter Sentiment Analysis")
user_input = st.text_area("Enter a tweet:")
if st.button("Analyze"):
    transformed = vectorizer.transform([user_input])
    prediction = model.predict(transformed)[0]
    st.write(f"Predicted Sentiment: {prediction}")
```

**Outcome:**
- Achieved successful local deployment via Streamlit.
- Prepared for online publishing after GitHub integration.

---

## 🗓️ Week 14 – Ethics & Explainability

**Objectives:**
- Interpret and explain model predictions.
- Reflect on ethical AI practices.

**Steps Taken:**
- Applied SHAP and LIME techniques to interpret word-level impact.
- Discussed bias, transparency, and accountability in sentiment analysis.

**Outcome:**
- Added interpretability insights to the final report.
- Reinforced ethical understanding of AI deployment.

---

## 🗓️ Week 15 – Project Finalization

**Objectives:**
- Finalize project notebook, documentation, and presentation.
- Prepare GitHub repository for submission.

**Steps Taken:**
- Cleaned all notebook outputs.
- Added visual summaries, metrics tables, and final reflection.
- Structured repository with README, model files, and app.

**Outcome:**
- Ready-to-submit project repository and documentation.

---

## 🗓️ Week 16 – Final Demo & Viva

**Objectives:**
- Present and defend the completed project.

**Steps Taken:**
- Demonstrated the Streamlit app live.
- Explained model pipeline, evaluation, and deployment workflow.

**Outcome:**
- Successfully completed the end-to-end project and defense.
- Project ready for publication and online deployment.

---

## 🧩 Summary

Weeks 8–16 transformed the baseline sentiment classifier into a complete AI solution:
- Integrated unsupervised, neural, and NLP-based methods.
- Deployed interactive Streamlit app for real-time predictions.
- Demonstrated ethical, explainable, and applied understanding of AI in data science.

The final deliverable is a fully functional, well-documented project showcasing the integration of **data preprocessing, ML modeling, neural networks, NLP, and deployment** — aligned with professional data science workflows.

---
