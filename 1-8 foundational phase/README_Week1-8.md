# Sentiment Analysis on Twitter Data — Half | 8 Week Project

**Notebook:** `Sentiment_Analysis_on_Twitter_Data_firstHalf.ipynb`  
**Platform:** Google Colab (GPU-ready)  
**Dataset:** [Twitter Subset Dataset](https://raw.githubusercontent.com/HarshalSanap/Twitter-Text-Mining/master/twittersubset.csv)  
**Tools & Libraries:** Python, Pandas, NumPy, Scikit-learn, NLTK, TensorFlow, Matplotlib, Seaborn, WordCloud  

This repository contains the first half of a **16-week Data Science & AI course project** (Weeks 1–8).  
The project demonstrates an end-to-end **sentiment analysis workflow** on Twitter data, covering setup, data cleaning, visualization, classical ML modeling, and unsupervised learning.

---

## Week 1 — Setup & Orientation

**Goal:** Set up the project environment, verify GPU availability, and preview dataset.

**Work Done:**
- Installed essential data science and AI libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `nltk`, `tensorflow`, `keras`, etc.).
- Verified GPU configuration on Google Colab.
- Loaded a preview of the Twitter dataset directly from a GitHub-hosted CSV.
- Displayed the first 10 rows to confirm structure and encoding.

**Outcome:**  
Project environment configured and dataset previewed successfully — ensuring compatibility for future processing.

---

## Week 2 — Data Collection & Cleaning

**Goal:** Load the full dataset, handle missing values and duplicates, and perform initial text cleaning.

**Work Done:**
- Loaded the complete dataset and inspected its shape, column names, and null/duplicate values.
- Implemented a `basic_clean()` function using regex for:
  - Lowercasing text  
  - Removing URLs, mentions, digits, and punctuation  
  - Trimming extra spaces
- Created a new column `clean_text` for preprocessed tweets.
- Identified or created the `sentiment` and `sentiment_num` columns for supervised learning.

**Outcome:**  
Dataset fully cleaned with a new `clean_text` feature ready for analysis and model training.

---

## Week 3 — Data Visualization

**Goal:** Explore sentiment distribution, text length, and frequent word patterns.

**Work Done:**
- Plotted:
  - **Sentiment Distribution** using Seaborn countplot.
  - **Tweet Length Distribution** using histogram.
  - **Top 20 Words** using word frequency bar chart.
  - **Tweet Length vs Sentiment** scatter plot.
  - **Word Cloud** to visualize dominant words.
- Used insights to assess dataset balance and identify common patterns in text.

**Key Insights:**
- Sentiment labels were moderately balanced.
- Average tweet length was within 50–120 characters.
- Common words included frequent positive/neutral terms.
- Weak correlation between text length and sentiment polarity.

---

## Week 4 — Statistics & Correlation

**Goal:** Perform descriptive statistics and analyze numerical-text feature correlations.

**Work Done:**
- Generated summary statistics using `df.describe()`.
- Created new numerical features:
  - `text_len` — character count per tweet  
  - `word_count` — total words per tweet  
  - `avg_word_len` — average word length per tweet
- Mapped categorical sentiment labels to numeric (`sentiment_num`).
- Calculated correlation between numerical features and sentiment score.
- Visualized relationships using boxplots.

**Insights on Feature Relationships:**
| Feature | Correlation with Sentiment Score | Observation |
|----------|----------------------------------|--------------|
| Word Count | -0.0763 | Weak negative correlation |
| Avg. Word Length | +0.0886 | Weak positive correlation |
| Tweet Length | Slight variance by sentiment | No strong pattern |

Overall, text-derived numerical features had weak associations with sentiment. Stronger predictors are likely to come from the words themselves (TF-IDF features).

---

## Week 5 — Regression (Conceptual Demo)

**Goal:** Demonstrate regression by predicting sentiment score from tweet length.

**Work Done:**
- Applied **Linear Regression** using Scikit-learn.
- Feature: `text_len`  
  Target: `sentiment_num`
- Performed an 80/20 train-test split.
- Evaluated results using MSE, MAE, and RMSE.

**Regression Results**

| Metric | Value |
|--------|--------|
| Mean Squared Error (MSE) | 0.9968 |
| Mean Absolute Error (MAE) | 0.9971 |
| Root Mean Squared Error (RMSE) | 0.9984 |

**Interpretation:**  
An MAE close to 1.0 indicates that tweet length alone is a poor predictor of sentiment. This step served as a conceptual introduction to supervised regression.

---

## Week 6 — Classification (Classic ML)

**Goal:** Apply traditional machine learning models for sentiment classification.

**Work Done:**
- Used **TF-IDF Vectorizer** (with bigrams, 5,000 features) to transform text.
- Split data into training and test sets (80/20).
- Trained and compared three models:
  - Logistic Regression  
  - Multinomial Naive Bayes  
  - Random Forest Classifier
- Evaluated each using accuracy on the test set.

**Model Accuracy Comparison**

| Model | Accuracy |
|--------|-----------|
| Logistic Regression | 0.7320 |
| Naive Bayes | 0.7150 |
| Random Forest | 0.6930 |

**Insight:**  
Logistic Regression performed best, confirming its reliability for text-based sentiment classification tasks.

---

## Week 7 — Model Evaluation

**Goal:** Assess model performance using detailed evaluation metrics.

**Work Done:**
- Chose **Logistic Regression** for evaluation.
- Generated **classification report** (precision, recall, F1-score).
- Visualized **Confusion Matrix** using Seaborn heatmap.
- Analyzed key performance metrics.

**Reflection:**  
Metric importance depends on project goals:
- **Precision** — Important for ensuring positive predictions are accurate (e.g., brand marketing).
- **Recall** — Important for identifying all negative tweets (e.g., customer complaints).
- **F1-Score** — Balanced measure for general sentiment analysis (used here as the main evaluation metric).

---

## Week 8 — Unsupervised Learning

**Goal:** Explore clustering behavior using K-Means on TF-IDF features.

**Work Done:**
- Applied **K-Means clustering (k=3)** to TF-IDF vectors.
- Reduced dimensions using **PCA (2D)** for visualization.
- Plotted clusters using Seaborn scatter plot.

**Outcome:**  
Distinct clusters were observed, suggesting that sentiment and text similarity contribute to natural groupings in tweet data.

---

## Project Summary

This notebook demonstrates an **8-week mini-project** on sentiment analysis using Twitter data — aligned with the **Data Science & AI (Practical, Project-Oriented Course)** structure.

### Key Steps
1. **Setup & Data Loading** — Installed dependencies and previewed dataset.  
2. **Data Cleaning** — Removed noise and created a cleaned text column.  
3. **Visualization** — Explored sentiment distribution and frequent words.  
4. **Feature Analysis** — Explored correlations among text-based metrics.  
5. **Regression Demo** — Conceptual modeling using numerical features.  
6. **Classification Models** — Compared Logistic Regression, Naive Bayes, and Random Forest.  
7. **Evaluation** — Precision, recall, F1-score, and confusion matrix analysis.  
8. **Unsupervised Learning** — K-Means and PCA visualization for cluster exploration.

---

### Final Insights
- Logistic Regression achieved the **best classification accuracy (≈73%)**.  
- Simple text features like tweet length showed **weak correlation** with sentiment.  
- TF-IDF-based text representation significantly improved model performance.  
- Unsupervised analysis revealed natural separations in text embeddings.

---

### Repository Structure
