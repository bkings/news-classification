# News Document Classifier (Naive Bayes)

Active learning document classification system for **Business, Entertainment, Health** news categories. Achieves **97.2% test accuracy** with user feedback loop for continuous improvement.

## 📋 Features
- ✅ Balanced dataset (60 docs per category from NewsApi)
- ✅ Multinomial Naive Bayes + TF-IDF (bi-grams)
- ✅ 97.2% test accuracy
- ✅ Train/test split + full metrics (confusion matrix, PRF)
- ✅ Active learning feedback → auto-retrain
- ✅ Live prediction with confidence scores


## 🛠️ Quick Start
```bash
git clone https://github.com/bkings/news-classification.git
cd news-classification
pip install -r requirements.txt

streamlit run classify_app.py
```

### Example predictions
"UK inflation falls" → Business (92%)

"Oscars nominations" → Entertainment (91%)

"NHS flu vaccine" → Health (95%)

"celebrity habits" → Entertainment (89%)

### Tech stack
Model: Scikit-learn MultinomialNB + TfidfVectorizer

Preprocess: NLTK (Porter stemmer, stopwords)

UI: Streamlit

Persistence: Pickle (model), JSON (feedback)

Metrics: Precision, Recall, F1, Confusion Matrix

