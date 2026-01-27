import re
import nltk
import json
import streamlit as st
import pandas as pd
import numpy as np
import pickle

from datetime import datetime
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path

from doc_gathering.collector import JSON_FILE
from feedback import Feedback


class Classification:

    MODEL_FILE = Path("data/nb_model.pkl")

    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    def __init__(self):
        self.docs = []

    def preprocess(_self, text):
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = text.split()
        tokens = [
            Classification.stemmer.stem(t)
            for t in tokens
            if t not in Classification.STOP_WORDS and len(t) > 2
        ]
        return " ".join(tokens)

    # @st.cache_resource
    def load_model(_self, docs):
        """Load or train new"""
        if Classification.MODEL_FILE.exists():
            with open(Classification.MODEL_FILE, "rb") as f:
                state = pickle.load(f)
                pipeline = state["pipeline"]
                test_acc = state["test_acc"]
                cm = state["cm"]
                df_results = state["df_results"]
                X_train = state["X_train"]
                Y_train = state["Y_train"]
                st.sidebar.success("Using improved model with feedback!")
            return pipeline, test_acc, cm, df_results, X_train, Y_train

        st.sidebar.info("Training baseline model")
        return _self.train_model(docs)

    @st.cache_data
    def load_and_balance_docs(_self):
        if not JSON_FILE.exists():
            st.error("Data source file is missing")
            raise ValueError("Data source file is missing")

        print("Loading json file ...")
        with open(JSON_FILE) as f:
            docs = json.load(f)

        category_docs = defaultdict(list)
        for doc in docs:
            category_docs[doc["category"]].append(doc)

        balanced = []
        for category_docs_list in category_docs.values():
            balanced.extend(category_docs_list[:60])

        categories = pd.Series([d["category"] for d in balanced]).value_counts()
        st.sidebar.write("Dataset:", categories.to_dict())
        return balanced

    # @st.cache_data
    def train_model(_self, docs):
        texts = [_self.preprocess(d["text"]) for d in docs]
        labels = [doc["category"] for doc in docs]

        # Train(80)-Test(20) Split
        X_train, X_test, Y_train, Y_test = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )

        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(max_features=3000, min_df=2, ngram_range=(1, 2)),
                ),
                ("nb", MultinomialNB(alpha=1.0)),
            ]
        )

        pipeline.fit(X_train, Y_train)

        # Test accuracy
        Y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)

        # Confusion matrix
        cm = confusion_matrix(
            Y_test, Y_pred, labels=["business", "entertainment", "health"]
        )

        df_results = pd.DataFrame(
            {
                "text": [doc["text"][:100] + "..." for doc in docs],
                "true_category": labels,
                "predicted": pipeline.predict(texts),  # Full set predictions
            }
        )

        return pipeline, accuracy, cm, df_results, X_train, Y_train

    def predict_category(self, query: str, pipeline: Pipeline):
        processedQuery = self.preprocess(query)
        pred = pipeline.predict([processedQuery])[0]
        probs = pipeline.predict_proba([processedQuery])[0]
        confidence = np.max(probs)
        return pred, confidence, probs

    def retrain_with_feedback(self, docs):
        feedback = Feedback()
        fb = feedback.load_feedback()

        all_docs = docs + [{"text": f["query"], "category": f["category"]} for f in fb]
        pipeline, acc, cm, df_results, X_train, Y_train = self.train_model(all_docs)

        # Save COMPLETE state
        state = {
            "pipeline": pipeline,
            "test_acc": acc,
            "cm": cm,
            "df_results": df_results,
            "X_train": X_train,
            "Y_train": Y_train,
            "feedback_count": len(fb),
            "train_date": datetime.now().isoformat(),
        }

        with open(Classification.MODEL_FILE, "wb") as f:
            pickle.dump(state, f)

        return state
