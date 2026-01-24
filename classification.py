import re
import nltk
import json
import streamlit as st
import pandas as pd

from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from doc_gathering.collector import JSON_FILE


class Classification:

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
            balanced.extend(category_docs_list[:40])

        categories = pd.Series([d["category"] for d in balanced]).value_counts()
        st.sidebar.write("Dataset:", categories.to_dict())
        return balanced
