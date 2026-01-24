import streamlit as st

from classification import Classification
from doc_gathering.collector import collect

st.set_page_config(page_title="Document Classification", layout="wide")
st.title("Classification by category")
st.markdown("Classification (Naive-Bayes) on Business/Entertainment/Health dataset")

classification = Classification()
try:
    docs = classification.load_and_balance_docs()
    load_docs = False
except:
    load_docs = st.button("Load Documents")
    if load_docs:
        collect()
    st.stop()
