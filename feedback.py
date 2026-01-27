import json
import streamlit as st
from datetime import datetime
from pathlib import Path


class Feedback:

    FEEDBACK_FILE = Path("data/user_feedback.json")

    def __init__(self):
        pass

    # @st.cache_data
    def load_feedback(_self):
        if Feedback.FEEDBACK_FILE.exists():
            with open(Feedback.FEEDBACK_FILE) as f:
                return json.load(f)

        return []

    def save_feedback(_self, query, true_category):
        feedback = _self.load_feedback()
        feedback.append(
            {
                "query": query,
                "category": true_category,
                "timestamp": str(datetime.now()),
            }
        )
        with open(Feedback.FEEDBACK_FILE, "w") as f:
            json.dump(feedback, f, indent=2)
