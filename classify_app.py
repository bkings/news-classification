import streamlit as st
import plotly.express as px
import time

from classification import Classification
from doc_gathering.collector import collect
from feedback import Feedback

from pathlib import Path
from sklearn.metrics import classification_report

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

pipeline, test_acc, cm, df_results, X_train, y_train = classification.load_model(docs)
tab1, tab2, tab3 = st.tabs(["Classify New Doc", "Model Performance", "Predictions"])

with tab1:
    st.header("Classify Document")
    queryCol, buttonCol = st.columns([15, 1], gap=None)

    with queryCol:
        new_document = st.text_input(
            "Enter query:",
            placeholder="Eg. 'Top business ...' or 'Latest celebrity gossip ...'",
            label_visibility="collapsed",
        )

    with buttonCol:
        search_clicked = st.button("🔍")

    if new_document or (new_document and search_clicked):
        pred, confidence, probs = classification.predict_category(
            new_document.strip(), pipeline
        )
        st.success(f"**Predicted: {pred.title()}** (Confidence: {confidence:.1%})")

        # Feedback buttons
        colA, colB = st.columns(2)
        feedback = Feedback()

        if "correct_clicked" not in st.session_state:
            st.session_state.correct_clicked = False

        if "wrong_clicked" not in st.session_state:
            st.session_state.wrong_clicked = False

        if colA.button("✔ Correct", key="correct"):
            st.session_state.correct_clicked = True

        if st.session_state.correct_clicked:
            feedback.save_feedback(new_document, pred)
            st.success("Saved to dataset!")
            time.sleep(1.2)
            st.session_state.correct_clicked = False
            st.rerun()

        if colB.button("⤫ Wrong", key="wrong"):
            st.session_state.wrong_clicked = True

        if st.session_state.wrong_clicked:
            st.error("Thank you for your feedback! Manual correction required.")
            true_category = st.selectbox(
                "Correct category: ",
                ["business", "entertainment", "health"],
                key="selected_category",
            )
            if st.button("Save correction"):
                feedback.save_feedback(new_document, true_category)
                st.success("Corrected and saved!")
                time.sleep(1.2)
                st.session_state.wrong_clicked = False
                st.rerun()

        col1, col2, col3 = st.columns(3)
        col1.metric("Business", f"{probs[0]:.1%}")
        col2.metric("Entertainment", f"{probs[1]:.1%}")
        col3.metric("Health", f"{probs[2]:.1%}")

    st.markdown("---")
    # Test cases
    tests = {
        "Business": "UK economy grows amid rate cuts.",
        "Entertainment": "Oscars 2026 best film nominees.",
        "Health": "NHS launches new flu vaccine campaign.",
        "Short": "Kim Kardashian gossip",
        "Long": "Jennifer lawrence speaking in favour of Donald Trump so that she can get work in films soon",
        "Mixed": "Movie box office boosts economy",
        "With stop words": "a common food preservative that can risk organs and feels more heavy when consumed",
    }
    st.subheader("✅ Test Cases")
    for name, test in tests.items():
        p, c, _ = classification.predict_category(test, pipeline)
        st.caption(f"**[{name}]** {test} → {p.title()} ({c:.1%})")

with tab2:
    st.header("Model Performance")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Test Accuracy", f"{test_acc:.1%}")
        st.subheader("Confusion Matrix")
        fig_cm = px.imshow(
            cm,
            x=["Business", "Entertainment", "Health"],
            y=["Business", "Entertainment", "Health"],
            color_continuous_scale="Blues",
            title="Predictions vs True",
        )
        st.plotly_chart(fig_cm)

    with col2:
        st.subheader("Classification Report")
        report = classification_report(
            y_train,
            pipeline.predict(X_train),
            target_names=["business", "entertainment", "health"],
            output_dict=True,
        )
        st.json({k: f"{v:.3f}" for k, v in report.items() if isinstance(v, float)})

with tab3:
    st.header("All predictions")
    st.dataframe(df_results, width="stretch", hide_index=True)

if st.sidebar.button("🔄 Retrain with feedback (5s)"):
    with st.sidebar.spinner("Re-training with updated data ..."):
        time.sleep(2)
        state = classification.retrain_with_feedback(docs)
        st.sidebar.success(f"Re-trained! New Accuracy: {state['test_acc']:.1%}")
        st.rerun()
st.markdown("---")
