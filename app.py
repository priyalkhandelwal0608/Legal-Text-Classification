# app.py
import streamlit as st
from transformers import pipeline

# -----------------------------
# Load LegalBERT pipeline
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="nlpaueb/legal-bert-base-uncased")

classifier = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Legal Case Outcome Predictor", page_icon="⚖️", layout="centered")

st.title("⚖️ Legal Case Outcome Predictor")
st.write("Enter a legal case text and predict whether the case was **Won** or **Lost**.")

# Demo run option
if st.button("Run Demo Case"):
    st.session_state["case_text"] = """The court considered whether refusal of a Calderbank offer 
                                       was unreasonable and awarded indemnity costs to the applicant."""
else:
    st.session_state.setdefault("case_text", "")

# Text input
case_text = st.text_area("Enter case text:", value=st.session_state["case_text"], height=200)

# Predict button
if st.button("Predict Outcome"):
    if case_text.strip():
        result = classifier(case_text, truncation=True)

        # Map generic labels to "Won"/"Lost"
        # NOTE: This is a placeholder mapping since LegalBERT isn't fine-tuned yet.
        # You can adjust based on your dataset outcomes.
        label = result[0]["label"]
        score = result[0]["score"]

        # Simple heuristic: LABEL_0 = Won, LABEL_1 = Lost
        outcome = "Case Won" if label in ["LABEL_0"] else "Case Lost"

        st.subheader("Prediction Result")
        st.write(f"**{outcome}** (confidence: {score:.2f})")
    else:
        st.warning("Please enter some text before predicting.")