# model.py
from transformers import pipeline

# -----------------------------
# Load LegalBERT pipeline
# -----------------------------
# This loads the pretrained LegalBERT model for text classification.
# Note: It is not fine-tuned on your dataset yet, so labels will be generic (LABEL_0, LABEL_1, ...).
classifier = pipeline("text-classification", model="nlpaueb/legal-bert-base-uncased")

# -----------------------------
# Predict function
# -----------------------------
def predict_case(case_text: str):
    """
    Run LegalBERT classification on a given case text.
    Args:
        case_text (str): Raw legal case text
    Returns:
        list: Prediction results with label and confidence score
    """
    result = classifier(case_text, truncation=True)
    return result

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    new_case = """The court considered whether refusal of a Calderbank offer 
                  was unreasonable and whether indemnity costs should be awarded."""
    outcome = predict_case(new_case)
    print("\nPredicted outcome:", outcome)