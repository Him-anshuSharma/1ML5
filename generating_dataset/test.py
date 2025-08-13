import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Make sure we can import constants.py from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import journals  # This should be a list of journal texts

# Path to your fine-tuned model folder
MODEL_PATH = "./finetuned_emotion"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Create a pipeline for inference
emotion_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

# Run predictions on all journals
for i, journal in enumerate(journals, start=1):
    print(f"\nJournal {i}: {journal}")
    results = emotion_pipeline(journal)
    for r in results[0]:
        print(f"{r['label']}: {r['score']:.4f}")
