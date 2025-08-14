import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Allow importing constants.py from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import journals  # List of journal texts

stress_test_journals = [
    """Today was my graduation day. Everyone cheered as I walked across the stage, but I noticed my grandmother’s seat was empty. I smiled for photos, but my heart felt heavy.""",

    """I got another late email from my coworker. I said ‘No problem’ in reply, but my hands tightened on the keyboard. I wonder if they even notice.""",

    """I’m going on my first solo trip abroad. I’m excited, but my hands won’t stop fidgeting. I imagine losing my passport or missing flights.""",

    """My project was approved at work, and everyone congratulated me. I celebrated, but I kept thinking my colleague’s proposal didn’t succeed.""",

    """I spent the afternoon reorganizing my bookshelf. Everything was tidy and quiet, yet there was a faint tug in my chest — something missing."""
]


# Paths to your fine-tuned models
MODEL_PATHS = [
    "./finetuned_emotion",
    "./finetuned_emotion_subtle",
    "./finetuned_emotion_subtle_mix"
]
MODEL_NAMES = ["M1", "M2", "M3"]

# Load all models and tokenizers once
pipelines = []
for path in MODEL_PATHS:
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    emotion_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True
    )
    pipelines.append(emotion_pipeline)

# Run all journals and print results side by side
for i, journal in enumerate(stress_test_journals, start=1):
    print(f"\n--- Journal {i} ---")
    print(journal[:300] + ("..." if len(journal) > 300 else ""))  # preview text

    # Collect results from all models
    all_results = []
    for pipe in pipelines:
        results = pipe(journal)[0]  # get first batch
        # convert to {label: score} dict
        score_dict = {r['label']: r['score'] for r in results}
        all_results.append(score_dict)

    # Get all unique labels across models
    labels = sorted({label for r in all_results for label in r.keys()})

    # Print table-like side by side comparison
    print("\nLabel\t" + "\t".join(MODEL_NAMES))
    for label in labels:
        scores = [f"{r.get(label, 0):.4f}" for r in all_results]
        print(f"{label}\t" + "\t".join(scores))
