import time
import spacy
import pytextrank
import numpy as np
import matplotlib.pyplot as plt
import json
from transformers import pipeline
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import journals


# ====== Emotion Classifiers ======
def get_emotion_classifier(option="default"):
    """Load the appropriate emotion classification pipeline."""
    if option == "default":
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        class_names = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    elif option == "goemotions":
        classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
        class_names = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
                       "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude",
                       "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse",
                       "sadness", "surprise", "neutral"]
    elif option == "custom":
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        class_names = ["anger", "joy", "sadness"]
    else:
        raise ValueError("Invalid classifier option")
    return classifier, class_names


def classify_emotions(journals, emotion_classifier, class_names):
    """Run emotion classification on all journals."""
    all_scores = []
    for text in journals:
        results = emotion_classifier(text)[0]
        scores = [0] * len(class_names)
        for r in results:
            if r['label'] in class_names:
                scores[class_names.index(r['label'])] = r['score']
        all_scores.append(scores)
    return all_scores

# ====== Extractive Summary ======
def important_points(journals):
    """Extractive summary using PyTextRank."""
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("textrank")
    summaries = []
    for text in journals:
        doc = nlp(text)
        sentences = [sent.text for sent in doc._.textrank.summary(limit_sentences=2)]
        summaries.append(" ".join(sentences))
    return summaries

# ====== Abstractive Summary ======
def summarize_abstractive(journals):
    """Generate only abstractive summaries."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []
    for text in journals:
        result = summarizer(text, max_length=60, min_length=10, do_sample=False)
        summaries.append(result[0]['summary_text'])
    return summaries

# ====== Creative Abstractive Summary ======
def summarize_creative(journals):
    """Generate creative summaries with more imaginative phrasing."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []
    for text in journals:
        prompt = f"Write a vivid, creative summary of the following text:\n{text}\nSummary:"
        result = summarizer(prompt, max_length=80, min_length=15, do_sample=True, temperature=1.0)
        summaries.append(result[0]['summary_text'])
    return summaries

# ====== Plotting ======
def plot_graph(emotion_scores, class_names, journal_index):
    """Plot emotion scores for a given journal."""
    scores = emotion_scores[journal_index]
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, scores)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Score")
    plt.title(f"Emotion Distribution for Journal {journal_index + 1}")
    plt.tight_layout()
    plt.show()

# ====== Results Printing ======
def print_results(results):
    for res in results:
        print("\nJournal:", res["journal"])
        print("Summary:", res["summary"])
        print("Emotions:", res["emotion_scores"])

# ====== Save Results ======
def save_results(results, classifier_type, summary_type):
    """Save results to a JSON file."""
    filename = f"results_{classifier_type}_summary{summary_type}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"✅ Results saved to {filename}")

def run_pipeline(classifier_type, summary_type):
    """
    classifier_type: "default", "goemotions", "custom"
    summary_type: 1 = extractive, 2 = abstractive, 3 = creative abstractive
    """
    start = time.time()

    # Load classifier
    emotion_classifier, class_names = get_emotion_classifier(option=classifier_type)
    emotion_scores = classify_emotions(journals, emotion_classifier, class_names)

    # Summaries
    if summary_type == 1:
        summaries = important_points(journals)  # Extractive
    elif summary_type == 2:
        summaries = summarize_abstractive(journals)  # Abstractive
    elif summary_type == 3:
        summaries = summarize_creative(journals)  # Creative
    else:
        raise ValueError("Invalid summary type")

    # Combine results
    results = []
    for i, journal in enumerate(journals):
        results.append({
            "journal": journal.strip(),
            "summary": summaries[i].strip(),
            "emotion_scores": {cls: float(score) for cls, score in zip(class_names, emotion_scores[i])}
        })

    end = time.time()
    print(f"[{classifier_type.upper()} | Summary Type {summary_type}] Time Taken: {round(end - start, 2)}s")

    save_results(results, classifier_type, summary_type)

# ====== Run All Combinations ======
if __name__ == "__main__":
    run_pipeline("default", 1)
    # run_pipeline("default", 2)
    # run_pipeline("default", 3)
    # run_pipeline("goemotions", 1)
    # run_pipeline("goemotions", 2)
    # run_pipeline("goemotions", 3)
    # run_pipeline("custom", 1)
    # run_pipeline("custom", 2)
    # run_pipeline("custom", 3)