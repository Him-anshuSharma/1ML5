from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import time
import spacy
import pytextrank
import numpy as np
from matplotlib import pyplot as plt
import os
import json

# ===============================
#   Emotion Classification
# ===============================
def classify_emotions(texts, emotion_classifier, class_names):
    results = []
    for text in texts:
        outputs = emotion_classifier(text, return_all_scores=True)[0]
        scores = [next((item['score'] for item in outputs if item['label'] == c), 0) for c in class_names]
        results.append(scores)
    return np.array(results)


# ===============================
#   Simple Helpers
# ===============================
def word_count(text):
    return len(text.split())

def compression_ratio(original, summary):
    return round(len(original.split()) / len(summary.split()), 2) if len(summary.split()) else None


# ===============================
#   Abstractive Summarization
# ===============================
def summarize_journals(journals):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    abstractive_summaries = []
    creative_summaries = []
    for journal in journals:
        abs_sum = summarizer(journal, max_length=100, min_length=10, do_sample=False)[0]['summary_text']
        cre_sum = summarizer(journal, max_length=100, min_length=10, do_sample=True, top_k=50, top_p=0.95)[0]['summary_text']
        abstractive_summaries.append(abs_sum)
        creative_summaries.append(cre_sum)
    return abstractive_summaries, creative_summaries


# ===============================
#   Extractive Summarization (TextRank)
# ===============================
def important_points(journals):
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("textrank")
    extractive_summaries = []
    for journal in journals:
        doc = nlp(journal)
        summary = " ".join([sent.text for sent in doc._.textrank.summary(limit_sentences=2)])
        extractive_summaries.append(summary)
    return extractive_summaries


# ===============================
#   LIME Explanation
# ===============================
def generate_lime_explanation(text, emotion_classifier, class_names, num_features=5, top_labels=1):
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba(texts):
        results = []
        for t in texts:
            outputs = emotion_classifier(t, return_all_scores=True)[0]
            scores = [next((item['score'] for item in outputs if item['label'] == c), 0) for c in class_names]
            results.append(scores)
        return np.array(results)

    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=num_features,
        top_labels=top_labels
    )

    top_label = exp.available_labels()[0]
    explanation_list = exp.as_list(label=top_label)

    return {
        "top_emotion": class_names[top_label],
        "explanation": [{"word": word, "weight": weight} for word, weight in explanation_list]
    }


# ===============================
#   Save Results
# ===============================
def save_results(results, filename):
    current_file = os.path.splitext(os.path.basename(filename))[0]
    results_folder = f"results/{current_file}"
    os.makedirs(results_folder, exist_ok=True)
    output_path = os.path.join(results_folder, "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_path}")


# ===============================
#   Plot Emotion Graph
# ===============================
def plot_graph(emotion_scores, class_names, journal_index=0):
    labels = class_names
    scores = emotion_scores[journal_index]
    
    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores, color='skyblue')
    plt.xlabel('Emotions')
    plt.ylabel('Scores')
    plt.title(f'Emotion Analysis for Journal {journal_index + 1}')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


# ===============================
#   Model Options
# ===============================
def get_emotion_classifier(option="default"):
    """
    option: "default" -> 7-class model
            "goemotions" -> Google's 28-class GoEmotions
            "custom" -> another emotion model
    """
    if option == "default":
        model_id = "j-hartmann/emotion-english-distilroberta-base"
        class_names = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    elif option == "goemotions":
        model_id = "SamLowe/roberta-base-go_emotions"
        # From GoEmotions 28-class version
        class_names = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]

    elif option == "custom":
        model_id = "bhadresh-savani/distilbert-base-uncased-emotion"
        class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    else:
        raise ValueError("Unknown classifier option.")

    emotion_classifier = pipeline(
        "text-classification",
        model=model_id,
        top_k=None
    )
    return emotion_classifier, class_names


# ===============================
#   Main
# ===============================
if __name__ == "__main__":
    from constants import journals  # Your journal entries list
    start = time.time()

    # Choose model: "default", "goemotions", "custom"
    emotion_classifier, class_names = get_emotion_classifier(option="goemotions")

    # Emotion classification
    emotion_scores = classify_emotions(journals, emotion_classifier, class_names)

    # Summarization
    abstractive_summaries, creative_summaries = summarize_journals(journals)

    # Extractive summary
    extractive_summaries = important_points(journals)

    # Build results
    results = []
    for i, journal in enumerate(journals):
        lime_exp = generate_lime_explanation(journal, emotion_classifier, class_names)
        results.append({
            "journal": journal.strip(),
            "extractive_summary": extractive_summaries[i].strip(),
            "abstractive_summary": abstractive_summaries[i].strip(),
            "creative_abstractive_summary": creative_summaries[i].strip(),
            "emotion_scores": {cls: float(score) for cls, score in zip(class_names, emotion_scores[i])},
            "lime_explanation": lime_exp
        })

    # Save results
    save_results(results, __file__)

    end = time.time()
    print(f"Total processing time: {round(end - start, 2)} seconds")

    # Plot emotion graphs
    for i in range(len(journals)):
        plot_graph(emotion_scores, class_names, i)
