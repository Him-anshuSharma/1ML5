import spacy
import pytextrank
from constants import journals
from transformers import pipeline
import time
import os, json
import sys
from collections import defaultdict
from matplotlib import pyplot as plt

# Utility functions
def word_count(text):
    return len(text.split())

def compression_ratio(original, summary):
    return round(len(original.split()) / len(summary.split()), 2) if len(summary.split()) else None

# Create results folder
current_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
results_folder = f"results/{current_file}"
os.makedirs(results_folder, exist_ok=True)

start = time.time()

# Load summarization pipelines
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
creative_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Generate abstractive and creative summaries
abstractive_summaries = [
    abstractive_summarizer(x, max_length=100, min_length=10, do_sample=False)[0]['summary_text']
    for x in journals
]

creative_summaries = [
    creative_summarizer(x, max_length=100, min_length=10, do_sample=True, top_k=50, top_p=0.95)[0]['summary_text']
    for x in journals
]

# Extractive summarizer using spaCy + TextRank
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("textrank")

docs = [nlp(journal) for journal in journals]
extractive_summaries = [
    " ".join([sent.text for sent in doc._.textrank.summary(limit_sentences=2)])
    for doc in docs
]

# Emotion classifier pipeline
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Sentence-level emotion function on full journal text
def sentence_level_emotions(text):
    doc = nlp(text)
    sentence_emotions = []
    for sent in doc.sents:
        emotions = emotion_analyzer(sent.text)
        top_emotion = max(emotions[0], key=lambda x: x['score'])
        sentence_emotions.append({
            "sentence": sent.text.strip(),
            "emotion": top_emotion['label'],
            "score": top_emotion['score']
        })
    return sentence_emotions

# Overall emotion analysis on combined text (journal + abstractive + extractive)
sorted_emotions = []
sentence_emotion_details = []

for (journal, asum, esum) in zip(journals, abstractive_summaries, extractive_summaries):
    combined_text = journal + " " + asum + " " + esum
    emotions = emotion_analyzer(combined_text)
    
    scores = defaultdict(float)
    for chunk in emotions:
        for e in chunk:
            scores[e['label']] += e['score']
    avg_scores = [{'label': k, 'score': v / len(emotions)} for k, v in scores.items()]
    sorted_emotions.append(sorted(avg_scores, key=lambda x: x['score'], reverse=True))
    
    # Sentence-level emotion on full journal text
    sentence_emotion_details.append(sentence_level_emotions(journal))

end = time.time()
print("Time taken for processing:", round(end - start, 2), "seconds")

# Prepare results with emotion data included
results = []
for journal, ext_sum, abs_sum, cre_sum, emotions, sent_emotions in zip(
    journals, extractive_summaries, abstractive_summaries, creative_summaries, sorted_emotions, sentence_emotion_details
):
    results.append({
        "journal": journal.strip(),
        "extractive_summary": ext_sum.strip(),
        "extractive_word_count": word_count(ext_sum),
        "extractive_compression_ratio": compression_ratio(journal, ext_sum),
        "abstractive_summary": abs_sum.strip(),
        "abstractive_word_count": word_count(abs_sum),
        "abstractive_compression_ratio": compression_ratio(journal, abs_sum),
        "creative_abstractive_summary": cre_sum.strip(),
        "creative_word_count": word_count(cre_sum),
        "creative_compression_ratio": compression_ratio(journal, cre_sum),
        "extractive_emotions": emotions,
        "sentence_level_emotions": sent_emotions
    })

# Save to JSON
output_path = os.path.join(results_folder, "results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f"Results saved to {output_path}")

# Plot overall emotion charts (combined text)
for i, emotions in enumerate(sorted_emotions):
    labels = [e['label'] for e in emotions]
    scores = [e['score'] for e in emotions]
    
    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores, color='skyblue')
    plt.xlabel('Emotions')
    plt.ylabel('Scores')
    plt.title(f'Emotion Analysis of Combined Text for Journal {i+1}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
