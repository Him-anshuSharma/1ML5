from transformers import pipeline
from matplotlib import pyplot as plt
import time
from collections import defaultdict
from constants import journals  # Your list of journal entries

# Load models
print("Loading models...")
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
)

start_time = time.time()

# Summarize all journals
summaries = [
    summarizer(journal, max_length=100, min_length=10, do_sample=True)[0]['summary_text']
    for journal in journals
]

# Analyze emotions for all journals (averaging across chunks)
sorted_emotions = []
for journal in summaries:
    emotions = emotion_analyzer(journal)
    scores = defaultdict(float)
    for chunk in emotions:
        for e in chunk:
            scores[e['label']] += e['score']
    avg_scores = [
        {'label': k, 'score': v / len(emotions)}
        for k, v in scores.items()
    ]
    sorted_emotions.append(sorted(avg_scores, key=lambda x: x['score'], reverse=True))

end_time = time.time()

# Print summaries and emotions
for i, journal in enumerate(journals):
    print(f"\nJournal {i+1}:")
    print(f"Original: {journal}")
    print(f"Summary: {summaries[i]}")
    print("Emotions:")
    for emotion in sorted_emotions[i]:
        print(f"  {emotion['label']}: {emotion['score']:.4f}")

print("\nTotal Inference Time:", round(end_time - start_time, 2), "seconds")

# Plot emotion charts for all journals
for i, emotions in enumerate(sorted_emotions):
    labels = [e['label'] for e in emotions]
    scores = [e['score'] for e in emotions]
    
    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores, color='skyblue')
    plt.xlabel('Emotions')
    plt.ylabel('Scores')
    plt.title(f'Emotion Analysis of Journal {i+1}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
