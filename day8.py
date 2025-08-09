from transformers import pipeline
from matplotlib import pyplot as plt
import time
from constants import journal

start_time = time.time()

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")



summary = summarizer(journal)

print("Summary:", summary[0])

emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

emotions = emotion_analyzer(journal)
sorted_emotions = sorted(emotions[0],key = lambda x: x['score'], reverse=True)
print("Emotions:",sorted_emotions)

end_time = time.time()
print("Time taken:", end_time - start_time)

labels = [emo['label'] for emo in sorted_emotions]
score = [emo['score'] for emo in sorted_emotions]

plt.bar(labels, score, color='blue')
plt.xlabel('Emotions')
plt.ylabel('Score')
plt.title('Emotion Analysis')
plt.show()

