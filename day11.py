# %%
import time

start = time.time()

from transformers import pipeline

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)


class_names = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=class_names)

import numpy as np

def predict_proba(texts):
    results = []
    for text in texts:
        outputs = emotion_classifier(text, return_all_scores=True)[0]
        # Create an ordered list of scores for each class
        scores = [next((item['score'] for item in outputs if item['label'] == c), 0) for c in class_names]
        results.append(scores)
    return np.array(results)

from constants import journals

exp = explainer.explain_instance(
    journals[1], 
    predict_proba, 
    num_features=5,    # Show top 5 important words
    top_labels=2       # Explain top 2 emotions predicted
)


top_emotion = exp.available_labels()[0]
print(f"Top predicted emotion: {class_names[top_emotion]}")

# Print words and their importance weights (positive means trigger)
print(exp.as_list(label=top_emotion))

# %%
index = exp.available_labels()[0]
print(index)
print(class_names[index])
list = exp.as_list(label=index)
for word,weight in list:
    print(f"{word}: {weight:.2f}")

# %%
end = time.time()
print("Time taken for processing:", round(end - start, 2), "seconds")
