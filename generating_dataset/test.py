import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Allow importing constants.py from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from constants import journals  # List of journal texts

journals = [
    """Today started off pretty stressful. I overslept and missed my morning meeting, which made me anxious. 
At lunch, a friend surprised me with a small gift, and that made me smile and feel appreciated. 
Later, I received some criticism on a project I worked hard on, which frustrated me a bit. 
By the evening, I went for a walk in the park, enjoyed the fresh air, and felt calm and reflective. 
Overall, the day had a mix of anxiety, frustration, joy, and calm moments.
"""
]

# journals = [
#     "Today was amazing! I woke up early, went for a run, and felt full of energy. "
#     "Work went smoothly, and I even got a compliment from my boss. In the evening, "
#     "I met friends and laughed so much that my cheeks hurt. I feel happy and grateful "
#     "for everything today.",
    
#     "I can’t believe what happened today. Someone cut in front of me in traffic, and "
#     "I lost an important file at work because of a colleague’s mistake. Everything "
#     "felt unfair, and I just wanted to scream. I’m so frustrated and annoyed.",
    
#     "I got some bad news today. A friend I trusted betrayed me, and it made me question "
#     "a lot of things. I felt alone and down for most of the day. Even watching a movie "
#     "didn’t cheer me up. I just wanted to stay in bed and think.",
    
#     "I had an important presentation today. I was nervous and anxious beforehand, "
#     "but it went better than expected. Some parts were stressful, yet my colleagues "
#     "seemed happy with my work. I feel relieved but also a bit drained.",
    
#     "I walked into the office, and everyone shouted 'surprise!' for my birthday. "
#     "I didn’t expect it at all. I was shocked but really happy, and the decorations "
#     "and gifts made me smile. What a day!",
    
#     "I woke up, had breakfast, went to work, attended meetings, came back home, "
#     "and watched TV. Nothing unusual happened today, just a normal day with routine tasks."
# ]

# Expected emotions per journal (all 7 emotions)
# 1 if the emotion is present, 0 if absent
expected_labels = [
    {"anger":0, "disgust":0, "fear":0, "joy":1, "neutral":0, "sadness":0, "surprise":0},   # Journal 1
    {"anger":1, "disgust":0, "fear":0, "joy":0, "neutral":0, "sadness":0, "surprise":0},   # Journal 2
    {"anger":0, "disgust":0, "fear":0, "joy":0, "neutral":0, "sadness":1, "surprise":0},   # Journal 3
    {"anger":0, "disgust":0, "fear":0, "joy":1, "neutral":0, "sadness":0, "surprise":0},   # Journal 4
    {"anger":0, "disgust":0, "fear":0, "joy":1, "neutral":0, "sadness":0, "surprise":1},   # Journal 5 (joy + surprise)
    {"anger":0, "disgust":0, "fear":0, "joy":0, "neutral":1, "sadness":0, "surprise":0}    # Journal 6
]

# Paths to models
MODEL_PATHS = [
    "./finetuned_emotion",
    "./finetuned_emotion_subtle",
    # "./finetuned_emotion_subtle_mix"
]
MODEL_NAMES = ["M1", "M2"]

# Load models
pipelines_list = []
for path in MODEL_PATHS:
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    emotion_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True
    )
    pipelines_list.append(emotion_pipeline)

THRESHOLD = 0.5  # consider any emotion >= threshold as predicted

# Initialize counters for accuracy per model
correct_counts = {name: 0 for name in MODEL_NAMES}
total_labels = len(expected_labels) * 7  # 7 emotions per journal

# Evaluate
for i, journal in enumerate(journals, start=1):
    print(f"\n--- Journal {i} ---")
    print(journal)

    all_results = []
    for pipe in pipelines_list:
        results = pipe(journal)[0]
        score_dict = {r['label']: r['score'] for r in results}
        all_results.append(score_dict)

    # Print emotion probabilities
    labels = ["anger","disgust","fear","joy","neutral","sadness","surprise"]
    print("\nLabel\t" + "\t".join(MODEL_NAMES))
    for label in labels:
        scores = [f"{r.get(label,0):.4f}" for r in all_results]
        print(f"{label}\t" + "\t".join(scores))

    # Check which emotions were correctly predicted
    # for idx, scores_dict in enumerate(all_results):
    #     predicted_emotions = {label: 1 if scores_dict.get(label,0) >= THRESHOLD else 0 for label in labels}
        # # Count correct labels
        # correct_counts[MODEL_NAMES[idx]] += sum(
        #     1 for label in labels if predicted_emotions[label] == expected_labels[i-1][label]
        # )
        #print(f"{MODEL_NAMES[idx]} predicted: {predicted_emotions}, expected: {expected_labels[i-1]}")

# Compute overall accuracy
# print("\n=== Overall Accuracy (All 7 Emotions) ===")
# for name in MODEL_NAMES:
#     accuracy = correct_counts[name] / total_labels
#     print(f"{name}: {accuracy:.2f}")
