import time
import spacy
import pytextrank
import numpy as np
import matplotlib.pyplot as plt
import json
from transformers import pipeline, AutoTokenizer
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from constants import journals

journals = ['''I’ve been trying to figure out all evening what exactly I’m feeling, but it’s like my emotions are stuck in a blender. Nothing is making complete sense.
The day started normally — she sent me a “good morning” text at 8:15, just like she usually does. I should have felt good about it, but something in me immediately noticed it didn’t have the little heart emoji she usually adds. It’s such a small thing, but my brain instantly latched onto it like it meant something. Maybe she was just rushing to get ready. Or maybe… I don’t even know.
All through work, I kept checking my phone, half-hoping for a random text from her, half-dreading that there would be nothing. She did reply when I messaged, but her responses were short — one or two words, no follow-up questions. I kept telling myself she’s probably busy, but at the same time I was scanning every word she sent, looking for some hidden tone or clue. It’s exhausting when even normal messages feel like puzzles I’m trying to solve.

We had already planned to talk in the evening, around 9. I had this hope that hearing her voice would make all this doubt disappear. The call started fine, but there was this… background distance. Like she was talking to me, but not with me. Sometimes she’d laugh, but it felt like her mind was somewhere else. I tried to tell her about something funny that happened at work, and she smiled, but her reaction was delayed, like she wasn’t fully listening.

Halfway through the call, she said she was tired. I asked if she was okay, and she said “Yeah, just a long day,” but it was one of those “yeah” responses that feels like it’s closing the door instead of opening it. I didn’t push, but my mind wouldn’t stop imagining scenarios — maybe she’s upset with me and doesn’t want to say it. Maybe she’s losing interest. Or maybe I’m just projecting my own insecurities onto her.

But then, just before we hung up, her tone softened. She said, “Goodnight, take care,” in that voice she only uses when she genuinely cares. And that one moment threw me off completely. How can she feel distant and close at the same time? How can I feel like I’m losing her but also believe she still cares?

Now I’m lying here in bed, replaying the whole day like a movie on repeat, pausing at every moment to analyze her words and my own reactions. I hate feeling like this — like my love for her is tangled up with doubt and fear. I wish I could just trust what she says and not overthink everything. I wish I could turn off this constant need to measure how close or far away she feels.

Maybe tomorrow will feel different. Or maybe I’ll wake up with the same questions, pretending I don’t have them, just to keep the peace.''']

# ====== Emotion Classifiers ======
def get_emotion_classifier(option="default"):
    """Load the appropriate emotion classification pipeline."""
    
    if option == "default":
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        class_names = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    elif option == "goemotions":
        classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
        tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
        class_names = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
                       "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude",
                       "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse",
                       "sadness", "surprise", "neutral"]
    elif option == "custom":
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        class_names = ["anger", "joy", "sadness"]
    else:
        raise ValueError("Invalid classifier option")
    return classifier, class_names, tokenizer


def classify_emotions(journals, emotion_classifier, class_names, tokenizer):
    """Run emotion classification on all journals."""
    all_scores = []
    for text in journals:
        # Encode once to get tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_chunks = 0
        scores = [0.0] * len(class_names)

        for i in range(0, len(tokens), 512):
            total_chunks += 1
            chunk_tokens = tokens[i:i+512]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            results = emotion_classifier(chunk_text, truncation=True, max_length=512)[0]

            for r in results:
                if r['label'] in class_names:
                    scores[class_names.index(r['label'])] += r['score']

        # Avoid divide-by-zero for short texts
        if total_chunks > 0:
            scores = [score / total_chunks for score in scores]

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
        result = summarizer(text, max_length=1000, min_length=10, do_sample=False)
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
    emotion_classifier, class_names, tokenizer = get_emotion_classifier(option=classifier_type)
    emotion_scores = classify_emotions(journals, emotion_classifier, class_names,tokenizer)

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
    run_pipeline("default", 2)
    run_pipeline("default", 3)
    run_pipeline("goemotions", 1)
    run_pipeline("goemotions", 2)
    run_pipeline("goemotions", 3)
    run_pipeline("custom", 1)
    run_pipeline("custom", 2)
    run_pipeline("custom", 3)