import spacy
import pytextrank
from constants import journals
from transformers import pipeline
import time
import os, json
from datetime import datetime
import sys

# Utility functions
def word_count(text):
    return len(text.split())

def compression_ratio(original, summary):
    return round(len(original.split()) / len(summary.split()), 2) if len(summary.split()) else None

# Create results folder
current_file = os.path.splitext(os.path.basename(__file__))[0]
results_folder = f"results/{current_file}"
os.makedirs(results_folder, exist_ok=True)

start = time.time()

# Abstractive summarizer (normal)
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Abstractive summarizer (creative with sampling)
creative_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

abstractive_summaries = [
    abstractive_summarizer(x, max_length=100, min_length=10, do_sample=False)[0]['summary_text']
    for x in journals
]

creative_summaries = [
    creative_summarizer(x, max_length=100, min_length=10, do_sample=True, top_k=50, top_p=0.95)[0]['summary_text']
    for x in journals
]

# Extractive summarizer
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("textrank")

docs = [nlp(journal) for journal in journals]
extractive_summaries = [
    " ".join([sent.text for sent in doc._.textrank.summary(limit_sentences=2)]) for doc in docs
]

end = time.time()
print("Time taken for processing:", round(end - start, 2), "seconds")

# Prepare results
results = []
for journal, ext_sum, abs_sum, cre_sum in zip(journals, extractive_summaries, abstractive_summaries, creative_summaries):
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
        "creative_compression_ratio": compression_ratio(journal, cre_sum)
    })

# ---------- Save to JSON ----------
current_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
output_dir = os.path.join("results", current_file)
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Results saved to {output_path}")
