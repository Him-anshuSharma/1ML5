import os
import json
import csv
import threading
import queue
import re
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------- CONFIG ----------------
TOTAL_JOURNALS = 65
RESULTS_FILE = "confusion_set_journals.csv"
CHECKPOINT_FILE = "progress1.json"
THREADS = 3  # number of API keys
# ----------------------------------------

def normalize_text(text):
    """Normalize quotes, ellipsis, dashes, etc."""
    text = text.replace("…", "...").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    return text

# Load API keys from .env
load_dotenv()
API_KEYS = [os.getenv(f"API_KEY_{i+1}") for i in range(THREADS)]
API_KEYS = [k for k in API_KEYS if k]
if len(API_KEYS) < THREADS:
    raise ValueError(f"Expected {THREADS} API keys, found {len(API_KEYS)}.")

# Thread-safe locks
write_lock = threading.Lock()
progress_lock = threading.Lock()

# Load checkpoint
if Path(CHECKPOINT_FILE).exists():
    with open(CHECKPOINT_FILE, "r") as f:
        try:
            completed = json.load(f)
        except Exception:
            completed = {}
else:
    completed = {}
completed.setdefault("done", 0)

# Load existing results
if Path(RESULTS_FILE).exists():
    with open(RESULTS_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing_results = list(reader)
else:
    existing_results = []

# Prompt template
PROMPT = """
Generate 1 detailed personal journal entry (400–500 words) where the emotional tone is subtle, mixed, or ambiguous.
Reflect everyday life moments that could contain traces of joy, sadness, nostalgia, anxiety, and calm all at once.
Return the result as valid JSON in the following format:

[
  {{
    "journal": "...full text...",
    "label": {{
      "anger": float,
      "disgust": float,
      "fear": float,
      "joy": float,
      "neutral": float,
      "sadness": float,
      "surprise": float
    }}
  }}
]
"""

# ---------------- JSON Handling ----------------
def extract_json(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else text

def clean_and_parse_json(raw_text):
    try:
        json_str = extract_json(raw_text)
        json_str = normalize_text(json_str)
        json_str = re.sub(r'(?<=\b"journal":\s")(.+?)(?=",\s*"label")', 
                          lambda m: m.group(0).replace('"', '\\"').replace('\n', '\\n'), 
                          json_str, flags=re.DOTALL)
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
        return json.loads(json_str)
    except Exception as e:
        print(f"[JSON ERROR] {e}\nRaw output:\n{raw_text}")
        return None

# ---------------- Worker ----------------
def worker(thread_id, api_key, task_queue):
    client = genai.Client(api_key=api_key)

    while True:
        try:
            idx = task_queue.get_nowait()
        except queue.Empty:
            break

        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=PROMPT,
                config=types.GenerateContentConfig(
                    temperature=0.9,
                    max_output_tokens=1200,
                )
            )
            raw_output = response.text.strip()
            entry = clean_and_parse_json(raw_output)

            if entry is None:
                print(f"[Thread-{thread_id}] JSON parsing failed, skipping {idx}.")
                continue

            with write_lock:
                with open(RESULTS_FILE, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["journal","anger","disgust","fear","joy","neutral","sadness","surprise"])
                    if f.tell() == 0:
                        writer.writeheader()
                    for item in entry:
                        row = {"journal": normalize_text(item["journal"]), **item["label"]}
                        writer.writerow(row)

                with progress_lock:
                    completed["done"] += 1
                    with open(CHECKPOINT_FILE, "w") as cf:
                        json.dump(completed, cf)

            print(f"[Thread-{thread_id}] Completed {idx} ({completed['done']}/{TOTAL_JOURNALS})")

        except Exception as e:
            print(f"[Thread-{thread_id}] Error on entry {idx}: {e}")
        finally:
            task_queue.task_done()

# ---------------- Main ----------------
def main():
    task_queue = queue.Queue()
    for i in range(completed["done"], TOTAL_JOURNALS):
        task_queue.put(i + 1)

    threads = []
    for i, key in enumerate(API_KEYS):
        t = threading.Thread(target=worker, args=(i+1, key, task_queue))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("✅ All 65 journals generated!")

if __name__ == "__main__":
    main()
