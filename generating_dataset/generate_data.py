# generate_journals_parallel.py
import os
import json
import csv
import time
import random
import threading
import queue
from math import ceil
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ----------------- CONFIG -----------------
TOTAL_CALLS = 750
RESULTS_FILE = "emotion_journals.csv"
CHECKPOINT_FILE = "progress.json"
MAX_RETRIES = 2
API_TIMEOUT = 30  # seconds per API call
# ------------------------------------------

# Load .env from parent (same pattern as your original)
load_dotenv(Path("../.env"))

# Load API keys: either JSON array or comma-separated string
raw_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
try:
    API_KEYS = json.loads(raw_keys) if raw_keys.strip().startswith("[") else [k.strip() for k in raw_keys.split(",") if k.strip()]
except Exception:
    API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

if not API_KEYS:
    raise RuntimeError("No API keys found in .env (GEMINI_API_KEYS or GEMINI_API_KEY)")

# Create one genai client per key
clients = [genai.Client(api_key=k) for k in API_KEYS]

# ===== your random elements and base_prompts (copied) =====
times_of_day = ["early morning", "late morning", "afternoon", "evening", "night", "midnight", "sunrise", "sunset"]
weathers = ["sunny", "rainy", "stormy", "foggy", "windy", "snowy", "humid", "chilly"]
locations = [
    "school library", "city café", "beach", "mountain trail", "bus stop",
    "apartment balcony", "university dorm", "train station", "park bench", "rooftop garden"
]
names = ["Aarav", "Meera", "Kabir", "Priya", "Ishaan", "Ananya", "Rohan", "Simran", "Vikram", "Neha"]
twists = [
    "lost my keys", "missed the bus", "ran into an old friend",
    "spilled coffee", "forgot my wallet", "found a stray cat",
    "phone battery died", "unexpectedly got good news", "accidentally overheard a conversation"
]

emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
scenarios = [
    "a regular school/college day",
    "a special celebration",
    "a personal conflict",
    "a moment of self-reflection",
    "a challenging task or event",
    "time spent with friends/family",
    "an unexpected incident"
]

base_prompts = []
for emotion in emotions:
    for scenario in scenarios:
        base_prompts.append(f"Write a first-person diary entry expressing strong {emotion} during {scenario}.")
# =========================================================

def inject_randomness(base_prompt):
    tod = random.choice(times_of_day)
    weather = random.choice(weathers)
    location = random.choice(locations)
    name = random.choice(names)
    twist = random.choice(twists)
    return (
        f"{base_prompt} "
        f"Set it in the {tod} on a {weather} day at a {location}. "
        f"Include a character named {name}. "
        f"Something unexpected happens: {twist}. "
        "The diary entry should be 400–500 words, rich in sensory and emotional detail. "
        "After writing the diary, output a JSON object with:\n"
        "journal (string), summary (string), emotion_scores (object with anger, disgust, fear, joy, neutral, sadness, surprise as float values, sum ≤ 1). "
        "Do not include commentary, only valid JSON."
    )

def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end+1]
    raise ValueError("No JSON object found in text")

# ---------- checkpoint helpers ----------
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return set(data.get("completed_runs", []))
            except Exception:
                return set()
    return set()

def save_checkpoint_set(completed_runs_set):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"completed_runs": sorted(list(completed_runs_set))}, f, indent=2)

# ---------- prepare tasks ----------
# Determine how many variations per base_prompt are needed
variations = ceil(TOTAL_CALLS / len(base_prompts))
tasks = []  # each task: dict {run_id, prompt_idx, i, base_prompt, retries}
for prompt_idx, bp in enumerate(base_prompts):
    for i in range(variations):
        run_id = f"{prompt_idx}-{i}"
        tasks.append({"run_id": run_id, "prompt_idx": prompt_idx, "i": i, "base_prompt": bp, "retries": 0})
# crop to TOTAL_CALLS
tasks = tasks[:TOTAL_CALLS]

# Map run_id -> sequence number (1..TOTAL_CALLS) for printing
run_order_map = {t["run_id"]: (idx + 1) for idx, t in enumerate(tasks)}

# Load checkpoint and filter tasks
completed_runs = load_checkpoint()
pending_tasks = [t for t in tasks if t["run_id"] not in completed_runs]

# Thread-safe queues
task_queue = queue.Queue()
write_queue = queue.Queue()

for t in pending_tasks:
    task_queue.put(t)

# Ensure CSV header exists
FIELDNAMES = ['journal', 'summary', 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

# ---------- timeout helper ----------
def call_with_timeout(func, args=(), kwargs={}, timeout=API_TIMEOUT):
    result = [None]
    exc = [None]
    def target():
        try:
            result[0] = func(*args, **(kwargs or {}))
        except Exception as e:
            exc[0] = e
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None, "timeout"
    if exc[0]:
        return None, exc[0]
    return result[0], None

# ---------- worker ----------
shutdown_event = threading.Event()
RESOURCE_EXHAUSTED_FLAG = threading.Event()
lock_completed = threading.Lock()  # used by writer to update completed_runs set safely

def worker_thread(client: genai.Client, worker_idx: int):
    while not shutdown_event.is_set():
        try:
            task = task_queue.get(timeout=3)
        except queue.Empty:
            return
        run_id = task["run_id"]
        seq_no = run_order_map.get(run_id, None)
        base_prompt = task["base_prompt"]
        final_prompt = inject_randomness(base_prompt)

        # Print prompt with call number
        if seq_no is not None:
            print(f"\n[Call {seq_no}/{TOTAL_CALLS}] Prompt:\n{final_prompt}\n")
        else:
            print(f"\n[Call ?/{TOTAL_CALLS}] Prompt:\n{final_prompt}\n")

        def gen_call():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    temperature=random.uniform(0.7, 1.0),
                    top_p=random.uniform(0.8, 1.0)
                )
            )

        response, err = call_with_timeout(gen_call, timeout=API_TIMEOUT)
        if err == "timeout":
            print(f"[worker {worker_idx}] Timeout for run {run_id}. retry {task['retries']}/{MAX_RETRIES}")
            task["retries"] += 1
            if task["retries"] <= MAX_RETRIES:
                task_queue.put(task)
            else:
                print(f"[worker {worker_idx}] Giving up on {run_id} after retries")
            task_queue.task_done()
            continue
        if isinstance(err, Exception):
            err_str = str(err)
            if "RESOURCE_EXHAUSTED" in err_str:
                print(f"[worker {worker_idx}] RESOURCE_EXHAUSTED encountered. Saving progress and exiting.")
                RESOURCE_EXHAUSTED_FLAG.set()
                # put the task back so it can be resumed later
                task_queue.put(task)
                task_queue.task_done()
                shutdown_event.set()
                return
            else:
                print(f"[worker {worker_idx}] Error for run {run_id}: {err_str}. retry {task['retries']}/{MAX_RETRIES}")
                task["retries"] += 1
                if task["retries"] <= MAX_RETRIES:
                    task_queue.put(task)
                else:
                    print(f"[worker {worker_idx}] Skipping {run_id} after retries")
                task_queue.task_done()
                continue

        # success: parse response
        try:
            raw_text = response.text.replace("\\n", "").strip()
            json_str = extract_json(raw_text)
            entry = json.loads(json_str)
            # Basic validation: must have journal, summary and emotion_scores
            if ('journal' not in entry) or ('summary' not in entry) or ('emotion_scores' not in entry):
                raise ValueError("Missing keys in entry JSON")
            # Ensure all emotion keys present (fill zeros if missing)
            for k in FIELDNAMES[2:]:
                entry['emotion_scores'].setdefault(k, 0.0)
            # Push to write queue as tuple (run_id, entry, raw_text)
            write_queue.put((run_id, entry, raw_text))
            print(f"[worker {worker_idx}] Enqueued {run_id} for write.")
        except Exception as e:
            print(f"[worker {worker_idx}] JSON parse/validation error for {run_id}: {e}")
            task["retries"] += 1
            if task["retries"] <= MAX_RETRIES:
                task_queue.put(task)
            else:
                print(f"[worker {worker_idx}] Skipping {run_id} after parse failures")
        finally:
            task_queue.task_done()
        time.sleep(random.uniform(1.0, 2.5))  # small throttle to be polite

# ---------- writer ----------
def writer_thread():
    local_completed = None
    # We'll keep a single in-memory set mirror of completed_runs to avoid repeated file IO
    with lock_completed:
        local_completed = load_checkpoint()  # set
    while True:
        try:
            item = write_queue.get(timeout=5)
        except queue.Empty:
            # If no more workers and queue empty, exit
            if task_queue.empty() and all(not t.is_alive() for t in worker_threads):
                break
            continue
        if item is None:
            write_queue.task_done()
            break
        run_id, entry, raw_text = item
        seq_no = run_order_map.get(run_id, None)
        # Build row dict
        row = {
            'journal': entry.get('journal', ''),
            'summary': entry.get('summary', '')
        }
        for k in FIELDNAMES[2:]:
            row[k] = float(entry['emotion_scores'].get(k, 0.0))
        # Write row safely
        try:
            with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
            # Update checkpoint (atomically in-memory then disk)
            with lock_completed:
                local_completed.add(run_id)
                save_checkpoint_set(local_completed)
            # Print the response after safe write
            if seq_no is not None:
                print(f"[Call {seq_no}/{TOTAL_CALLS}] Response:\n{raw_text}\n")
            else:
                print(f"[Call ?/{TOTAL_CALLS}] Response:\n{raw_text}\n")
            print(f"[writer] Wrote and checkpointed {run_id}")
        except Exception as e:
            print(f"[writer] Failed to write {run_id}: {e}")
            # Requeue item for attempt later (avoid infinite loop)
            write_queue.put(item)
            time.sleep(1)
        finally:
            write_queue.task_done()

# ---------- start threads ----------
worker_threads = []
for idx, client in enumerate(clients):
    t = threading.Thread(target=worker_thread, args=(client, idx), daemon=True)
    worker_threads.append(t)
    t.start()

writer = threading.Thread(target=writer_thread, daemon=True)
writer.start()

# Wait for all tasks processed or a RESOURCE_EXHAUSTED flag
try:
    # block until queue empty and workers done
    task_queue.join()
    # After all tasks processed, wait for write queue to finish
    write_queue.join()
except KeyboardInterrupt:
    print("KeyboardInterrupt: shutting down gracefully.")
    shutdown_event.set()

# If resource exhausted, we set flag and exit gracefully (checkpoint already saved by writer)
if RESOURCE_EXHAUSTED_FLAG.is_set():
    print("Exiting due to RESOURCE_EXHAUSTED. Resume later.")
else:
    print("All tasks processed. Finalizing...")

# final cleanup: make sure writer finished
writer.join(timeout=5)

# final checkpoint save (redundant but safe)
with lock_completed:
    current_done = load_checkpoint()
    save_checkpoint_set(current_done)

print("Done. Results saved in:", RESULTS_FILE)
