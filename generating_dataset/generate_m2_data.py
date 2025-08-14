# generate_m2_journals_balanced.py
import os, json, csv, time, random, threading, queue
from math import ceil
from pathlib import Path
from dotenv import load_dotenv
import requests
from google import genai
from google.genai import types
from groq import Groq
from huggingface_hub import InferenceClient
from openai import OpenAI
import re
from time import sleep

# for i in range(1,200):
#     print(i)
#     sleep(1)

# ---------- CONFIG ----------
TOTAL_CALLS = 2000
RESULTS_FILE = "m2_emotion_journals.csv"
CHECKPOINT_FILE = "m2_progress.json"
MAX_RETRIES = 2
API_TIMEOUT = 120
MIN_WORDS = 300
MAX_WORDS = 500
EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

# ---------- Load .env ----------
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

def load_keys(env_var):
    raw = os.getenv(env_var, "")
    try:
        return json.loads(raw) if raw.strip().startswith("[") else [k.strip() for k in raw.split(",") if k.strip()]
    except:
        return [k.strip() for k in raw.split(",") if k.strip()]

PROVIDERS = {
    "Gemini": load_keys("GEMINI_API_KEYS"),
    # "HuggingFace": load_keys("HF_API_KEYS"),
    "OpenRouter": load_keys("OPENROUTER_API_KEYS"),
    "Groq": load_keys("GROQ_API_KEYS"),
    # "ShaleProtocol": load_keys("SHALE_API_KEYS"),
}

# ---------- Build provider-key pairs ----------
provider_key_pairs = []
for p, keys in PROVIDERS.items():
    for k in keys:
        provider_key_pairs.append((p, k))
if not provider_key_pairs:
    raise RuntimeError("No API keys found!")

# ---------- Base prompts ----------
times_of_day = ["early morning","late morning","afternoon","evening","night","midnight"]
weathers = ["sunny","rainy","stormy","foggy","windy","snowy"]
locations = ["school library","city café","beach","mountain trail","bus stop",
             "apartment balcony","train station","park bench","rooftop garden"]
names = ["Aarav","Meera","Kabir","Priya","Ishaan","Ananya","Rohan","Simran","Vikram","Neha"]
twists = ["lost my keys","missed the bus","ran into an old friend","spilled coffee",
          "forgot my wallet","found a stray cat","phone battery died",
          "unexpectedly got good news","overheard a conversation"]

def make_prompt(emotion):
    base_prompt = (
        f"Write a first-person diary entry expressing {emotion}. "
        "Include subtle mixed emotions, and cases where some emotions are absent. "
    )
    tod = random.choice(times_of_day)
    weather = random.choice(weathers)
    location = random.choice(locations)
    name = random.choice(names)
    twist = random.choice(twists)
    words = random.randint(MIN_WORDS, MAX_WORDS)

    prompt = (
        f"{base_prompt} Set it in the {tod} on a {weather} day at a {location}. "
        f"Include a character named {name}. Something unexpected happens: {twist}. "
        f"Entry should be around {words} words. \n\n"
        "IMPORTANT: Return ONLY valid JSON in the following format:\n"
        "{\n"
        '  "journal": "<diary text>",\n'
        '  "summary": "<brief summary>",\n'
        '  "emotion_scores": {\n'
        '    "anger": <float>, "disgust": <float>, "fear": <float>, '
        '"joy": <float>, "neutral": <float>, "sadness": <float>, "surprise": <float>\n'
        "  }\n"
        "}\n"
        "Do NOT include any commentary or text outside this JSON. "
        "Ensure the JSON parses correctly."
    )
    return prompt

# ---------- Resume from existing CSV ----------
FIELDNAMES = ['run_id','journal','summary','anger','disgust','fear','joy','neutral','sadness','surprise']
processed_count = 0
completed_runs = set()
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed_runs.add(row.get("run_id", row.get("journal","")))
    processed_count = len(completed_runs)
    print(f"🟢 Resuming. Already processed {processed_count} journals.")
else:
    with open(RESULTS_FILE,"w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()

# ---------- Prepare tasks ----------
task_queue = queue.Queue()
write_queue = queue.Queue()
tasks = []
per_emotion = TOTAL_CALLS // len(EMOTIONS)
for idx, emotion in enumerate(EMOTIONS):
    for i in range(per_emotion):
        run_id = f"{emotion}-{i}"
        if run_id in completed_runs:
            continue
        tasks.append({"run_id": run_id, "emotion": emotion, "retries": 0})

random.shuffle(tasks)
for t in tasks:
    task_queue.put(t)

shutdown_event = threading.Event()
lock_completed = threading.Lock()

API_TIMEOUT = 30
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# ---------- Provider call ----------
def safe_parse_json(text):
    """
    Safely extract and parse JSON from API responses.
    Escapes literal newlines, tabs, carriage returns inside string values.
    """
    # Extract JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    raw_json = match.group()

    # Escape literal control characters inside string values
    def escape_string_literals(m):
        s = m.group(0)
        s = s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return s

    # Apply escaping to all string literals in JSON
    escaped_json = re.sub(r'"(.*?)"', escape_string_literals, raw_json, flags=re.DOTALL)

    try:
        return json.loads(escaped_json)
    except json.JSONDecodeError:
        # Fallback: decode unicode escapes
        safe_json = escaped_json.encode('utf-8').decode('unicode_escape')
        return json.loads(safe_json)


def call_provider(provider, key, prompt, callback=None, retries=0, max_retries=3):
    """
    Calls a provider safely. If rate-limited, schedules a retry without blocking.
    `callback` is called with the result when done.
    """
    try:
        text = ""

        # ---------------- Provider Calls ----------------
        if provider == "Gemini":
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=random.uniform(0.7, 1.0),
                    top_p=random.uniform(0.8, 1.0)
                )
            )
            text = response.text.strip()

        elif provider == "HuggingFace":
            client = InferenceClient(api_key=key)
            completion = client.chat.completions.create(
                model="Qwen/Qwen2-7B-Instruct",
                messages=[{"role": "user", "content": prompt}]
            )
            text = completion.choices[0].message.content

        elif provider == "OpenRouter":
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
            completion = client.chat.completions.create(
                extra_headers={"HTTP-Referer": "https://your-site.com", "X-Title": "MyApp"},
                model="openai/gpt-oss-20b:free",
                messages=[{"role": "user", "content": prompt}],
            )
            text = completion.choices[0].message.content

        elif provider == "Groq":
            client = Groq(api_key=key)
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            text = completion.choices[0].message.content

        else:
            raise ValueError(f"Unknown provider: {provider}")

        # ---------------- Safe JSON Parsing ----------------
        entry = safe_parse_json(text)
        if not entry:
            print(f"❌ {provider} returned invalid JSON. Raw response:\n{text}...")
            if callback:
                callback(None)
            return None

        # ---------------- Ensure Data Consistency ----------------
        scores = entry.get("emotion_scores", {})
        entry["emotion_scores"] = {e: float(scores.get(e, 0.0)) for e in EMOTIONS}
        entry.setdefault("journal", "")
        entry.setdefault("summary", "")

        # Skip empty or zero-score responses
        if not entry["journal"].strip() or all(v == 0.0 for v in entry["emotion_scores"].values()):
            print(f"⚠️ {provider} returned empty journal or zero scores. Raw response:\n{text}...")
            if callback:
                callback(None)
            return None

        print(f"✅ {provider} returned journal with {len(entry['journal'].split())} words")
        if callback:
            callback(entry)
        return entry

    except Exception as ex:
        print(f"❌ Error calling {provider}: {ex}\nRaw response (if any):\n{text}")

        # ---------------- Cool-off Logic ----------------
        if "rate limit" in str(ex).lower() or "limit reached" in str(ex).lower():
            if retries >= max_retries:
                print(f"⚠️ {provider} failed after {max_retries} retries.")
                if callback:
                    callback(None)
                return None

            wait_time = 600 if provider == "Groq" else 60 if provider == "Gemini" else 30
            print(f"⏱ {provider} rate limit reached. Scheduling retry in {wait_time//60} minutes...")

            # Schedule retry without blocking other threads
            threading.Timer(wait_time, call_provider, args=(provider, key, prompt, callback, retries + 1, max_retries)).start()
        else:
            if callback:
                callback(None)
        return None

# ---------- Worker thread ----------
provider_index = 0
provider_lock = threading.Lock()

def worker_thread(worker_idx):
    global provider_index
    while not shutdown_event.is_set():
        try:
            task = task_queue.get(timeout=3)
        except queue.Empty:
            return
        run_id = task["run_id"]
        final_prompt = make_prompt(task["emotion"])

        with provider_lock:
            provider, key = provider_key_pairs[provider_index % len(provider_key_pairs)]
            provider_index += 1

        response = call_provider(provider, key, final_prompt)

        if response is None:
            if task["retries"] < MAX_RETRIES:
                task["retries"] += 1
                task_queue.put(task)
            task_queue.task_done()
            continue

        response["run_id"] = run_id
        write_queue.put((run_id, response))
        task_queue.task_done()
        time.sleep(random.uniform(0.3,1.0))

# ---------- Writer thread ----------
processed_lock = threading.Lock()

def writer_thread():
    global processed_count
    while True:
        try:
            item = write_queue.get(timeout=5)
        except queue.Empty:
            if task_queue.empty() and all(not t.is_alive() for t in worker_threads):
                break
            continue

        run_id, entry = item
        row = {"run_id": entry.get("run_id",""),
               "journal": entry.get("journal",""),
               "summary": entry.get("summary","")}
        for k in ['anger','disgust','fear','joy','neutral','sadness','surprise']:
            row[k] = float(entry["emotion_scores"].get(k,0.0))

        with lock_completed:
            with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
                writer.writerow(row)

            with processed_lock:
                processed_count += 1
                print(f"📝 Journal {processed_count} processed: {entry['journal'][:60]}...")
                print(f"   Emotion scores: {entry['emotion_scores']}")

        write_queue.task_done()

# ---------- Start ----------
num_workers = len(provider_key_pairs)
worker_threads = []
for i in range(num_workers):
    t = threading.Thread(target=worker_thread,args=(i,),daemon=True)
    worker_threads.append(t)
    t.start()

writer = threading.Thread(target=writer_thread,daemon=True)
writer.start()

task_queue.join()
write_queue.join()
writer.join(timeout=5)
print("✅ Balanced M2 dataset generation complete. Saved in:", RESULTS_FILE)
