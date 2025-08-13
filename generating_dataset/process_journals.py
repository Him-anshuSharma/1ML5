import os
import json
import re
import csv
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv(Path("../.env"))
key = os.getenv("GEMINI_API_KEY")

prompt = """
You are given a personal diary journal entry written in first person.
Your task is to:
Read the journal text carefully.
Generate a short, creative, and vivid summary of the day's events (1-2 sentences).
Analyze the emotions expressed in the journal and output numerical emotion scores between 0 and 1 for the following seven emotions:
anger, disgust, fear, joy, neutral, sadness, and surprise.
Scores should sum to less than or equal to 1.
Use high precision (up to 10 decimal places).
Output the result strictly in JSON format with the keys:
"journal" (string, original text without changes)
"summary" (string, your generated summary)
"emotion_scores" (object with keys "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise", each a float value)
Example JSON format:
{
    "journal": "...",
    "summary": "...",
    "emotion_scores": {
        "anger": 0.0024589167,
        "disgust": 0.0025963401,
        "fear": 0.0021084223,
        "joy": 0.9759507179,
        "neutral": 0.0080765523,
        "sadness": 0.0074607483,
        "surprise": 0.0013481779
    }
}

Do not include extra commentary, only return valid JSON.
"""

def clean_text(text):
    pattern = re.compile(r'^Here are .*? diary entries from .*?:\s*', re.IGNORECASE | re.DOTALL)
    text = re.sub(pattern, '', text)
    text = text.strip()
    text = re.sub(r'\n+', '\n', text)
    return text

def clean_output_entry(entry):
    entry['journal'] = clean_text(entry.get('journal', ''))
    entry['summary'] = clean_text(entry.get('summary', ''))
    emotions = entry.get('emotion_scores', {})
    cleaned_emotions = {}
    for k in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']:
        val = emotions.get(k, 0.0)
        try:
            val = round(float(val), 10)
        except Exception:
            val = 0.0
        cleaned_emotions[k] = val
    entry['emotion_scores'] = cleaned_emotions
    return entry

def extract_json_from_response(text):
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return text

def save_to_csv(results, filename):
    fieldnames = ['journal', 'summary', 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    print(f"Saving {len(results)} entries to CSV file '{filename}'...")
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in results:
            writer.writerow({
                'journal': entry['journal'],
                'summary': entry['summary'],
                'anger': entry['emotion_scores'].get('anger', 0.0),
                'disgust': entry['emotion_scores'].get('disgust', 0.0),
                'fear': entry['emotion_scores'].get('fear', 0.0),
                'joy': entry['emotion_scores'].get('joy', 0.0),
                'neutral': entry['emotion_scores'].get('neutral', 0.0),
                'sadness': entry['emotion_scores'].get('sadness', 0.0),
                'surprise': entry['emotion_scores'].get('surprise', 0.0),
            })
    print(f"CSV file '{filename}' saved successfully.")

def main():
    filename = "journals.json"
    print(f"Loading diary entries from '{filename}'...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} diary entries.")
    except Exception as e:
        print(f"Failed to load '{filename}': {e}")
        return

    print("Cleaning diary entries...")
    cleaned_entries = [clean_text(entry) for entry in data]
    print("Cleaning complete.\n")

    client = genai.Client(api_key=key)
    results = []

    print("Starting processing of entries with Gemini API...\n")
    for idx, entry in enumerate(cleaned_entries, 1):
        print(f"Processing entry {idx} of {len(cleaned_entries)}...")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, entry],
            )
            clean_json_str = extract_json_from_response(response.text)
            output = json.loads(clean_json_str)
            output = clean_output_entry(output)
            results.append(output)
            print(f"Entry {idx} processed successfully.")
        except Exception as e:
            print(f"Error processing entry {idx}: {e}")

        print("Sleeping for 1.5 seconds to avoid rate limit...")
        time.sleep(1.5)

    if results:
        csv_filename = "diary_analysis.csv"
        save_to_csv(results, csv_filename)
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
