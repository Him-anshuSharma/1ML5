import os                                                                                                                                                                                                          
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv(Path("../.env"))
key = os.getenv("GEMINI_API_KEY")
# print(key)

# Import necessary modules
from google import genai
from google.genai import types
import re

client = genai.Client(api_key=key)

journals = []

prompts = [
    # Age 15-16 – Teen friendship & early crush
    "Generate 5 separate first-person diary entries (400–500 words each) from a 15-year-old experiencing intense emotions of friendship, early crushes, and self-discovery in school life. Each entry should mix joy with nervous excitement and awkwardness. Include vivid details of school hallways, sports fields, or teenage hangouts. Separate entries with =====END OF JOURNAL=====",

    # Age 16-17 – High school love & heartbreak
    "Generate 5 separate first-person diary entries (400–500 words each) from a 16-year-old navigating first love, heartbreak, and emotional highs and lows. Include moments like texting late at night, arguments, and moments of connection. Blend love, fear, and vulnerability. Separate entries with =====END OF JOURNAL=====",

    # Age 17-18 – Graduation & uncertainty
    "Generate 5 separate first-person diary entries (400–500 words each) from an 18-year-old facing high school graduation, balancing excitement for the future and sadness over leaving friends behind. Include scenes like farewell parties, signing yearbooks, and last walks around school. Separate entries with =====END OF JOURNAL=====",

    # Age 18-19 – Moving to college
    "Generate 5 separate first-person diary entries (400–500 words each) from a 19-year-old moving into their first college dorm, mixing independence with homesickness. Include campus life, meeting roommates, and adjusting to a new city. Separate entries with =====END OF JOURNAL=====",

    # Age 19-20 – First serious relationship
    "Generate 5 separate first-person diary entries (400–500 words each) from a 20-year-old in their first serious relationship, balancing romance with personal growth. Mix joy, deep connection, and occasional misunderstandings. Include dates, quiet moments, and long conversations. Separate entries with =====END OF JOURNAL=====",

    # Age 20-21 – Part-time job struggles
    "Generate 5 separate first-person diary entries (400–500 words each) from a 21-year-old balancing college and a part-time job, facing exhaustion, career doubts, and moments of pride. Include workplace scenes, late-night studying, and supportive friendships. Separate entries with =====END OF JOURNAL=====",

    # Age 21-22 – Internship & career excitement
    "Generate 5 separate first-person diary entries (400–500 words each) from a 22-year-old starting their first internship, feeling excitement and anxiety about their career path. Include office dynamics, imposter syndrome, and small wins. Separate entries with =====END OF JOURNAL=====",

    # Age 22-23 – Long-distance relationship
    "Generate 5 separate first-person diary entries (400–500 words each) from a 23-year-old in a long-distance relationship, mixing longing, joy in reunions, and sadness in goodbyes. Include video calls, letters, and travel moments. Separate entries with =====END OF JOURNAL=====",

    # Age 23-24 – First apartment
    "Generate 5 separate first-person diary entries (400–500 words each) from a 24-year-old moving into their first apartment, mixing freedom, financial stress, and pride in independence. Include decorating, cooking experiments, and late-night thoughts. Separate entries with =====END OF JOURNAL=====",

    # Age 24-25 – Career growth & pressure
    "Generate 5 separate first-person diary entries (400–500 words each) from a 25-year-old facing rapid career growth, juggling ambition, stress, and moments of joy in achievements. Include workplace wins, late nights at the office, and personal sacrifices. Separate entries with =====END OF JOURNAL=====",

    # Age 25-26 – Travel adventures
    "Generate 5 separate first-person diary entries (400–500 words each) from a 26-year-old traveling abroad, mixing joy, cultural surprises, and occasional loneliness. Include markets, train rides, and unexpected friendships. Separate entries with =====END OF JOURNAL=====",

    # Age 26-27 – Engagement & life changes
    "Generate 5 separate first-person diary entries (400–500 words each) from a 27-year-old recently engaged, mixing love, planning stress, and hope for the future. Include proposal moments, family reactions, and dreams for married life. Separate entries with =====END OF JOURNAL=====",

    # Age 27-28 – Career setbacks
    "Generate 5 separate first-person diary entries (400–500 words each) from a 28-year-old dealing with a major career setback, mixing disappointment, resilience, and small victories. Include conversations with mentors, self-reflection, and moments of hope. Separate entries with =====END OF JOURNAL=====",

    # Age 28-29 – Health challenges
    "Generate 5 separate first-person diary entries (400–500 words each) from a 29-year-old recovering from a health scare, mixing fear, gratitude, and determination. Include hospital visits, supportive relationships, and personal realizations. Separate entries with =====END OF JOURNAL=====",

    # Age 29-30 – Wedding preparations
    "Generate 5 separate first-person diary entries (400–500 words each) from a 30-year-old preparing for their wedding, mixing joy, stress, and reflection on the journey so far. Include family interactions, dress fittings, and quiet emotional moments. Separate entries with =====END OF JOURNAL=====",

    # Age 15-18 – Teenage summer freedom
    "Generate 5 separate first-person diary entries (400–500 words each) from a teenager enjoying a summer of freedom, mixing joy, adventure, and small conflicts. Include bike rides, bonfires, and night swims. Separate entries with =====END OF JOURNAL=====",

    # Age 18-22 – University sports
    "Generate 5 separate first-person diary entries (400–500 words each) from a university athlete balancing competition pressure, friendships, and love life. Include training, victories, and emotional struggles. Separate entries with =====END OF JOURNAL=====",

    # Age 22-26 – Starting a business
    "Generate 5 separate first-person diary entries (400–500 words each) from a young entrepreneur starting a small business, mixing excitement, fear, and learning experiences. Include first sales, challenges, and late-night planning. Separate entries with =====END OF JOURNAL=====",

    # Age 26-30 – Major life reflection
    "Generate 5 separate first-person diary entries (400–500 words each) from someone at age 30 reflecting on their twenties, mixing nostalgia, pride, and unresolved feelings. Include key memories, regrets, and hopes for the next decade. Separate entries with =====END OF JOURNAL====="
]

count = 0
for prompt in prompts:
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
        ),
    )
    print(f"Processing prompt {count + 1}/{len(prompts)}")
    for journal in response.text.split("=====END OF JOURNAL====="):
        if journal.strip():  # Check if the journal entry is not empty
            journal = journal.strip()
            journal = re.sub(r"\*\*Entry\s+\d+:[^\*]+\*\*", "", journal)
            # Replace escaped \n with real newlines
            journal = journal.replace("\\n", "\n")
            # Remove extra spaces
            journal = re.sub(r'\n\s*\n', '\n\n', journal)
            journals.append(journal)
    count += 1
    print(f"Completed prompt {count}/{len(prompts)}")

#save to file
import json 
with open("journals.json", "w") as f:
    json.dump(journals, f,ensure_ascii=False, indent=4)
