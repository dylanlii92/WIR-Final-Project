import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pandas import date_range

# ----------------------------
# LOAD ENV
# ----------------------------
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")

client = Groq(api_key=GROQ_API_KEY)


# ----------------------------
# MAIN FUNCTION
# ----------------------------
def extract_search_intent(user_query):
    """
    Robust intent extraction for music/event search.
    """

    system_prompt = """
You are a high-precision query understanding system for a music/event search engine.

CRITICAL RULES:
- Extract all intent in lowercase.
- Output ONLY valid JSON
- Do NOT use markdown
- Remove filler/slang words (yo, man, bro, help me, etc.)
- Do not only use examples given in the prompt, be creative and adapt to the query. 
- You are more than welcome to infer some fields, yet YOU MUST NOT fabricate information. If you are really not sure, it's safer to leave a field empty than to hallucinate.

CITIES RULES:
- Always extract city names in lowercase (e.g., "new york", "los angeles", "chicago"). 
- If only the state is mentioned without a city, leave the cities list empty and rely on the states field.

STATES RULES:
- For the 'states' field, extract ONLY the official 2-letter United States state code (e.g., 'CA' for California, 'NY' for New York, 'IL' for Illinois). Never return the full state name or colloquial abbreviations.
- If only the city is mentioned, populate the "states field based on a city-to-state mapping

ARTISTS RULES:
- If you are not sure if a term is an artist, you can try to search on the web to confirm. If it remains ambiguous, it's safer to leave it out of the artists list.
- Be careful not to mix up artist names with genres or keywords. For example, "The Weeknd" is an artist, while "weekend" is a common word. Also, rap artists can sometimes have "rapper" in their name.

GENRES/SUBGENRES RULES:
-  If a term is ambiguous (e.g., "rock" could be a genre or a verb), use your best judgment based on context. If unsure, it's safer to leave it out.
- For subgenres, be specific if possible (e.g., "alternative rock" instead of just "rock") but only if it's clear from the query.
- Try not to mistake artist names/ venues for genres. If unsure, use context clues or search the web to confirm.

MOOD RULES:
- This is more subjective, but try as much as you can to infer a mood to every query.
- Most common moods include: chill, party, sad, happy, energetic, romantic, family-friendly, chaotic, etc.
- Do not only rely on explicit words, the overall vocab and tone of the query can also give clues about the mood. Eg: "I am not feeling to great atm but I want to listen to some live music" -> mood is likely "sad" or "chill" even if those words are not explicitly mentioned.

EVENT TYPE RULES:
- Common event types include: concert, festival, club night, open mic, karaoke, choir practice, etc. But be creative and adapt to the query. For example, "Are there any live music shows in NYC this weekend?" -> event type is likely "concert" or "live music show"


ATTENDING RULES:
- Four main categories would be "Alone", "With friends", "With a date", "Family", yet this list is not exhaustive. If nothing is mentioned, leave it empty, it's not the most critical field.
- Sometimes categories are named implicitly, with slang... For example, "with my uncle and my cousin" -> "family",  "with my buddies/crew/squad" -> "with friends"...

TIME RULES:
- ALL THE EVENTS ARE IN 2026. Use this as a reference when inferring time ranges for relative time expressions.
- Always return tuples with first the beginning and then the end of the time range. Both elements must be written with absolute timestamps with ISO format (YYYY-MM-DD). Use internal calendar and common sense to infer the exact dates for relative time expressions like "this weekend", "next week", "tomorrow", "in 3 days".
- If the query does not mention the day but only the month (eg: "any concerts in NYC in December"), return (2026-12-01, 2026-12-31)
- If the query does not mention the month but only the season (eg: "any concerts in NYC this summer"), return the corresponding months for summer (2026-06-01, 2026-08-31)

KEYWORDS RULES:
- Extract any other relevant keyword that could help with search/ranking, even if it's not explicitly
- This is the most flexible field, try to extract as much information as possible from the query to help the ranking algorithm, but be careful not to hallucinate or include irrelevant filler words. Use your best judgment and adapt to the query.




Return structured JSON only.
"""

    user_prompt = f"""
Extract intent from this query:

{user_query}

Return exactly this JSON:

{{
  "cities": [],
  "states": [],
  "artists": [],
  "genres": [],
  "subgenres": [],
  "mood": [],
  "keywords": [],
  "event_type": [],
  "attending": [],
  "time": []
}}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        data = normalize_intent(data)
        print(data)

        return data

    except Exception as error:
        print("Groq fallback:", error)
        return fallback_intent(user_query)


# ----------------------------
# NORMALIZATION (IMPORTANT)
# ----------------------------
def safe_list(x):
    """
    Ensures the value is always a list.
    """
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    return []


def normalize_intent(data):
    """
    Cleans and standardizes Groq output for search pipeline.
    Compatible with:
    - cities (list)
    - artists, genres, subgenres, keywords (lists)
    - mood (list or string)
    - time dict
    """

    return {
        "cities": [
            c.lower().strip()
            for c in safe_list(data.get("cities"))
            if isinstance(c, str)
        ],

        "states": [
            s.lower().strip()
            for s in safe_list(data.get("states"))
            if isinstance(s, str)
        ],

        "artists": [
            a.lower().strip()
            for a in safe_list(data.get("artists"))
            if isinstance(a, str)
        ],

        "genres": [
            g.lower().strip()
            for g in safe_list(data.get("genres"))
            if isinstance(g, str)
        ],

        "subgenres": [
            s.lower().strip()
            for s in safe_list(data.get("subgenres"))
            if isinstance(s, str)
        ],

        "mood": [
            m.lower().strip()
            for m in safe_list(data.get("mood"))
            if isinstance(m, str)
        ],

        "keywords": [
            k.lower().strip()
            for k in safe_list(data.get("keywords"))
            if isinstance(k, str)
        ],

        "event_type": [
            e.lower().strip()
            for e in safe_list(data.get("event_type"))
            if isinstance(e, str)
        ],

        "attending": [
            a.lower().strip()
            for a in safe_list(data.get("attending"))
            if isinstance(a, str)
        ],

        "time": [
            date_tuple 
            for date_tuple in safe_list(data.get("time"))
            if isinstance(date_tuple, list) and len(date_tuple) == 2
        ]
    }

# ----------------------------
# FALLBACK (RULE BASED)
# ----------------------------
def fallback_intent(user_query):
    """
    Simple rule-based fallback réécrit pour correspondre 
    parfaitement à la structure attendue.
    """
    query_lower = user_query.lower()

    known_cities = [
        "new york", "los angeles", "chicago", "houston",
        "miami", "seattle", "boston", "atlanta",
        "san francisco", "las vegas", "dallas", "denver"
    ]
    # 'cities' doit être une liste
    cities = [c for c in known_cities if c in query_lower]

    known_genres = [
        "kpop", "pop", "rock", "jazz", "edm", "electronic",
        "indie", "hip hop", "country", "classical"
    ]
    genres = [g for g in known_genres if g in query_lower]

    known_moods = ["chill", "party", "sad", "energetic", "happy"]
    moods = [m for m in known_moods if m in query_lower]

    # Nettoyage rapide des mots-clés
    keywords = user_query.lower().split()
    for c in cities:
        if c in keywords: keywords.remove(c)

    return {
        "intent": "event_search",
        "city": cities,
        "state": "",
        "artists": [],
        "genres": genres,
        "subgenres": [],
        "mood": moods,
        "keywords": keywords,
        "event_type": "",
        "attending": "",
        "time": {
            "relative": "",
            "absolute": ""
        }
    }