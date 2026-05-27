import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

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
- Output ONLY valid JSON
- Do NOT use markdown
- You can try to infer missing information (for example: if the artist is Shakira, the genre is likely pop/latin. If the query mentions "me and my girlfriend", field "attending" is likely "couple", and so on), but only if it's very obvious. Otherwise, leave it blank.
- Remove filler/slang words (yo, man, bro, help me, etc.)

TIME RULES:
- Convert relative time (this weekend, tomorrow, next Saturday)
- Convert absolute dates (30 May, May 30) into ISO format: YYYY-MM-DD if possible. If not mentioned explicitly, year is always 2026
- If both exist, keep both

KEYWORDS RULES:S
- Keep only meaningful descriptive terms
- Remove slang (lit, fire, yo, etc.)
- Remove filler words

EVENT TYPES:
- rave, techno party, concert, festival, nightlife, club event...

ATTENDING:
- alone, friends, couple, family, etc.

Return structured JSON only.
"""

    user_prompt = f"""
Extract intent from this query:

{user_query}

Return exactly this JSON:

{{
  "cities": [],
  "artists": [],
  "genres": [],
  "subgenres": [],
  "mood": [],
  "keywords": [],
  "event_type": "",
  "attending": "",
  "time": {{
    "relative": "",
    "absolute": ""
  }}
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

        "event_type": (data.get("event_type") or "").lower().strip(),

        "attending": (data.get("attending") or "").lower().strip(),

        "time": {
            "relative": (data.get("time") or {}).get("relative", ""),
            "absolute": (data.get("time") or {}).get("absolute", "")
        }
    }

# ----------------------------
# FALLBACK (RULE BASED)
# ----------------------------
def fallback_intent(user_query):
    """
    Simple rule-based fallback if Groq fails.
    """

    query_lower = user_query.lower()

    known_cities = [
        "new york", "los angeles", "chicago", "houston",
        "miami", "seattle", "boston", "atlanta",
        "san francisco", "las vegas", "dallas", "denver",
        "phoenix", "portland", "orlando", "san diego",
        "washington", "philadelphia", "austin", "nashville"
    ]

    city = ""
    for c in known_cities:
        if c in query_lower:
            city = c
            break

    known_genres = [
        "kpop", "pop", "rock", "jazz", "edm", "electronic",
        "indie", "hip hop", "country", "classical", "metal",
        "r&b", "soul", "latin"
    ]

    genres = [g for g in known_genres if g in query_lower]

    known_moods = ["chill", "party", "sad", "energetic", "happy"]
    mood = ""
    for m in known_moods:
        if m in query_lower:
            mood = m
            break

    keyword = user_query.lower()

    if city:
        keyword = keyword.replace(city, "")

    return {
        "intent": "event_search",
        "city": city,
        "artists": [],
        "genres": genres,
        "subgenres": [],
        "mood": mood,
        "keywords": keyword.split(),
        "time_filter": "",
        "popularity_bias": "any"
    }