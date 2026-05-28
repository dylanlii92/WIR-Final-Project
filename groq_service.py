import json
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")

client = Groq(api_key=GROQ_API_KEY)

def extract_search_intent(user_query):
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
- However, it must be an adjective refering to the vibe of an event. Words like "lit", "amazing", "biggest", etc., don't belong to this category, but more to the keywords category.

EVENT TYPE RULES:
- Common event types include: concert, festival, club night, open mic, karaoke, choir practice, etc. But be creative and adapt to the query. For example, "Are there any live music shows in NYC this weekend?" -> event type is likely "concert" or "live music show"


ATTENDING RULES:
-  Main categories would be "Alone", "With friends", "With a date", "Family", "Colleagues". try to stick to these categories, unless the query explicitly mentions something else. 
- Sometimes categories are named implicitly, with slang... For example, "with my uncle and my cousin" -> "family",  "with my buddies/crew/squad" -> "with friends"...

TIME RULES:
- ALL THE EVENTS ARE IN 2026. Use 2026 as the base year for relative expressions (e.g., "this weekend", "in august").
- Return a dictionary with "start" and "end" keys using absolute ISO format (YYYY-MM-DD).
- If the query mentions a month (e.g., "in december"), return {"start": "2026-12-01", "end": "2026-12-31"}.
- If no time is specified, return {"start": "", "end": ""}.

KEYWORDS RULES:
- Extract any other relevant keyword that could help with search/ranking, even if it's not explicitly mentioned.
- This is the most flexible field, try to extract as much information as possible from the query to help the ranking algorithm, but be careful not to hallucinate or include irrelevant filler words. Use your best judgment and adapt to the query.
- Try not to repeat information already captured in other fields (e.g., cities, artists, genres). 
"""

    user_prompt = f"""
Extract intent from this query: "{user_query}"

Return exactly this JSON structure:
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
  "time": {{
    "start": "",
    "end": ""
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
        print("Groq fallback triggered due to error:", error)
        return fallback_intent(user_query)


def safe_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str): return [x]
    return []


def normalize_intent(data):
    """ Standardize data for ranking algorithm """
    time_data = data.get("time", {})
    if not isinstance(time_data, dict):
        time_data = {"start": "", "end": ""}

    return {
        "cities": [str(c).lower().strip() for c in safe_list(data.get("cities")) if c],
        "states": [str(s).lower().strip() for s in safe_list(data.get("states")) if s],
        "artists": [str(a).lower().strip() for a in safe_list(data.get("artists")) if a],
        "genres": [str(g).lower().strip() for g in safe_list(data.get("genres")) if g],
        "subgenres": [str(s).lower().strip() for s in safe_list(data.get("subgenres")) if s],
        "mood": [str(m).lower().strip() for m in safe_list(data.get("mood")) if m],
        "keywords": [str(k).lower().strip() for k in safe_list(data.get("keywords")) if k],
        "event_type": [str(e).lower().strip() for e in safe_list(data.get("event_type")) if e],
        "attending": [str(a).lower().strip() for a in safe_list(data.get("attending")) if a],
        "time": {
            "start": time_data.get("start", ""),
            "end": time_data.get("end", "")
        }
    }


def fallback_intent(user_query):
    """ Fallback in case of Groq API failure"""
    return {
        "cities": [],
        "states": [],
        "artists": [],
        "genres": [],
        "subgenres": [],
        "mood": [],
        "keywords": [user_query.lower()],
        "event_type": [],
        "attending": [],
        "time": {"start": "", "end": ""}
    }