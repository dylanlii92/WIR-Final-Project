import json
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")

client = Groq(api_key=GROQ_API_KEY)

import json
from datetime import datetime

def extract_search_intent(user_query):
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")                      
    day_of_week = now.strftime("%A")                           

    system_prompt = f"""
You are a high-precision query understanding system for a music/event search engine.

CRITICAL TIME ANCHOR:
- TODAY'S DATE IS EXACTLY: {today_str} ({day_of_week}).
- Use this exact date as your reference point for ALL relative time calculations.

CRITICAL RULES:
- Extract all intent in lowercase.
- Output ONLY valid JSON
- Do NOT use markdown
- Remove filler/slang words (yo, man, bro, help me, etc.)
- Do not only use examples given in the prompt, be creative and adapt to the query. 
- Try not to repeat information accross fields, for example if a event type is "party", do not mention it again in keywords, mood... In that case, mood could be "fun", "energetic"...
- You are more than welcome to infer some fields, yet if you are really not sure, it's safer to leave a field empty than to hallucinate.

TITLES RULES:
- Extract any mention of specific event titles. 
- Only extract explicit names. For example, "I am looking for tickets for Lollapalooza" -> title is "lollapalooza", but "I want to go to a Coldplay concert" -> no title, even if Coldplay is an artist, the user didn't explicitly mention a specific event title. 

CITIES RULES:
- Always extract city names in lowercase (e.g., "new york", "los angeles", "chicago"). 
- If only the state is mentioned without a city, leave the cities list empty and rely on the states field.

STATES RULES:
- For the 'states' field, extract ONLY the official 2-letter United States state code (e.g., 'CA' for California, 'NY' for New York, 'IL' for Illinois). Never return the full state name or colloquial abbreviations.
- If only the city is mentioned, populate the "states" field based on a city-to-state mapping.

ARTISTS RULES:
- If you are not sure if a term is an artist, you can try to search on the web to confirm. If it remains ambiguous, it's safer to leave it out of the artists list.
- Be careful not to mix up artist names with genres or keywords. For example, "The Weeknd" is an artist, while "weekend" is a common word. Also, rap artists can sometimes have "rapper" in their name. Moreover, "female rap" is not likely to be an artist name, unless the query says something like "I want to go to the concert of "Female Rap" this summer".

GENRES/SUBGENRES RULES:
-  If a term is ambiguous (e.g., "rock" could be a genre or a verb), use your best judgment based on context. If unsure, it's safer to leave it out.
- Try to extract first the main genres, and then the subgenres. For example, if the query says "I want to go to a jazz or blues concert", you can write "jazz" and "blues" in the genres field, but if the query says "I want to go to a smooth jazz concert", you can write "jazz" in the genres field and "smooth jazz" in the subgenres field.
- Do not ever fill both fields with the same term. For example, if the query says "I want to go to a rock concert", you can write "rock" in the genres field, but not in the subgenres field.
- Try not to mistake artist names/ venues for genres. If unsure, use context clues or search the web to confirm.
- If the query mentions "i can listen to any genre except rock", do not write "every genre" in the genres field, but rather leave it empty and write "rock" in the exclusions.genres field. The same applies to subgenres.

MOOD RULES:
- This is more subjective, but try as much as you can to infer one or several moods to every query.
- Most common moods include: chill, fun, sad, happy, energetic, romantic, family-friendly, chaotic, etc.
- Do not only rely on explicit words, the overall vocabulary and tone of the query can also give clues about the mood. Eg: "I am not feeling too great atm but I want to listen to some live music" -> mood is likely "sad" or "chill" even if those words are not explicitly mentioned.
- However, it must be an adjective refering to the vibe of an event. Words like "lit", "amazing", "biggest", etc., don't belong to this category, but more to the keywords category.

EVENT TYPE RULES:
- Common event types include: concert, festival, club night, open mic, karaoke, choir practice, etc. But be creative and adapt to the query. For example, "Are there any live music shows in NYC this weekend?" -> event type is likely "concert" or "live music show"

ATTENDING RULES:
-  Main categories would be "Alone", "With friends", "With a date", "With family (adults)", "With family(children)", "With colleagues". Try to stick to these categories, unless the query explicitly mentions something else. 
- Sometimes categories are named implicitly, with slang... For example, "with my uncle and my cousin" -> "family",  "with my buddies/crew/squad" -> "with friends"...
- Response must be a possible answer to the question "Who are you going to the event with?"
- You must be able to distinguish between "with family (adults)" and "with family (children)", as lots of events are not kid-friendly. If both groups are mentioned or if the query just says "with family" without specifying, put "With family (children)", as children are likely to be the most restrictive factor when it comes to event attendance.

 
TIME RULES:
- Convert any text dates into absolute ISO format (YYYY-MM-DD) inside the "start" and "end" fields based on the CRITICAL TIME ANCHOR provided above.
- CALENDAR CALCULATION RULES:
  * "tomorrow": Add 1 day to the anchor date.
  * "this weekend": Return the range from the upcoming Friday to Sunday. (e.g., If today is Monday {today_str}, 'this weekend' starts this coming Friday and ends on Sunday).
  * "next week": Calculate the dates for the subsequent Monday to Sunday range.
  * "in X days": Add X days to the anchor date.
- EXAMPLES of conversions (Assuming anchor is Monday 2026-06-01):
  * "tomorrow" -> {{"start": "2026-06-02", "end": "2026-06-02"}}
  * "this weekend" -> {{"start": "2026-06-05", "end": "2026-06-07"}}
  * "from 5th july to 15th august" -> {{"start": "2026-07-05", "end": "2026-08-15"}}
  * "in december" -> {{"start": "2026-12-01", "end": "2026-12-31"}}
- If the user specifies a range without an explicit year, always assume it happens in 2026.
- If the user specifies a date range that has already started or ended, keep the original absolute dates without adjusting to the current date.

KEYWORDS RULES:
- Extract any other relevant keyword that could help with search/ranking, even if it's not explicitly mentioned.
- This is the most flexible field, try to extract as much information as possible from the query to help the ranking algorithm, but be careful not to hallucinate or include irrelevant filler words. Use your best judgment and adapt to the query.
- Try not to repeat information already captured in other fields (e.g., cities, artists, genres). 

EXCLUSIONS RULES:
- If the user query contains any explicit exclusions (e.g., "not in New York", "without Drake", "no rock music", "any time but June", "exclude concerts"...), capture those in the "exclusions" field with the same structure as the main intent fields. This will help the ranking algorithm to filter out unwanted results more effectively.
"""

    user_prompt = f"""
Extract intent from this query: "{user_query}"

Return exactly this JSON structure:
{{
  "titles": [],
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
    }},
  "exclusions": {{
    "titles": [],
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
        
        if not data.get("time"):
            data["time"] = {"start": "", "end": ""}
            
        data = normalize_intent(data)
        print("🎯 EXTRACTED INTENT:", data)
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
        "titles": [str(t).lower().strip() for t in safe_list(data.get("titles")) if t],
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
        },
        "exclusions": {
            "titles": [str(t).lower().strip() for t in safe_list(data.get("exclusions", {}).get("titles")) if t],
            "cities": [str(c).lower().strip() for c in safe_list(data.get("exclusions", {}).get("cities")) if c],
            "states": [str(s).lower().strip() for s in safe_list(data.get("exclusions", {}).get("states")) if s],
            "artists": [str(a).lower().strip() for a in safe_list(data.get("exclusions", {}).get("artists")) if a],
            "genres": [str(g).lower().strip() for g in safe_list(data.get("exclusions", {}).get("genres")) if g],
            "subgenres": [str(s).lower().strip() for s in safe_list(data.get("exclusions", {}).get("subgenres")) if s],
            "mood": [str(m).lower().strip() for m in safe_list(data.get("exclusions", {}).get("mood")) if m],
            "keywords": [str(k).lower().strip() for k in safe_list(data.get("exclusions", {}).get("keywords")) if k],
            "event_type": [str(e).lower().strip() for e in safe_list(data.get("exclusions", {}).get("event_type")) if e],
            "attending": [str(a).lower().strip() for a in safe_list(data.get("exclusions", {}).get("attending")) if a],
            "time": {
                "start": data.get("exclusions", {}).get("time", {}).get("start", ""),
                "end": data.get("exclusions", {}).get("time", {}).get("end", "")
            }
        }
    }


def fallback_intent(user_query):
    """ Fallback in case of Groq API failure"""
    return {
        "titles": [],
        "cities": [],
        "states": [],
        "artists": [],
        "genres": [],
        "subgenres": [],
        "mood": [],
        "keywords": [user_query.lower()],
        "event_type": [],
        "attending": [],
        "time": {"start": "", "end": ""},
        "exclusions": {
            "titles": [],
            "cities": [],
            "states": [],
            "artists": [],
            "genres": [],
            "subgenres": [],
            "mood": [],
            "keywords": [],
            "event_type": [],
            "attending": [],
            "time": {"start": "", "end": ""}
        }
    }