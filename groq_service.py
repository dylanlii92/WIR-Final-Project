import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)


def extract_search_intent(user_query):
    """
    Extracts structured search intent from a natural-language music event query.
    Falls back to rule-based parsing if Groq fails.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON. "
                        "Do not use markdown. "
                        "Do not explain anything."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
Extract search intent from this query:

{user_query}

Return exactly this JSON format:
{{
  "keyword": "music",
  "city": "",
  "mood": "",
  "genres": []
}}
"""
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        text = response.choices[0].message.content.strip()
        print("Groq raw response:", text)

        return json.loads(text)

    except Exception as error:
        print("Groq fallback:", error)
        return fallback_intent(user_query)


def fallback_intent(user_query):
    """
    Rule-based fallback parser.
    """

    query_lower = user_query.lower()

    city = ""
    known_cities = [
        "New York", "Los Angeles", "Chicago", "Houston",
    "Miami", "Seattle", "Boston", "Atlanta",
    "San Francisco", "Las Vegas", "Dallas", "Denver", "Phoenix", "Portland", "Orlando",
    "San Diego", "Washington", "Philadelphia", "Austin", "Nashville", "Detroit", "Minneapolis",
    "Cleveland", "St. Louis", "Baltimore", "Charlotte", "Columbus", "Indianapolis", "San Antonio",
    "San Jose", "New Orleans", "Memphis", "Kansas City", "Sacramento", "Raleigh", "Oklahoma City",
    "Cincinnati", "Pittsburgh", "Las Cruces", "Albuquerque"
    ]

    for known_city in known_cities:
        if known_city in query_lower:
            city = known_city.title()
            break

    genres = []
    known_genres = [
        "kpop", "pop", "rock", "jazz", "edm", "electronic",
        "indie", "hip hop", "country", "classical", "metal",
        "folk", "r&b", "soul", "latin"
    ]

    for genre in known_genres:
        if genre in query_lower:
            genres.append(genre)

    mood = ""
    known_moods = [
        "chill", "romantic", "energetic", "sad",
        "happy", "party", "relaxed"
    ]

    for known_mood in known_moods:
        if known_mood in query_lower:
            mood = known_mood
            break

    keyword = user_query

    if city:
        keyword = query_lower.replace(city.lower(), "")
        keyword = keyword.replace(" in ", " ")
        keyword = keyword.strip()

    if not keyword:
        keyword = user_query

    return {
        "keyword": keyword,
        "city": city,
        "mood": mood,
        "genres": genres
    }