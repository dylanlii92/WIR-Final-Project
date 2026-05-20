import json
import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")


def extract_search_intent(user_query):
    """
    Extracts structured music-event search intent from a natural-language query.
    The system only supports US events.
    """

    prompt = f"""
    Extract structured search intent from this music-event query:

    "{user_query}"

    Return ONLY valid JSON:

    {{
        "keyword": string,
        "city": string,
        "mood": string,
        "genres": list of strings
    }}

    Rules:
    - The system only supports events in the United States.
    - Do not include countryCode.
    - Return only JSON.
    - No explanations.
    """

    response = model.generate_content(prompt)

    text = response.text.strip()

    try:
        return json.loads(text)

    except Exception:
        return {
            "keyword": user_query,
            "city": "",
            "mood": "",
            "genres": []
        }