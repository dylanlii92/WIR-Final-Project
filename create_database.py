import os
import time
import sqlite3
import requests

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("JENNY_TICKETMASTER_API_KEY")
BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"

conn = sqlite3.connect("ticketmaster_music.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    name TEXT,
    artist TEXT,
    date TEXT,
    city TEXT,
    country TEXT,
    venue TEXT,
    genre TEXT,
    subgenre TEXT,
    description TEXT,
    image TEXT,
    url TEXT,
    source TEXT
)
""")

conn.commit()


KEYWORDS = [
    "concert", "music", "live", "festival", "tour", "DJ", "band", "orchestra",
    "symphony", "opera", "hip hop", "rock", "pop", "jazz", "classical", "electronic",
    "indie", "alternative", "country", "reggae", "blues", "metal", "punk", "folk", "R&B",
    "soul", "funk", "latin", "world", "gospel", "choir", "karaoke", "open mic", "jam session",
    "music hall", "arena", "stadium", "amphitheater", "venue", "evening", "night", "party", "weekend"
]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston",
    "Miami", "Seattle", "Boston", "Atlanta",
    "San Francisco", "Las Vegas", "Dallas", "Denver", "Phoenix", "Portland", "Orlando",
    "San Diego", "Washington", "Philadelphia", "Austin", "Nashville", "Detroit", "Minneapolis",
    "Cleveland", "St. Louis", "Baltimore", "Charlotte", "Columbus", "Indianapolis", "San Antonio",
    "San Jose", "New Orleans", "Memphis", "Kansas City", "Sacramento", "Raleigh", "Oklahoma City",
    "Cincinnati", "Pittsburgh", "Las Cruces", "Albuquerque"
]

def extract_artists(event):
    """
    Extracts artist names from Ticketmaster attractions.
    """

    attractions = event.get("_embedded", {}).get("attractions", [])

    artist_names = []

    for attraction in attractions:
        name = attraction.get("name")

        if name:
            artist_names.append(name)

    return ", ".join(artist_names)

def save_event(event, source):
    event_id = event.get("id")
    name = event.get("name")
    artist = extract_artists(event)
    url = event.get("url", "")

    start = event.get("dates", {}).get("start", {})
    date = start.get("localDate")

    description = (
        event.get("info")
        or event.get("pleaseNote")
        or event.get("description")
        or ""
    )

    venue = ""
    city = ""
    country = ""

    venues = event.get("_embedded", {}).get("venues", [])
    if venues:
        venue_data = venues[0]
        venue = venue_data.get("name", "")
        city = venue_data.get("city", {}).get("name", "")
        country = venue_data.get("country", {}).get("name", "")

    genre = ""
    subgenre = ""

    classifications = event.get("classifications", [])
    if classifications:
        genre = classifications[0].get("genre", {}).get("name", "")
        subgenre = classifications[0].get("subGenre", {}).get("name", "")

    images = event.get("images", [])
    image = images[0].get("url", "") if images else ""

    cursor.execute("""
    INSERT OR IGNORE INTO events
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    event_id,
    name,
    artist,
    date,
    city,
    country,
    venue,
    genre,
    subgenre,
    description,
    image,
    url,
    source
))


def fetch(params, source):
    page = 0
    total_pages = 1

    while page < total_pages:
        params["page"] = page
        params["size"] = 200

        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if "_embedded" not in data:
            break

        events = data["_embedded"]["events"]

        for event in events:
            save_event(event, source)

        conn.commit()

        total_pages = data.get("page", {}).get("totalPages", 1)
        page += 1

        print(f"{source} → page {page}/{total_pages}")

        time.sleep(0.2)


for keyword in KEYWORDS:
    fetch({
        "apikey": API_KEY,
        "classificationName": "Music",
        "countryCode": "US",
        "keyword": keyword
    }, source=f"keyword:{keyword}")


for city in CITIES:
    fetch({
        "apikey": API_KEY,
        "classificationName": "Music",
        "city": city,
        "countryCode": "US"
    }, source=f"city:{city}")


cursor.execute("SELECT COUNT(*) FROM events")
print("TOTAL EVENTS STORED:", cursor.fetchone()[0])

conn.close()