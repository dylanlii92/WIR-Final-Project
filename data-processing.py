import requests
import sqlite3
import time

API_KEY = "4TXAXqdt2PlnR7i4QK6ceYp1pjivoB0n"
BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"

# ---------------------------
# SQLITE
# ---------------------------
conn = sqlite3.connect("ticketmaster_music.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    name TEXT,
    date TEXT,
    city TEXT,
    country TEXT,
    source TEXT
)
""")

conn.commit()

# ---------------------------
# STRATEGY (IMPORTANT)
# ---------------------------
KEYWORDS = [
    "concert", "music", "live", "festival", "tour", "DJ", "band", "orchestra", "symphony", "opera", "hip hop", "rock", "pop", "jazz", "classical", "electronic", "indie", "alternative", "country", "reggae", "blues", "metal", "punk", "folk", "R&B", "soul", "funk", "latin", "world", "gospel", "choir", "karaoke", "open mic", "jam session", "music hall", "arena", "stadium", "amphitheater", "venue", "evening", "night", "party", "weekend"
]
CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston",
    "Miami", "Seattle", "Boston", "Atlanta",
    "San Francisco", "Las Vegas", "Dallas", "Denver", "Phoenix", "Portland", "Orlando", "San Diego", "Washington", "Philadelphia", "Austin", "Nashville", "Detroit", "Minneapolis", "Cleveland", "St. Louis", "Baltimore", "Charlotte", "Columbus", "Indianapolis", "San Antonio", "San Jose", "New Orleans", "Memphis", "Kansas City", "Sacramento", "Raleigh", "Oklahoma City", "Cincinnati", "Pittsburgh", "Las Cruces", "Albuquerque"
]

# ---------------------------
def save_event(event, source):
    try:
        event_id = event["id"]
        name = event.get("name")

        date = None
        if event.get("dates"):
            date = event["dates"]["start"].get("localDate")

        city = None
        country = None

        if event.get("_embedded"):
            venues = event["_embedded"].get("venues", [])
            if venues:
                city = venues[0].get("city", {}).get("name")
                country = venues[0].get("country", {}).get("name")

        cursor.execute("""
            INSERT OR IGNORE INTO events
            VALUES (?, ?, ?, ?, ?, ?)
        """, (event_id, name, date, city, country, source))

    except Exception:
        pass

# ---------------------------
def fetch(params, source):
    page = 0
    total_pages = 1

    while page < total_pages:
        params["page"] = page
        params["size"] = 200

        r = requests.get(BASE_URL, params=params)
        data = r.json()

        if "_embedded" not in data:
            break

        events = data["_embedded"]["events"]

        for e in events:
            save_event(e, source)

        conn.commit()

        total_pages = data["page"]["totalPages"]
        page += 1

        print(f"{source} → page {page}/{total_pages}")

        time.sleep(0.2)

# ---------------------------
# RUN PIPELINE
# ---------------------------

# 1. keyword-based crawl
for kw in KEYWORDS:
    fetch({
        "apikey": API_KEY,
        "classificationName": "Music",
        "countryCode": "US",
        "keyword": kw
    }, source=f"kw:{kw}")

# 2. city-based crawl
for city in CITIES:
    fetch({
        "apikey": API_KEY,
        "classificationName": "Music",
        "city": city,
        "countryCode": "US"
    }, source=f"city:{city}")

# ---------------------------
# RESULT
# ---------------------------
cursor.execute("SELECT COUNT(*) FROM events")
print("\nTOTAL EVENTS STORED:", cursor.fetchone()[0])

conn.close()