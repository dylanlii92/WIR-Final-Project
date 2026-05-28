import os
import time
import sqlite3
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("JENNY_TICKETMASTER_API_KEY")
BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"

# Connexion à ta base v1
conn = sqlite3.connect("ticketmaster_music_v1.db")
cursor = conn.cursor()

# Création de la table (100% rétrocompatible avec ta structure initiale)
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

# Liste des 40 États US majeurs pour un maillage complet du territoire
STATES = [
    "CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MD", "MO", "WI",
    "CO", "MN", "SC", "AL", "LA", "KY", "OR", "OK", "CT", "UT",
    "NV", "IA", "AR", "MS", "KS", "NM", "NE", "ID", "WV", "HI"
]

def extract_artists(event):
    """ Extrait les noms d'artistes depuis les attractions du JSON Ticketmaster """
    attractions = event.get("_embedded", {}).get("attractions", [])
    return ", ".join([a.get("name") for a in attractions if a.get("name")])

def save_event(event, source):
    """ Extrait et sauvegarde un événement unique en base de données """
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

    venue, city, country = "", "", ""
    venues = event.get("_embedded", {}).get("venues", [])
    if venues:
        venue_data = venues[0]
        venue = venue_data.get("name", "")
        city = venue_data.get("city", {}).get("name", "")
        country = venue_data.get("country", {}).get("name", "")

    genre, subgenre = "", ""
    classifications = event.get("classifications", [])
    if classifications:
        genre = classifications[0].get("genre", {}).get("name", "")
        subgenre = classifications[0].get("subGenre", {}).get("name", "")

    images = event.get("images", [])
    image = images[0].get("url", "") if images else ""

    cursor.execute("""
        INSERT OR IGNORE INTO events
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (event_id, name, artist, date, city, country, venue, genre, subgenre, description, image, url, source))

def fetch_with_retry(params, source):
    """
    Télécharge toutes les pages d'une requête en gérant le Rate Limiting (429)
    et en surveillant la barrière physique des 1000 items de Ticketmaster.
    """
    page = 0
    total_pages = 1

    while page < total_pages:
        # Sécurité : Ticketmaster bloque l'accès au catalogue au-delà de 1000 items par filtre
        if page * 200 >= 1000:
            print(f"   -> Deep-paging barrier imminent (1000 items limit) for {source}. Snipping.")
            break

        params["page"] = page
        params["size"] = 200

        try:
            response = requests.get(BASE_URL, params=params)
            
            # Gestion intelligente du Rate Limiting (HTTP 429 Too Many Requests)
            if response.status_code == 429:
                print("⚠️ HTTP 429 detected. Sleeping for 4 seconds to cooldown...")
                time.sleep(4)
                continue  # Relance la même page exacte après la pause
                
            if response.status_code != 200:
                print(f"❌ Error {response.status_code}, skipping page {page}")
                break

            data = response.json()

            if "_embedded" not in data:
                break

            events = data["_embedded"]["events"]
            for event in events:
                save_event(event, source)

            conn.commit()

            total_pages = data.get("page", {}).get("totalPages", 1)
            page += 1
            print(f"   [{source}] Page {page}/{total_pages}")
            
            # Pause réglementaire pour respecter les Quotas par seconde (QPS)
            time.sleep(0.25)

        except Exception as e:
            print(f"⚠️ Exception occurred: {e}. Retrying in 2 seconds...")
            time.sleep(2)

# --- CONFIGURATION DYNAMIQUE JUSQU'À FIN 2026 ---

start_date = datetime.now()
end_target_date = datetime(2026, 12, 31, 23, 59, 59)

# Calcul automatique de la distance temporelle en jours
days_remaining = (end_target_date - start_date).days
# Division par fenêtres de 4 jours pour obtenir le nombre de blocs requis
total_chunks_needed = (days_remaining // 4) + 1

date_intervals = []

for i in range(total_chunks_needed): 
    window_start = start_date + timedelta(days=i*4)
    window_end = window_start + timedelta(days=3)
    
    # Si le bloc déborde sur 2027, on le tronque pile au 31 décembre 2026
    if window_end > end_target_date:
        window_end = end_target_date
        
    date_intervals.append((
        window_start.strftime("%Y-%m-%dT00:00:00Z"),
        window_end.strftime("%Y-%m-%dT23:59:59Z")
    ))
    
    if window_start >= end_target_date:
        break

print(f"🚀 Starting High-Density Scraper (Target: December 31, 2026)...")
print(f"Calendar span: {days_remaining} days to harvest.")
print(f"Total workload: {len(STATES)} states x {len(date_intervals)} windows = {len(STATES)*len(date_intervals)} chunks to scan.")
print("="*60)

# --- BOUCLE PRINCIPALE DE COLLECTE CHIRURGICALE ---

for state in STATES:
    print(f"\n🌍 SCANNING STATE: {state}")
    for start_dt, end_dt in date_intervals:
        
        params = {
            "apikey": API_KEY,
            "classificationName": "Music",
            "countryCode": "US",
            "stateCode": state,
            "startDateTime": start_dt,
            "endDateTime": end_dt
        }
        
        # Format du tag source stocké en BDD : Ex "CA_06-01_to_06-04"
        short_source = f"{state}_{start_dt[5:10]}_to_{end_dt[5:10]}"
        fetch_with_retry(params, source=short_source)

    # --- COMPTEUR EN TEMPS RÉEL APRES CHAQUE ÉTAT ---
    cursor.execute("SELECT COUNT(*) FROM events")
    current_total = cursor.fetchone()[0]
    print(f"\n📊 [PROGRESS] Finished State {state}")
    print(f"📈 Total unique music events in BDD now: {current_total}")
    print("-" * 60)

# --- FIN DU SCRIPT ---
cursor.execute("SELECT COUNT(*) FROM events")
final_total = cursor.fetchone()[0]

print("\n" + "="*60)
print("🎯 MASS SCRAPING MISSION COMPLETED!")
print(f"FINAL VOLUME RECORDED IN DATABASE: {final_total} UNIQUE EVENTS")
print("="*60 + "\n")

conn.close()