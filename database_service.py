import sqlite3
from pathlib import Path

DATABASE_PATH = Path(__file__).parent / "ticketmaster_music_v1.db"


STOP_WORDS = {
    "music", "event", "show",
    "tour", "in", "near", "the", "a", "an", "for",
    "i", "want", "to", "see", "find", "me"
}

GENRE_SYNONYMS = {
    "edm": [
        "edm", "electronic", "dance", "dance/electronic",
        "house", "techno", "dubstep", "trance"
    ],
    "kpop": [
        "kpop", "k-pop", "korean pop"
    ],
    "hiphop": [
        "hip hop", "hip-hop", "rap"
    ],
    "rnb": [
        "r&b", "rnb", "soul"
    ],
    "pop": [
        "pop"
    ],
    "rock": [
        "rock", "alternative rock", "indie rock"
    ],
    "jazz": [
        "jazz"
    ],
    "country": [
        "country"
    ]
}


def remove_city_from_query(query, city):
    """
    Removes the detected city from the query so artist/genre terms remain.
    """

    if not city:
        return query

    query_lower = query.lower()
    city_lower = city.lower()

    cleaned = query_lower.replace(city_lower, "")
    cleaned = cleaned.replace(" in ", " ")
    cleaned = cleaned.replace(" near ", " ")

    return cleaned.strip()


def search_events(query, city="", genres=None):
    """
    Searches the local event database and ranks results by relevance.
    """

    if genres is None:
        genres = []

    cleaned_query = remove_city_from_query(query, city)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
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
            url
        FROM events
    """)

    rows = cursor.fetchall()
    connection.close()

    events = [dict(row) for row in rows]

    query_lower = cleaned_query.lower().strip()

    query_terms = [
        term for term in query_lower.split()
        if term not in STOP_WORDS
    ]

    city_lower = city.lower().strip()
    genre_terms = [genre.lower() for genre in genres]

    expanded_genres = expand_genres(query_terms, genre_terms)

    ranked_events = []

    for event in events:
        name = str(event.get("name") or "").lower()
        artist = str(event.get("artist") or "").lower()
        event_city = str(event.get("city") or "").lower()
        venue = str(event.get("venue") or "").lower()
        genre = str(event.get("genre") or "").lower()
        subgenre = str(event.get("subgenre") or "").lower()
        description = str(event.get("description") or "").lower()

        searchable_text = " ".join([
            name,
            artist,
            event_city,
            venue,
            genre,
            subgenre,
            description
        ])

        score = 0
        important_match = False
        genre_match = False

        if query_lower and query_lower in artist:
            score += 500
            important_match = True

        if query_lower and query_lower in name:
            score += 300
            important_match = True

        for term in query_terms:
            if term in artist:
                score += 120
                important_match = True

            elif term in name:
                score += 100
                important_match = True

            elif term in genre or term in subgenre:
                score += 70
                important_match = True

            elif term in description:
                score += 20
                important_match = True

        for genre_term in expanded_genres:
            if genre_term in genre or genre_term in subgenre:
                score += 100
                genre_match = True
                important_match = True

            elif genre_term in searchable_text:
                score += 40
                genre_match = True
                important_match = True

        if city_lower and city_lower == event_city:
            score += 100

        elif city_lower and city_lower in event_city:
            score += 70

        if expanded_genres and not genre_match:
            continue

        if not important_match:
            continue

        event["score"] = score
        ranked_events.append(event)

    ranked_events.sort(
        key=lambda event: event["score"],
        reverse=True
    )

    return ranked_events[:12]


def expand_genres(query_terms, genre_terms):
    """
    Expands genre terms with synonyms.
    """

    expanded = []

    all_terms = query_terms + genre_terms

    for term in all_terms:
        normalized = term.lower().replace("-", "")

        expanded.append(term)

        if term in GENRE_SYNONYMS:
            expanded.extend(GENRE_SYNONYMS[term])

        if normalized in GENRE_SYNONYMS:
            expanded.extend(GENRE_SYNONYMS[normalized])

    return list(set(expanded))

print("test")


