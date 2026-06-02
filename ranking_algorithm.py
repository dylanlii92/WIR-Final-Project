import json
import re
from groq_service import client
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# LOAD PRECOMPUTED DATA
# ---------------------------------------------------------
# Load the vectorized representation of the Ticketmaster catalog
DOCS = np.load("docs.npy", allow_pickle=True)          # Textual documents used for BM25 indexing
EMBEDDINGS = np.load("embeddings.npy")                # Dense vectors used for Semantic/Vector Search
METADATA = np.load("metadata.npy", allow_pickle=True)  # Structured database fields (name, date, city, etc.)

if len(DOCS) == 0:
    raise ValueError("docs.npy is empty. Please run your embedding generator script first.")

KNOWN_CITIES = sorted({
    str(meta.get("city", "")).lower().strip()
    for meta in METADATA
    if str(meta.get("city", "")).strip()
}, key=len, reverse=True)

KNOWN_ARTISTS = sorted({
    str(meta.get("artist", "")).lower().strip()
    for meta in METADATA
    if str(meta.get("artist", "")).strip()
}, key=len, reverse=True)

CITY_TO_STATES = {}
for meta in METADATA:
    city = str(meta.get("city", "")).lower().strip()
    state = str(meta.get("state", "")).lower().strip()

    if city and state:
        CITY_TO_STATES.setdefault(city, set()).add(state)

# ---------------------------------------------------------
# EMBEDDING MODEL DEFINITION
# ---------------------------------------------------------
# Lightweight and fast bi-encoder to map the search query into the same vector space
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# TOKENIZATION & TEXT CLEANING
# ---------------------------------------------------------
# Stopwords filtering out transactional and generic search terms to optimize keyword matching
STOP_WORDS = {
    "music", "event", "show", "tour",
    "in", "near", "the", "a", "an",
    "for", "i", "want", "to", "see",
    "find", "me"
}

def tokenize(text: str):
    """
    Splits string into lowercase tokens and drops non-informative stopwords.
    """
    return [
        t for t in text.lower().split()
        if t not in STOP_WORDS
    ]

def phrase_in_query(phrase, query):
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, query) is not None

def enrich_intent_from_query(query, intent):
    """
    Adds exact city and artist matches from the local catalog when the LLM intent
    parser misses an obvious entity in the raw query.
    """
    if not isinstance(intent, dict):
        intent = {}

    normalized_query = query.lower().strip()
    enriched = dict(intent)

    for key in ("cities", "states", "artists"):
        value = enriched.get(key, [])
        enriched[key] = value if isinstance(value, list) else []

    for city in KNOWN_CITIES:
        if phrase_in_query(city, normalized_query) and city not in enriched["cities"]:
            enriched["cities"].append(city)

            states = CITY_TO_STATES.get(city, set())
            if len(states) == 1:
                state = next(iter(states))
                if state not in enriched["states"]:
                    enriched["states"].append(state)

    for artist in KNOWN_ARTISTS:
        if phrase_in_query(artist, normalized_query) and artist not in enriched["artists"]:
            enriched["artists"].append(artist)

    return enriched

def sort_event_key(event):
    return (
        event.get("exact_query_match_count", 0),
        event.get("artist_query_match", 0),
        event.get("city_query_match", 0),
        event.get("criteria_count", 0),
        event.get("score", 0)
    )

# ---------------------------------------------------------
# INITIALIZE KEYWORD SEARCH INDEX (BM25)
# ---------------------------------------------------------
# Build the sparse matrix index on the preloaded documents
BM25 = BM25Okapi([tokenize(doc) for doc in DOCS])

# ---------------------------------------------------------
# MAIN SEARCH & HYBRID TIERING ENGINE
# ---------------------------------------------------------
def search_events(query, intent, limit=None, offset=0):
    """
    Executes a hybrid search combining sparse (BM25) and dense (Embeddings) scores,
    applies business logic constraints (Tiers matching rules), and triggers an LLM reranker.
    """
    retrieval_query = query.lower().strip()
    intent = enrich_intent_from_query(retrieval_query, intent)

    if not retrieval_query:
        if limit is None:
            return []

        return {
            "results": [],
            "next_offset": offset,
            "has_more": False
        }

    if limit is not None:
        limit = max(1, int(limit))
        offset = max(0, int(offset))

    # Tokenize query for the BM25 statistical relevance formula
    query_tokens = tokenize(retrieval_query)

    # 1. Compute and normalize BM25 keyword scores to a [0.0 - 1.0] range
    bm25_scores = BM25.get_scores(query_tokens)
    max_bm25 = np.max(bm25_scores) if len(bm25_scores) > 0 else 0
    if max_bm25 > 0:
        bm25_scores = bm25_scores / max_bm25

    # 2. Compute Semantic/Dense score using Cosine Similarity via Dot Product
    query_emb = model.encode(
        retrieval_query,
        normalize_embeddings=True
    )
    emb_scores = np.dot(EMBEDDINGS, query_emb)

    all_scored_events = []

    # 3. Iterate through every catalog item to compute hybrid baseline and score boosts
    for i in range(len(DOCS)):

        bm25_part = float(bm25_scores[i])
        emb_part = float(emb_scores[i])
        
        # Linear combination: 75% keyword exactness, 25% semantic contextual meaning
        base_score = 0.75 * bm25_part + 0.25 * emb_part

        meta = METADATA[i]
        multiplier = 1.0
        bonus_details = {}
        
        # --- STRICT CRITERIA COUNTER (Tier Sorting Flag) ---
        # Keeps track of how many essential user constraints are perfectly satisfied
        criteria_met_count = 0

        # Safely extract entities parsed from the LLM Search Intent Extraction layer
        intent_artists = [a.lower() for a in intent.get("artists", [])]
        intent_cities = [c.lower() for c in intent.get("cities", [])]
        intent_states = [s.lower() for s in intent.get("states", [])]
        intent_titles = [t.lower() for t in intent.get("titles", [])]
        intent_genres = [g.lower() for g in intent.get("genres", [])]
        intent_subgenres = [s.lower() for s in intent.get("subgenres", [])]
        
        time_intent = intent.get("time", {})
        start_target = time_intent.get("start", "")
        end_target = time_intent.get("end", "")

        exclusions = intent.get("exclusions", {})
        exclusion_cities = [c.lower() for c in exclusions.get("cities", [])]
        exclusion_states = [s.lower() for s in exclusions.get("states", [])]
        exclusion_artists = [a.lower() for a in exclusions.get("artists", [])]
        exclusion_titles = [t.lower() for t in exclusions.get("titles", [])]
        exclusion_genres = [g.lower() for g in exclusions.get("genres", [])]
        exclusion_subgenres = [s.lower() for s in exclusions.get("subgenres", [])]
        exclusion_time = exclusions.get("time", {})
        exclusion_start = exclusion_time.get("start", "")
        exclusion_end = exclusion_time.get("end", "")

        city_query_match = 0
        artist_query_match = 0
        exact_query_match_count = 0

        # ---- RULE 1: City Check ----
        city_matched = False
        for city in intent_cities:
            if city in meta["city"].lower():
                multiplier += 0.5
                bonus_details["city_match"] = 0.5
                city_matched = True
        if city_matched:
            criteria_met_count += 1
            city_query_match = 1
            exact_query_match_count += 1
        
        # ---- RULE 2: State Code Check ----
        state_matched = False
        for state in intent_states:
            if state in meta["state"].lower():
                multiplier += 0.3
                bonus_details["state_match"] = 0.3
                state_matched = True
        if state_matched:
            criteria_met_count += 1

        # ---- RULE 3: Event Title Check ----
        title_matched = False
        for title in intent_titles: 
            if title in meta["name"].lower():
                multiplier += 1.0
                bonus_details["title_match"] = 1.0
                title_matched = True
        if title_matched:
            criteria_met_count += 1 

        # ---- RULE 4: Performer/Artist Check ----
        artist_matched = False
        for artist in intent_artists:
            if artist in meta["artist"].lower():
                multiplier += 1.0
                bonus_details["artist_match"] = 1.0
                artist_matched = True
        if artist_matched:
            criteria_met_count += 1
            artist_query_match = 1
            exact_query_match_count += 1

        # ---- RULE 5: Main Music Genre Check ----
        genre_matched = False
        for genre in intent_genres:
            if genre in (meta["genre"] + " " + meta["subgenre"]).lower():
                multiplier += 0.5
                bonus_details["genre_match"] = 0.5
                genre_matched = True
        if genre_matched:
            criteria_met_count += 1

        # ---- RULE 6: Specific Subgenre Check ----
        subgenre_matched = False
        for subgenre in intent_subgenres:
            if subgenre in (meta["genre"] + " " + meta["subgenre"]).lower():
                multiplier += 0.5
                bonus_details["subgenre_match"] = 0.5
                subgenre_matched = True
        if subgenre_matched:
            criteria_met_count += 1

        # ---- RULE 7: Absolute Date / Time Window Check ----
        event_date = meta.get("date", "") # Expected format: "YYYY-MM-DD"
        time_matched = False
        if start_target and end_target and event_date:
            if start_target <= event_date <= end_target:
                multiplier += 0.5
                bonus_details["date_window_match"] = 0.5
                time_matched = True          
        if time_matched:
            criteria_met_count += 1
                
        # Final compounding score multiplication
        final_score = float(base_score * multiplier)

        if intent_cities and artist_matched and not city_matched:
            final_score = final_score * 0.35
            bonus_details["missing_requested_city_penalty"] = -0.65

        is_excluded = False
        reasons_for_exclusion = []

        for exc in exclusion_cities:
            if exc in meta["city"].lower():
                is_excluded = True
                reasons_for_exclusion.append(f"Excluded city match: {exc}")
        
        for exc in exclusion_states:
            if exc in meta["state"].lower():
                is_excluded = True
                reasons_for_exclusion.append(f"Excluded state match: {exc}")
        
        for exc in exclusion_titles:
            if exc in meta["name"].lower():
                is_excluded = True
                reasons_for_exclusion.append(f"Excluded title match: {exc}")
        
        for exc in exclusion_artists:
            if exc in meta["artist"].lower():
                is_excluded = True
                reasons_for_exclusion.append(f"Excluded artist match: {exc}")
        
        for exc in exclusion_genres:
            if exc in (meta["genre"] + " " + meta["subgenre"]).lower():
                is_excluded = True
                reasons_for_exclusion.append(f"Excluded genre match: {exc}")
        
        for exc in exclusion_subgenres:
            if exc in (meta["genre"] + " " + meta["subgenre"]).lower():
                is_excluded = True
                reasons_for_exclusion.append(f"Excluded subgenre match: {exc}")

        if exclusion_start and exclusion_end and event_date:
            if exclusion_start <= event_date <= exclusion_end:
                is_excluded = True
                reasons_for_exclusion.append(f"Excluded date match: {event_date} falls between {exclusion_start} and {exclusion_end}")
        
        if is_excluded:
            final_score = final_score * 0.05  
            criteria_met_count = 0            
            bonus_details["CRITICAL_MALUS"] = reasons_for_exclusion

        # Append payload schema required for frontend injection and debug logs
        all_scored_events.append({
            "id": i,
            "score": final_score,
            "criteria_count": criteria_met_count,
            "exact_query_match_count": exact_query_match_count,
            "artist_query_match": artist_query_match,
            "city_query_match": city_query_match,
            "score_breakdown": {
                "bm25_raw": bm25_part,
                "embedding_raw": emb_part,
                "base_hybrid_score": base_score,
                "total_multiplier": multiplier,
                "bonuses": bonus_details
            },
            "name": meta["name"],
            "artist": meta["artist"],
            "date": meta["date"],
            "city": meta["city"],
            "state": meta["state"],
            "venue": meta["venue"],
            "genre": meta["genre"],
            "subgenre": meta["subgenre"],
            "description": meta["description"],
            "image": meta["image"],
            "url": meta["url"]
        })

    # --- TWO-STEP STRICT TIERS SORTING ---
    # Primary sorting key: Number of satisfied core criteria (descending)
    # Secondary sorting key: Mathematical hybrid baseline weight (descending)
    all_scored_events.sort(key=sort_event_key, reverse=True)

    # Process enough candidates to answer the requested page and know whether more exist.
    candidate_count = 30
    rerank_limit = 21

    if limit is not None:
        rerank_limit = offset + limit + 1
        candidate_count = max(30, rerank_limit)

    candidates = all_scored_events[:candidate_count]

    # 4. Perform localized LLM contextual re-ranking on the high-quality candidates subset
    intent_for_llm = {
        "mood": [m.lower() for m in intent.get("mood", [])],
        "event_type": [e.lower() for e in intent.get("event_type", [])],
        "keywords": [k.lower() for k in intent.get("keywords", [])],
        "attending": [a.lower() for a in intent.get("attending", [])]
    }
    reranked = rerank_events(query, intent_for_llm, candidates, max_results=rerank_limit)

    best_recommendations = []
    you_may_also_like = []
    show_more = []

    for e in reranked:
        if len(best_recommendations) < 3:
            best_recommendations.append(e)
        elif len(you_may_also_like) < 6:
            you_may_also_like.append(e)
        else:
            show_more.append(e)

    # --- COMPREHENSIVE SCOREBOARD PRINT IN CONSOLE ---
    print("\n" + "="*70)
    print(f"📊 SCORE BREAKDOWN FOR QUERY: '{query}'")
    print("="*60)
    
    for rank, e in enumerate(reranked, 1):
        bd = e["score_breakdown"]
        strategy_label = "PRIORITY MATCH" if e["criteria_count"] > 0 else "FILLING EVENT"
        
        print(f"\n🔝 #{rank} : {e['name']} (Artist: {e['artist']} | City: {e['city']})")
        print(f"   ↳ 🎯 STRATEGY LAYER : [{strategy_label}] -> Satisfied {e['criteria_count']} strict criteria")
        print(f"   ↳ 🏆 FINAL SCORE : {e['score']:.4f}")
        print(f"     |- 📝 Base Hybrid Score: {bd['base_hybrid_score']:.4f}")
        print(f"     |  (Normalized BM25: {bd['bm25_raw']:.2f} * 0.75 | Embedding: {bd['embedding_raw']:.2f} * 0.25)")
        print(f"     |- ✖️ Applied Multiplier: x{bd['total_multiplier']:.2f}")
        
        if bd["bonuses"]:
            print("     |  Accumulated Bonus Details:")
            for bonus_name, val in bd["bonuses"].items():
                print(f"     |   • {bonus_name} : +{val}")
        else:
            print("     |  Bonus Details: No bonus applied")
            
# Detailed LLM Reranker Output (Objective vs Subjective)
        llm_score = e.get("llm_score", 0)
        reason = e.get("llm_reason", "No reason provided")
        print(f"     |- 🤖 LLM RERANKER ANALYSIS : {llm_score}/5 pts (Contribution: +{llm_score/5:.2f})")
        print(f"        |    📢 Reason: {reason}")
    

    print(f"Best Recommendations: {len(best_recommendations)}")
    for i, e in enumerate(best_recommendations, 1):
        print(f"   {i}. {e['name']} (Artist: {e['artist']})")
    
    print(f"You May Also Like: {len(you_may_also_like)}")
    for i, e in enumerate(you_may_also_like, 1):
        print(f"   {i}. {e['name']} (Artist: {e['artist']})")

    print(f"Show More: {len(show_more)}")
    for i, e in enumerate(show_more, 1):
        print(f"   {i}. {e['name']} (Artist: {e['artist']})")

    print("="*70 + "\n")

    if limit is None:
        return reranked

    page_results = reranked[offset:offset + limit]

    return {
        "results": page_results,
        "next_offset": offset + len(page_results),
        "has_more": len(reranked) > offset + limit
    }




import json

def rerank_events(query, intent, candidates, max_results=21):
    if not candidates:
        return []
    
    shuffled_candidates = list(candidates)

    events_text = ""
    for i, e in enumerate(shuffled_candidates):
        events_text += f"""
[{i}]
Title: {e['name']}
Artist: {e['artist']}
Genre: {e['genre']}
Subgenre: {e['subgenre']}
City: {e['city']}
State: {e['state']}
Date: {e['date']}
Description: {e['description']}
"""

    prompt = f"""
You are an uncompromising, high-precision music event critic. Your job is to strictly audit and score each candidate event based on how flawlessly it matches the user's query intent.

- Score is out of 5 points.
- Your goal is to assess if the event matches the SUBJECTIVE criteria of the user's query, ie the fields "mood", "keywords", "event_type" and "attending". 
- To do so, you will mainly focus on the description and the title of the event. 
- First check if the event description is relevant or only contains practical information (opening hours, ticketing info, etc). If the description is not relevant, you MUST give a score of 0 points and reason should simply be "No description" or "Description only contains practical information".

### SCORING SCALE EXAMPLES:
- 0 points: No description / description only contains practical information OR the vibe of the event is completely off compared to the query (for example, a metal concert for a "chill evening" query).
- 1-2 points: Poor alignment with the query. Generic description with no clear vibe or which doesn't match the mood of the query.
- 3-4 points: Good alignment with the query. The description contains some relevant information and the vibe matches the context of the query.
- 5 points: Excellent alignment with the query. The event perfectly satisfies all subjective criteria. BE HIGHLY DISCRIMINATIVE.

### ANTI-LAZINESS MANDATE
- Evaluate each event INDEPENDENTLY. Treat every event index with the exact same critical attention and freshness.

User query:
{query}

Intent parsed:
{intent}

### OUTPUT FORMAT INSTRUCTIONS:
- You MUST evaluate and return an object for EVERY single event index provided. Do not skip any items.
- Return ONLY valid JSON matching this structure.

Return JSON only in this exact format structure:
{{
  "events": [    
    {{
      "index": Index of the event in the provided list (integer), 
      "reason": "Reason for the score given. Be specific about which criteria align or misalign with the query intent.",
      "score": Score out of 5 points (integer or float)
    }}
  ]
}}

Events list to audit:
{events_text}
"""

    for e in candidates:
        e["llm_score"] = 0
        e["llm_reason"] = "No reason provided"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        response_content = json.loads(response.choices[0].message.content)
        llm_events = response_content.get("events", [])

        for item in llm_events:
            idx = item.get("index")
            if idx is not None and idx < len(shuffled_candidates):
                score_received = item.get("score", 0)
                reason_received = item.get("reason", "No reason provided")
                
                shuffled_candidates[idx]["llm_score"] = score_received
                shuffled_candidates[idx]["llm_reason"] = reason_received
                shuffled_candidates[idx]["score"] += float(score_received) / 5.0

    except Exception as error:
        print(f"⚠️ Error during LLM reranking: {error}. Falling back to initial algorithmic scores.")

    final_sorted_list = sorted(candidates, key=sort_event_key, reverse=True)

    if max_results is None:
        return final_sorted_list

    return final_sorted_list[:max_results]
