import json
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

# ---------------------------------------------------------
# INITIALIZE KEYWORD SEARCH INDEX (BM25)
# ---------------------------------------------------------
# Build the sparse matrix index on the preloaded documents
BM25 = BM25Okapi([tokenize(doc) for doc in DOCS])

# ---------------------------------------------------------
# MAIN SEARCH & HYBRID TIERING ENGINE
# ---------------------------------------------------------
def search_events(query, intent):
    """
    Executes a hybrid search combining sparse (BM25) and dense (Embeddings) scores,
    applies business logic constraints (Tiers matching rules), and triggers an LLM reranker.
    """
    retrieval_query = query.lower().strip()

    if not retrieval_query:
        return []

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
        intent_cities = [c.lower() for c in intent.get("cities", [])]
        intent_states = [s.lower() for s in intent.get("states", [])]
        intent_artists = [a.lower() for a in intent.get("artists", [])]
        intent_genres = [g.lower() for g in intent.get("genres", [])]
        intent_subgenres = [s.lower() for s in intent.get("subgenres", [])]
        
        time_intent = intent.get("time", {})
        start_target = time_intent.get("start", "")
        end_target = time_intent.get("end", "")

        # ---- RULE 1: City Check ----
        city_matched = False
        for city in intent_cities:
            if city in meta["city"].lower():
                multiplier += 0.5
                bonus_details["city_match"] = 0.5
                city_matched = True
        if city_matched:
            criteria_met_count += 1
        
        # ---- RULE 2: State Code Check ----
        state_matched = False
        for state in intent_states:
            if state in meta["state"].lower():
                multiplier += 0.4
                bonus_details["state_match"] = 0.4
                state_matched = True
        if state_matched:
            criteria_met_count += 1

        # ---- RULE 3: Performer/Artist Check ----
        artist_matched = False
        for artist in intent_artists:
            if artist in meta["artist"].lower():
                multiplier += 1.0
                bonus_details["artist_match"] = 1.0
                artist_matched = True
        if artist_matched:
            criteria_met_count += 1

        # ---- RULE 4: Main Music Genre Check ----
        genre_matched = False
        for genre in intent_genres:
            if genre in (meta["genre"] + " " + meta["subgenre"]).lower():
                multiplier += 0.3
                bonus_details["genre_match"] = 0.4
                genre_matched = True
        if genre_matched:
            criteria_met_count += 1

        # ---- RULE 5: Specific Subgenre Check ----
        subgenre_matched = False
        for subgenre in intent_subgenres:
            if subgenre in (meta["genre"] + " " + meta["subgenre"]).lower():
                multiplier += 0.3
                bonus_details["subgenre_match"] = 0.4
                subgenre_matched = True
        # Avoid double-counting criterion if genre and subgenre matched the same keyword token
        if subgenre_matched and not genre_matched:
            criteria_met_count += 1

        # ---- RULE 6: Absolute Date / Time Window Check ----
        event_date = meta["date"] # Expected format from database structure: "YYYY-MM-DD"
        time_matched = False
        
        if start_target and end_target and event_date:
            # Check if the catalog event falls within the bounds of the extracted dates
            if start_target <= event_date <= end_target:
                multiplier += 0.5
                bonus_details["date_window_match"] = 0.5
                time_matched = True
                
        if time_matched:
            criteria_met_count += 1

        # ---- BONUS LAYERS (Soft criteria - boost scores without altering strict tiers) ----
        for keyword in intent.get("keywords", []):
            if keyword.lower() in DOCS[i]:
                multiplier += 0.1
                bonus_details[f"keyword_{keyword}"] = 0.1

        for mood in intent.get("mood", []):
            if mood.lower() in DOCS[i]:
                multiplier += 0.2
                bonus_details["mood_match"] = 0.2

        # Final compounding score multiplication
        final_score = float(base_score * multiplier)

        # Append payload schema required for frontend injection and debug logs
        all_scored_events.append({
            "id": i,
            "score": final_score,
            "criteria_count": criteria_met_count,
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
    all_scored_events.sort(key=lambda x: (x["criteria_count"], x["score"]), reverse=True)

    # Slice the top 20 best candidates out of the matrix to process through LLM evaluation
    candidates = all_scored_events[:20]

    # 4. Perform localized LLM contextual re-ranking on the high-quality candidates subset
    reranked = rerank_events(query, intent, candidates)

    # --- COMPREHENSIVE SCOREBOARD PRINT IN CONSOLE ---
    print("\n" + "="*60)
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
            
        llm_score = e.get("llm_score", 0)
        print(f"     |- 🤖 LLM Reranker Score: {llm_score}/10 (Added contribution: +{llm_score/10:.2f})")
    
    print("="*60 + "\n")

    return reranked


# ---------------------------------------------------------
# LLM STEP-2 CONTEXTUAL RE-RANKER
# ---------------------------------------------------------
def rerank_events(query, intent, candidates):
    """
    Leverages a reasoning model to cross-examine and score candidates, 
    acting as a logical filter against location and genre hallucinations.
    """
    # Safety Check: If no candidates reached this stage, return an empty list immediately
    if not candidates:
        return []

    # Flatten the payload into an optimized textual manifest to save LLM tokens
    events_text = ""
    for i, e in enumerate(candidates):
        events_text += f"""
[{i}]
Title: {e['name']}
Artist: {e['artist']}
Genre: {e['genre']}
Subgenre: {e['subgenre']}
City: {e['city']}
State: {e['state']}
Date: {e['date']}
Description: {e['description'][:200]}
"""

    prompt = f"""
You are a high-precision ranking expert. 
Your job is to rate how well each event matches the user's specific request.

CRITICAL: 
- Check the City, Genre, and Date carefully. If an event is in a completely different city or different genre than requested, its score should be extremely low (0 or 1), even if the description looks fun.

User query:
{query}

Intent parsed:
{intent}

Rate each event from 0 to 10 based on exact relevance.
Return JSON only in this format:
{{
  "scores": [0,0,0,...]
}}

Events list:
{events_text}
"""

    # Initialize all candidates with an LLM score of 0 before making the API call
    # This prevents incomplete states if the API fails midway or crashes
    for e in candidates:
        e["llm_score"] = 0

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            response_format={"type":"json_object"}
        )

        # Parse output safely
        response_content = json.loads(response.choices[0].message.content)
        scores = response_content.get("scores", [])

        # Map scores dynamically
        for i, e in enumerate(candidates):
            if i < len(scores):
                llm_score_normalized = float(scores[i]) / 10.0
                e["llm_score"] = scores[i]
                e["score"] += llm_score_normalized

    except Exception as error:
        # If Groq/Llama crashes, timeouts, or sends bad JSON, we safely catch it here
        print(f"⚠️ Error during LLM reranking: {error}. Falling back to initial algorithmic scores.")

    # --- CRITICAL FIX: ABSOLUTE AND UNCONDITIONAL RETURN ---
    # This line MUST be completely outside the try/except block (indented to the left).
    # It guarantees that whatever happens up there, a valid list is returned to app.py.
    final_sorted_list = sorted(candidates, key=lambda x: (x["criteria_count"], x["score"]), reverse=True)
    
    return final_sorted_list[:8]