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

        # ---- RULE 5: Main Music Genre Check ----
        genre_matched = False
        for genre in intent_genres:
            if genre in (meta["genre"] + " " + meta["subgenre"]).lower():
                multiplier += 0.3
                bonus_details["genre_match"] = 0.5
                genre_matched = True
        if genre_matched:
            criteria_met_count += 1

        # ---- RULE 6: Specific Subgenre Check ----
        subgenre_matched = False
        for subgenre in intent_subgenres:
            if subgenre in (meta["genre"] + " " + meta["subgenre"]).lower():
                multiplier += 0.3
                bonus_details["subgenre_match"] = 0.5
                subgenre_matched = True
        # Avoid double-counting criterion if genre and subgenre matched the same keyword token
        if subgenre_matched:
            criteria_met_count += 1

        # ---- RULE 7: Absolute Date / Time Window Check ----
        event_date = meta.get("date", "") # Expected format: "YYYY-MM-DD"
        time_matched = False
        
        # We check if both start and end dates are provided and not empty
        if start_target and end_target and event_date:
            # Check if the catalog event falls within the bounds of the extracted dates
            if start_target <= event_date <= end_target:
                multiplier += 0.5
                bonus_details["date_window_match"] = 0.5
                time_matched = True   
                
        if time_matched:
            criteria_met_count += 1
                
        # Final compounding score multiplication
        final_score = float(base_score * multiplier)

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

# 4. Trigger Re-ranker on the top 20 candidates
    reranked = rerank_events(query, intent, candidates)

    # --- SÉPARATION STRICTE EN DEUX CATÉGORIES POUR LE FRONTEND ---
    best_recommendations = []
    you_may_also_like = []

    for e in reranked:
        # Condition ultra-sélective : 4 critères stricts ou plus, et maximum 3 événements
        if e["criteria_count"] >= 4 and len(best_recommendations) < 3:
            best_recommendations.append(e)
        else:
            # Tout le reste (les critères < 4 ET les débordements au-delà du top 3) va ici
            you_may_also_like.append(e)

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
        llm_total = e.get("llm_score", 0)
        llm_obj = e.get("llm_objective", 0)
        llm_sub = e.get("llm_subjective", 0)
        print(f"     |- 🤖 LLM RERANKER ANALYSIS : {llm_total}/10 pts (Contribution: +{llm_total/10:.2f})")
        print(f"        |--> 🎯 Objective Criteria Met : {llm_obj}/6 pts  (Title, Artist, City, Genre, Date)")
        print(f"        |--> 🧠 Subjective Vibe Match  : {llm_sub}/4 pts  (Mood, Keywords, Context)")
    
    print("="*70 + "\n")

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
Description: {e['description']}
"""

    prompt = f"""
You are a high-precision ranking expert for music events.  
Your job is to rate how well each event matches the user's specific request.

CRITICAL: 
- Each score must be an integer from 0 to 10, where 0 means "not relevant at all" and 10 means "perfect match".
6 points are for the objective criteria, and 4 points are for the subjective criteria.

- First thing you do check is if the event satisfies the objective criteria extracted in the intent (city, state, artist, genre, subgenre, title, time fields). 
If an event meets none of theses criteria, it should be ranked 0 or 1. The more criteria it meets, the higher the score should be. 
The importance of each criterion is the following: Title > Artist > City > Genre/Subgenre > State > Date.
The maximum score for an event that meets all criteria is 6 points.

- After that, you must analyze how well the event matches the subjective fields (mood, keywords, attending, event_type fields).
To do so, you should mainly use the event description but also possibly the title (especially if the description is missing/very short).
Do not only look for exact word matches, use your reasoning of language to determine if the event is relevant to the user's intent.
For example, if the intent mood is "chill" and the event says "calm evening by the beach", this is a very good match. On the other hand, if the intent attending field says "with family", and the description says "only for 21++", this is a very bad match.
You grading policy must be discriminative, only the most relevant events can get the full 4 subjective points.

- The user may have expressed some exclusions in their intent (see the "exclusions" field). 
If an event matches any of theexclusion criteria, it should be heavilily penalized in the scoring, receiving a score of 0, even if it matches lots of inclusion criteria. 
For example, for the query "concert in june in boston, anything but jazz", if an event is in boston in june but is a jazz concert, it should get a LLM score of 0.
You should pay attention to both OBJECTIVE and SUBJECTIVE exclusions, for example if the user says "I want to go to a chill rock concert, not too crowded", if an event has its description saying "REAL LOUD ROCK CONCERT, NOT FOR THE FAINT OF HEART", even if it matches the rock genre criteria, it should get a score of 0 because of the subjective "not too crowded" exclusion match.



User query:
{query}

Intent parsed:
{intent}

CRITICAL OUTPUT INSTRUCTIONS:
- You MUST evaluate and return a score for EVERY single event in the list below.
- Do not skip any event. If there are 20 events, your "events" array MUST contain exactly 20 JSON objects.
- Use the exact integer index provided in the bracket (e.g., [0], [1], [2]...) as the "index" field.

Return JSON only in this exact format structure:
{{
  "events": [    
    {{"index": 0, "objective_score": 4, "subjective_score": 2, "total_score": 6}},
    {{"index": 1, "objective_score": 6, "subjective_score": 3, "total_score": 9}},
    {{"index": 2, "objective_score": 1, "subjective_score": 1, "total_score": 2}}
    // ... continue for ALL items up to the last event index
  ]
}}

Events list:
{events_text}
"""

    for e in candidates:
        e["llm_score"] = 0
        e["llm_objective"] = 0
        e["llm_subjective"] = 0

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            response_format={"type":"json_object"}
        )

        response_content = json.loads(response.choices[0].message.content)
        llm_events = response_content.get("events", [])

        # Map scores dynamically using index matching
        for item in llm_events:
            idx = item.get("index")
            if idx is not None and idx < len(candidates):
                obj = item.get("objective_score", 0)
                sub = item.get("subjective_score", 0)
                total = item.get("total_score", obj + sub)
                
                candidates[idx]["llm_score"] = total
                candidates[idx]["llm_objective"] = obj
                candidates[idx]["llm_subjective"] = sub
                
                # Normalization factor added to the final score matrix
                candidates[idx]["score"] += float(total) / 10.0

    except Exception as error:
        print(f"⚠️ Error during LLM reranking: {error}. Falling back to initial algorithmic scores.")
        for e in candidates:
            e["llm_score"] = 0
            e["llm_objective"] = 0
            e["llm_subjective"] = 0

    final_sorted_list = sorted(candidates, key=lambda x: (x["criteria_count"], x["score"]), reverse=True)
    return final_sorted_list[:8]