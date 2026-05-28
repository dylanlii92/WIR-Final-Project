import json
from groq_service import client
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# -------------------------
# LOAD PRECOMPUTED DATA
# -------------------------

DOCS = np.load("docs.npy", allow_pickle=True)
EMBEDDINGS = np.load("embeddings.npy")
METADATA = np.load("metadata.npy", allow_pickle=True)

if len(DOCS) == 0:
    raise ValueError("docs.npy is empty")

# -------------------------
# MODEL (for query embedding only)
# -------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# TOKENIZATION
# -------------------------

STOP_WORDS = {
    "music", "event", "show", "tour",
    "in", "near", "the", "a", "an",
    "for", "i", "want", "to", "see",
    "find", "me"
}

def tokenize(text: str):
    return [
        t for t in text.lower().split()
        if t not in STOP_WORDS
    ]

# -------------------------
# BM25 INDEX
# -------------------------

BM25 = BM25Okapi([tokenize(doc) for doc in DOCS])

# -------------------------
# MAIN SEARCH
# -------------------------

def search_events(query, intent):

    retrieval_query = query.lower().strip()

    if not retrieval_query:
        return []

    query_tokens = tokenize(retrieval_query)

    # 1. Calcul et normalisation du score BM25 (0.0 à 1.0)
    bm25_scores = BM25.get_scores(query_tokens)
    max_bm25 = np.max(bm25_scores) if len(bm25_scores) > 0 else 0
    if max_bm25 > 0:
        bm25_scores = bm25_scores / max_bm25

    # 2. Calcul du score sémantique (Dense Embedding)
    query_emb = model.encode(
        retrieval_query,
        normalize_embeddings=True
    )
    emb_scores = np.dot(EMBEDDINGS, query_emb)

    all_scored_events = []

    # 3. Boucle de scoring hybride + comptage des critères
    for i in range(len(DOCS)):

        bm25_part = float(bm25_scores[i])
        emb_part = float(emb_scores[i])
        base_score = 0.75 * bm25_part + 0.25 * emb_part

        meta = METADATA[i]
        multiplier = 1.0
        bonus_details = {}
        
        # --- COMPTEUR DE CRITÈRES STRICTS ---
        criteria_met_count = 0

        intent_cities = [c.lower() for c in intent.get("cities", [])]
        intent_artists = [a.lower() for a in intent.get("artists", [])]
        intent_genres = [g.lower() for g in intent.get("genres", [])]
        intent_subgenres = [s.lower() for s in intent.get("subgenres", [])]
        
        time_intent = intent.get("time", {})
        rel_time = time_intent.get("relative", "").lower() if isinstance(time_intent, dict) else ""
        abs_time = time_intent.get("absolute", "").lower() if isinstance(time_intent, dict) else ""

        # Check Ville
        city_matched = False
        for city in intent_cities:
            if city in meta["city"].lower():
                multiplier += 0.5
                bonus_details["city_match"] = 0.5
                city_matched = True
        if city_matched:
            criteria_met_count += 1

        # Check Artiste
        artist_matched = False
        for artist in intent_artists:
            if artist in meta["artist"].lower():
                multiplier += 1.0
                bonus_details["artist_match"] = 1.0
                artist_matched = True
        if artist_matched:
            criteria_met_count += 1

        # Check Genre
        genre_matched = False
        for genre in intent_genres:
            if genre in (meta["genre"] + " " + meta["subgenre"]).lower():
                multiplier += 0.3
                bonus_details["genre_match"] = 0.4
                genre_matched = True
        if genre_matched:
            criteria_met_count += 1

        # Check Subgenre
        subgenre_matched = False
        for subgenre in intent_subgenres:
            if subgenre in (meta["genre"] + " " + meta["subgenre"]).lower():
                multiplier += 0.3
                bonus_details["subgenre_match"] = 0.4
                subgenre_matched = True
        if subgenre_matched and not genre_matched: # Évite de double-compter si genre et subgenre match la même chose
            criteria_met_count += 1

        # Check Date / Temps
        event_date = meta["date"].lower()
        time_matched = False
        if abs_time and (abs_time in event_date or abs_time in DOCS[i]):
            multiplier += 0.5
            bonus_details["absolute_time_match"] = 0.3
            time_matched = True
        elif rel_time and (rel_time in event_date or rel_time in DOCS[i]):
            multiplier += 0.3
            bonus_details["relative_time_match"] = 0.2
            time_matched = True
        if time_matched:
            criteria_met_count += 1

        # Bonus secondaires (Mots-clés, Humeurs) -> Ne comptent pas comme critères stricts mais boostent le score
        for keyword in intent.get("keywords", []):
            if keyword.lower() in DOCS[i]:
                multiplier += 0.1
                bonus_details[f"keyword_{keyword}"] = 0.1

        for mood in intent.get("mood", []):
            if mood.lower() in DOCS[i]:
                multiplier += 0.2
                bonus_details["mood_match"] = 0.2

        final_score = float(base_score * multiplier)

        all_scored_events.append({
            "id": i,
            "score": final_score,
            "criteria_count": criteria_met_count, # Stockage du nombre de critères
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
            "venue": meta["venue"],
            "genre": meta["genre"],
            "subgenre": meta["subgenre"],
            "description": meta["description"],
            "image": meta["image"],
            "url": meta["url"]
        })

    # --- TRI PAR TIERS STRICTS (Ta nouvelle méthodologie) ---
    # On trie d'ABORD par le nombre de critères satisfaits (descending)
    # PUIS en cas d'égalité par le score de pondération final (descending)
    all_scored_events.sort(key=lambda x: (x["criteria_count"], x["score"]), reverse=True)

    # On sélectionne notre Top 20 de candidats pour le LLM basé sur ce tri ultra robuste
    candidates = all_scored_events[:20]

    # 4. Re-ranking par le LLM (uniquement sur les meilleurs Tiers disponibles)
    reranked = rerank_events(query, intent, candidates)

    # --- DETAILED SCORE DISPLAY IN CONSOLE ---
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


def rerank_events(query, intent, candidates):
    if not candidates:
        return []

    events_text = ""
    for i, e in enumerate(candidates):
        events_text += f"""
[{i}]
Title: {e['name']}
Artist: {e['artist']}
Genre: {e['genre']}
Subgenre: {e['subgenre']}
City: {e['city']}
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

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            response_format={"type":"json_object"}
        )

        scores = json.loads(response.choices[0].message.content)["scores"]

        for i, e in enumerate(candidates):
            if i < len(scores):
                llm_score_normalized = scores[i] / 10.0
                e["llm_score"] = scores[i]
                e["score"] += llm_score_normalized
            else:
                e["llm_score"] = 0
                
    except Exception as e:
        print("Error during LLM reranking, keeping initial scores:", e)
        for e in candidates:
            e["llm_score"] = 0

    return sorted(candidates, key=lambda x: (x["criteria_count"], x["score"]), reverse=True)[:8]