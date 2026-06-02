from flask import Flask, render_template, jsonify, request
from ranking_algorithm import search_events
from groq_service import extract_search_intent

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask")
def ask():
    user_query = request.args.get("q", "").strip()
    limit = get_positive_int_arg("limit", 6, max_value=24)
    offset = get_positive_int_arg("offset", 0, max_value=500)

    if not user_query:
        return jsonify({
            "results": [],
            "limit": limit,
            "offset": offset,
            "has_more": False
        })

    # INTENT EXTRACTION
    intent = extract_search_intent(user_query)

    if not isinstance(intent, dict) or "cities" not in intent:
        intent = {
            "titles": [],
            "cities": [],
            "states": [],
            "artists": [],
            "genres": [],
            "subgenres": [],
            "mood": [],
            "keywords": [user_query.lower()],
            "event_type": [],
            "attending": [],
            "time": {"start": "", "end": ""}
        }

    # RANKING
    page = search_events(user_query, intent, limit=limit, offset=offset)
    results = page["results"]

    # RESPONSE
    for r in results:
        r["explanation"] = f"score={r['score']:.2f}"

    return jsonify({
        "query": user_query,
        "intent": intent,
        "limit": limit,
        "offset": offset,
        "next_offset": page["next_offset"],
        "has_more": page["has_more"],
        "results": results
    })

def get_positive_int_arg(name, default, max_value):
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default

    return max(0, min(value, max_value))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
