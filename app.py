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

    if not user_query:
        return jsonify({"results": []})

    # -------------------------
    # INTENT EXTRACTION
    # -------------------------

    intent = extract_search_intent(user_query)

    if not isinstance(intent, dict):
        intent = {
            "keywords": [user_query],
            "artists": [],
            "genres": [],
            "subgenres": [],
            "city": ""
        }

    # -------------------------
    # RANKING
    # -------------------------

    results = search_events(user_query, intent)

    # -------------------------
    # RESPONSE
    # -------------------------

    for r in results:
        r["explanation"] = f"score={r['score']:.2f}"


    return jsonify({
        "query": user_query,
        "intent": intent,
        "results": results
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)