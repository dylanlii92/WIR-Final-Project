from flask import Flask, render_template, jsonify, request

from database_service import search_events
from groq_service import extract_search_intent

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask")
def ask():
    user_query = request.args.get("q", "")

    if not user_query:
        return jsonify({
            "results": []
        })

    intent = extract_search_intent(user_query)

    search_keyword = intent.get("keyword") or user_query
    search_city = intent.get("city", "")
    search_genres = intent.get("genres", [])

    results = search_events(
    query=user_query,
    city=intent.get("city", ""),
    genres=intent.get("genres", [])
)

    for event in results:
        event["explanation"] = (
            "This event matches your search based on its title, "
            "genre, location, venue, or description."
        )

    return jsonify({
        "query": user_query,
        "intent": intent,
        "results": results
    })


if __name__ == "__main__":
    app.run(debug=True)