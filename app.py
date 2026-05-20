from flask import Flask, render_template, jsonify, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask")
def ask():

    query = request.args.get("q", "")

    fake_results = [
        {
            "name": "Sunset Chill Festival",
            "date": "2026-07-18",
            "venue": "LA Downtown Arena",
            "city": "Los Angeles",
            "image": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f",
            "url": "https://www.ticketmaster.com",
            "explanation": f"This event matches '{query}' because it includes chill and indie music vibes."
        },
        {
            "name": "Midnight EDM Experience",
            "date": "2026-08-03",
            "venue": "Miami Beach Stage",
            "city": "Miami",
            "image": "https://images.unsplash.com/photo-1501386761578-eac5c94b800a",
            "url": "https://www.ticketmaster.com",
            "explanation": "This event is popular for energetic electronic music performances."
        }
    ]

    return jsonify({
        "results": fake_results
    })


if __name__ == "__main__":
    app.run(debug=True)