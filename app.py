from flask import Flask, request, render_template_string
import joblib
from preprocess import clean_text

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Review Authenticity Analyzer</title>

<style>
@keyframes neonMove1 {
    0% { transform: translate(-120%, 120%) rotate(25deg); }
    100% { transform: translate(120%, -120%) rotate(25deg); }
}
@keyframes neonMove2 {
    0% { transform: translate(120%, 120%) rotate(-30deg); }
    100% { transform: translate(-120%, -120%) rotate(-30deg); }
}

body {
    margin: 0;
    height: 100vh;
    background: #060818;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Segoe UI', sans-serif;
    color: white;
}

.neon {
    position: absolute;
    width: 220%;
    height: 3px;
    opacity: 0.6;
}
.neon.one {
    background: linear-gradient(90deg, transparent, #00f2ff, transparent);
    animation: neonMove1 6s linear infinite;
}
.neon.two {
    background: linear-gradient(90deg, transparent, #ff00ff, transparent);
    animation: neonMove2 8s linear infinite;
}

.card {
    position: relative;
    width: 480px;
    padding: 30px;
    border-radius: 18px;
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
    box-shadow: 0 0 30px rgba(0,255,255,0.25);
    z-index: 2;
    text-align: center;
}

textarea {
    width: 100%;
    height: 110px;
    margin-top: 15px;
    padding: 12px;
    border-radius: 12px;
    border: none;
    background: rgba(0,0,0,0.5);
    color: white;
    outline: none;
}

button {
    width: 100%;
    margin-top: 18px;
    padding: 12px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #00f2ff, #0066ff);
    color: black;
    font-weight: bold;
    font-size: 16px;
    cursor: pointer;
}

.result {
    margin-top: 15px;
    padding: 14px;
    border-radius: 12px;
    background: rgba(0,255,255,0.15);
    border: 1px solid rgba(0,255,255,0.5);
}

.confidence {
    margin-top: 6px;
    font-size: 14px;
}

.back-btn {
    background: transparent;
    border: 1px solid #00f2ff;
    color: #00f2ff;
}
.back-btn:hover {
    background: rgba(0,242,255,0.15);
}
</style>
</head>

<body>

<div class="neon one"></div>
<div class="neon two"></div>

<div class="card">

{% if not prediction %}
    <h2>⚡ Review Authenticity AI</h2>
    <p>Paste a review to analyze trust contribution</p>

    <form method="POST">
        <textarea name="review" placeholder="Paste the review here..." required></textarea>
        <button type="submit">Analyze Review</button>
    </form>

{% else %}
    <h2>📊 Analysis Result</h2>

    <div class="result">
        <strong>{{ prediction }}</strong>
        <div class="confidence">
            Trust Contribution: {{ weight }}<br>
            Confidence Score: {{ confidence }}%
        </div>
    </div>

    <form method="GET">
        <button class="back-btn" type="submit">← Analyze Another Review</button>
    </form>
{% endif %}

</div>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    confidence = None
    weight = None

    if request.method == 'POST':
        review = request.form['review']
        clean = clean_text(review)
        vec = vectorizer.transform([clean])

        probs = model.predict_proba(vec)[0]
        genuine_index = list(model.classes_).index("genuine")
        raw_confidence = probs[genuine_index] * 100

        words = len(review.split())
        if words < 6:
            raw_confidence -= 20
        elif words > 15:
            raw_confidence += 15

        raw_confidence = max(0, min(raw_confidence, 100))
        confidence = round(raw_confidence, 2)

        if raw_confidence >= 65:
            prediction = "High Authenticity"
            weight = "High Weight"
        elif raw_confidence >= 35:
            prediction = "Medium Authenticity"
            weight = "Moderate Weight"
        else:
            prediction = "Low Authenticity"
            weight = "Low Weight"

    return render_template_string(
        HTML,
        prediction=prediction,
        confidence=confidence,
        weight=weight
    )

if __name__ == '__main__':
    app.run(debug=True)
