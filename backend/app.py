from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from flask_cors import CORS
import os

# Download tokenizer
nltk.download('punkt')

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)  # Allow all origins — adjust if you want stricter security

def extractive_summarize(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text  # If text too short, just return it

    # TF-IDF & cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(vectors)

    # Rank sentences by importance
    scores = similarity_matrix.sum(axis=1)
    ranked_indices = np.argsort(scores)[-num_sentences:]
    ranked_indices = sorted(ranked_indices)

    # Join selected sentences into summary
    summary = ' '.join([sentences[i] for i in ranked_indices])
    return summary

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
    except Exception:
        return jsonify({"summary": "⚠️ Invalid JSON format."}), 400

    if not text:
        return jsonify({"summary": "⚠️ No text provided."}), 400

    try:
        summary = extractive_summarize(text)
        return jsonify({"summary": summary}), 200
    except Exception as e:
        return jsonify({"summary": f"⚠️ Failed to summarize: {str(e)}"}), 500

# Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5050))
app.run(host="0.0.0.0", port=port, debug=True)
