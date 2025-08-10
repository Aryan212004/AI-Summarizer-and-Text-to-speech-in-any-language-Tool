from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from flask_cors import CORS

# Download tokenizer
nltk.download('punkt')

app = Flask(__name__)
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

if __name__ == "__main__":
    # Use host='0.0.0.0' if you want external devices to access it
    app.run(debug=True, port=5050)
