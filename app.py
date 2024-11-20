import re
import os
import requests
import pandas as pd
from io import StringIO
from flask_cors import CORS
from rank_bm25 import BM25Plus
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables from .env file
load_dotenv()

# Fetch GitHub raw URL for the dataset
data_file_path = os.getenv('DATASET_URL')
if not data_file_path:
    raise ValueError("DATASET_URL is missing in the .env file")

# Fetch the dataset from the URL
response = requests.get(data_file_path)
if response.status_code != 200:
    raise Exception(f"Failed to fetch dataset. HTTP status code: {response.status_code}")

# Load the dataset into a Pandas DataFrame
data = pd.read_csv(StringIO(response.text))
print("Dataset loaded successfully.")

# Ensure the 'text' column exists in the dataset
if 'text' not in data.columns:
    raise ValueError("'text' column is missing in the dataset.")

# Extract unique questions
questions = list(dict.fromkeys(data["text"].dropna().tolist()))  # Drop NaN and keep unique entries

# Initialize TF-IDF Vectorizer with bigrams
vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2))
question_vectors = vectorizer.fit_transform(questions)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for integration with external services

@app.route('/search', methods=['GET'])
def search():
    """
    Search endpoint to retrieve questions based on a query.
    """
    # Retrieve the query from request parameters
    topic = request.args.get('query', '').strip().lower()
    if not topic:
        return jsonify({'error': 'Query parameter is required.'}), 400

    # Filter questions based on the topic (case-insensitive)
    filtered_questions = [
        (i, q) for i, q in enumerate(questions) if re.search(rf"\b{re.escape(topic)}\b", q, re.IGNORECASE)
    ]
    if not filtered_questions:
        return jsonify({'query': topic, 'results': [], 'total_results': 0})

    # Extract indices and texts of the filtered questions
    filtered_indices = [i for i, _ in filtered_questions]
    filtered_texts = [q for _, q in filtered_questions]

    # Compute TF-IDF vectors and initialize BM25 for filtered questions
    filtered_vectors = vectorizer.transform(filtered_texts)
    tokenized_questions = [re.findall(r'\b\w+\b', q.lower()) for q in filtered_texts]
    bm25 = BM25Plus(tokenized_questions)

    # Tokenize the topic and compute BM25 scores
    tokenized_topic = re.findall(r'\b\w+\b', topic)
    bm25_scores = bm25.get_scores(tokenized_topic)

    # Compute TF-IDF cosine similarity
    topic_vector = vectorizer.transform([topic])
    tfidf_scores = cosine_similarity(topic_vector, filtered_vectors).flatten()

    # Combine and rank results using both scores
    combined_scores = [
        (filtered_indices[i], max(tfidf_scores[i], bm25_scores[i])) for i in range(len(filtered_indices))
    ]
    ranked_results = sorted(combined_scores, key=lambda x: x[1], reverse=True)

    # Prepare JSON response
    results = [{'question': questions[i], 'score': round(score, 4)} for i, score in ranked_results]
    return jsonify({'query': topic, 'results': results, 'total_results': len(results)})

if __name__ == '__main__':
    # Fetch host and port dynamically for deployment
    host = os.getenv("FLASK_HOST", "0.0.0.0")  # Default to "0.0.0.0" for global
    port = int(os.getenv("PORT", 8000))        # Default to port 8000 for deployment
    app.run(host=host, port=port, debug=False) # Disable debug mode for production

