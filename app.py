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

# Fetch GitHub URL from the environment variable
data_file_path = os.getenv('DATASET_URL')

# Check if the URL is valid
if not data_file_path:
    raise ValueError("GitHub URL not found in the .env file")

# Fetch the dataset from the GitHub raw URL
response = requests.get(data_file_path)

# Ensure successful response (status code 200)
if response.status_code == 200:
    # Read the CSV content into a pandas dataframe
    data = pd.read_csv(StringIO(response.text))
    print("Dataset loaded successfully")
else:
    raise Exception(f"Failed to fetch dataset, HTTP status code {response.status_code}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for integration

# Reload the dataset (in case further processing is needed)
data = pd.read_csv(StringIO(response.text))

# Load unique questions from the dataset
questions = list(dict.fromkeys(data["text"].tolist()))

# TF-IDF Vectorizer with bigrams to represent the questions
vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2))
question_vectors = vectorizer.fit_transform(questions)

@app.route('/search', methods=['GET'])
def search():
    """
    Search endpoint to retrieve questions based on a topic query.
    This function handles the search logic using both BM25 and TF-IDF Cosine Similarity.
    """
    # Retrieve the query from request parameters
    topic = request.args.get('query', '').strip().lower()

    # Return an error if no query is provided
    if not topic:
        return jsonify({'error': 'No topic query provided'}), 400

    # Pre-filter questions based on topic
    filtered_questions = [
        (i, q) for i, q in enumerate(questions) if re.search(rf"\b{re.escape(topic)}\b", q, re.IGNORECASE)
    ]

    # If no questions match the topic, return empty results
    if not filtered_questions:
        return jsonify({'query': topic, 'results': [], 'total_results': 0})

    # Extract indices of the filtered questions
    filtered_indices = [i for i, _ in filtered_questions]
    filtered_question_texts = [q for _, q in filtered_questions]

    # Rebuild TF-IDF vectors and BM25 model for the filtered questions only
    filtered_vectors = vectorizer.transform(filtered_question_texts)
    tokenized_questions = [re.findall(r'\b\w+\b', q.lower()) for q in filtered_question_texts]
    bm25 = BM25Plus(tokenized_questions)
    tokenized_topic = re.findall(r'\b\w+\b', topic)
    bm25_scores = bm25.get_scores(tokenized_topic)

    # Compute TF-IDF Cosine Similarity for the filtered questions
    topic_vector = vectorizer.transform([topic])
    similarities = cosine_similarity(topic_vector, filtered_vectors).flatten()

    # Combine BM25 and TF-IDF similarities and rank the results
    combined_results = [
        (filtered_indices[i], max(similarities[i], bm25_scores[i]))  # Take the max score from both methods
        for i in range(len(filtered_indices))
    ]
    sorted_results = sorted(combined_results, key=lambda x: x[1], reverse=True)

    # Prepare the final results for the JSON response
    results = [{'question': questions[i], 'score': score} for i, score in sorted_results]
    total_results = len(results)

    # Return the search results in JSON format
    return jsonify({'query': topic, 'results': results, 'total_results': total_results})

if __name__ == '__main__':
    # Set host and port for local or global deployment
    host = os.getenv("FLASK_HOST", "0.0.0.0")  # "0.0.0.0" for global, "127.0.0.1" for local testing
    port = int(os.getenv("PORT", 5000))  # Default to port 5000 if no environment variable is set
    app.run(host=host, port=port, debug=True)
