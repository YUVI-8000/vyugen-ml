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
from collections import OrderedDict

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

# Add repetition count to the dataset
def add_repetition_count(data, column_name="text"):
    """
    Add a column to show how many times each question is repeated.
    """
    data["Repetitions"] = data.groupby(column_name)[column_name].transform("count")
    return data

# Update the dataset with repetition counts
data = add_repetition_count(data, column_name="text")

# Load questions from the dataset
questions = data["text"].tolist()
repetitions = data["Repetitions"].tolist()  # Store repetition counts

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

    # Prepare the results list
    results = [
        {
            'question': questions[i],
            'score': score,
            'repetitions': repetitions[i]
        }
        for i, score in sorted_results
    ]

    # Remove duplicates based on 'question' field using OrderedDict
    unique_results = list(OrderedDict((result['question'], result) for result in results).values())

    # Total unique results
    total_results = len(unique_results)

    # Return the search results in JSON format
    return jsonify({'query': topic, 'results': unique_results, 'total_results': total_results})

if __name__ == '__main__':
    # Start the Flask server for API
    app.run(debug=True, host="127.0.0.1", port=8000)