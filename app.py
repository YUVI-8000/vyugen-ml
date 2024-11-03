from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Plus
import pandas as pd
import requests
from io import StringIO
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from flask_cors import CORS
from dotenv import load_dotenv
import os
import nltk

# Uncomment if stopwords need to be downloaded
# nltk.download('stopwords')

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize stemming and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocess text by stemming
def preprocess_text(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    return " ".join(stemmed_words)

# Load & preprocess data
def load_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = pd.read_csv(StringIO(response.text))
        data = data.drop(columns='Marks', errors='ignore').drop_duplicates(subset=['text'])
        print("DataFrame preview:")
        print(data.head())
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Fetch data URL from environment variables
data_url = os.getenv("DATASET_URL")
print("Dataset URL:", data_url)
data = load_data(data_url)

# Ensure questions variable is defined after loading data
questions = [preprocess_text(question) for question in data["text"]] if not data.empty else []

# Initialize TF-IDF vectorizer and transform data
vectorizer = TfidfVectorizer(stop_words=list(stop_words))
question_vectors = vectorizer.fit_transform(questions)

# Initialize BM25 model for additional ranking
tokenized_questions = [question.split() for question in questions]
bm25 = BM25Plus(tokenized_questions)

@app.route('/ml-search', methods=['POST'])
def find_similar():
    topic = request.json.get('topic')
    if not topic:
        return jsonify({"error": "Topic is required."}), 400

    # Preprocess and vectorize the topic
    preprocessed_topic = preprocess_text(topic)
    topic_vector = vectorizer.transform([preprocessed_topic])

    # Compute TF-IDF cosine similarity
    similarities = cosine_similarity(topic_vector, question_vectors).flatten()
    matching_indices = [i for i, similarity in enumerate(similarities) if similarity >= 0.2]

    if not matching_indices:
        return jsonify({"message": "No similar questions found."}), 404

    # Prepare result list
    results = [
        {"similarity": round(similarities[i], 2), "question": data["text"].iloc[i]} 
        for i in matching_indices
    ]

    return jsonify(results)

@app.route('/')
def home():
    return "Flask API for Question Retrieval is running!"

if __name__ == '__main__':
    # Set host and port for local or global deployment
    host = os.getenv("FLASK_HOST", "0.0.0.0")  # "0.0.0.0" for global, "127.0.0.1" for local testing
    port = int(os.getenv("PORT", 5000))  # Default to port 5000 if no environment variable is set
    app.run(host=host, port=port, debug=True)
