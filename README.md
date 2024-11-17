# ML-Based Question Retrieval System

This repository contains the code for a **Flask API** that allows users to retrieve questions based on a specific topic using Machine Learning techniques like **TF-IDF Vectorization** and **BM25 Ranking**. The project integrates preprocessing, data ranking, and similarity scoring to deliver relevant questions from a dataset. The API is designed for easy deployment and accessibility, with CORS enabled for cross-origin requests.

---

## Features

- **TF-IDF Vectorization**: Used for computing similarity between topics and dataset questions.
- **BM25 Ranking**: Provides additional relevance scoring for better results.
- **Text Preprocessing**: Includes stemming and stopwords removal for cleaner data processing.
- **Dynamic Dataset**: Fetches a dataset from a given URL to allow flexibility in data management.
- **Flask API**: A lightweight, RESTful service for question retrieval.
- **CORS Enabled**: Supports cross-origin requests for easy integration with frontends.
- **Environment Variables**: Configurable API parameters using `.env`.

---

## Project Workflow

1. **Dataset Handling**: 
   - Fetches a dataset from a URL provided in the `.env` file.
   - Preprocesses the dataset by removing duplicates and cleaning the text data.
2. **Model Preparation**: 
   - Generates question vectors using **TF-IDF Vectorization**.
   - Creates a **BM25** model for ranking tokenized questions.
3. **API Endpoints**: 
   - `/ml-search`: Takes a topic as input and retrieves questions with similarity scores.
   - `/`: A simple home route to verify the API is running.
4. **Deployment-Ready**:
   - Configurable host and port through environment variables.

---

## Prerequisites

Before you begin, ensure you have the following:

1. **Python 3.7+**
2. **Flask**: Lightweight web framework.
3. **NLTK**: For text preprocessing and stemming.
4. **scikit-learn**: For TF-IDF and similarity computation.
5. **rank-bm25**: For BM25 scoring.
6. **Flask-CORS**: To enable cross-origin requests.
7. **dotenv**: For environment variable management.

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Environment Setup

1. Create a `.env` file in the root directory and add the following variables:

   ```env
   DATASET_URL=<your_dataset_csv_url>
   FLASK_HOST=0.0.0.0  # Default is for global deployment; use 127.0.0.1 for local testing
   PORT=5000  # Change the port if required
   ```

2. Example `.env` file:

   ```env
   DATASET_URL=https://example.com/dataset.csv
   FLASK_HOST=127.0.0.1
   PORT=5000
   ```

---

## How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/<your_username>/<your_repository>.git
   cd <your_repository>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:

   ```bash
   python app.py
   ```

4. Access the application:
   - Open a browser and go to `http://127.0.0.1:5000` to verify the API is running.
   - Use an API testing tool like Postman or curl to test the `/ml-search` endpoint.

---

## API Endpoints

### 1. **Home Endpoint**
   - **Route**: `/`
   - **Method**: `GET`
   - **Description**: Returns a simple message to indicate the API is running.

   **Response**:
   ```json
   {
     "message": "Flask API for Question Retrieval is running!"
   }
   ```

### 2. **ML-Search Endpoint**
   - **Route**: `/ml-search`
   - **Method**: `POST`
   - **Description**: Accepts a topic and returns similar questions with similarity scores.

   **Request**:
   ```json
   {
     "topic": "Enter your topic here"
   }
   ```

   **Response**:
   ```json
   [
     {
       "similarity": 0.85,
       "question": "Sample question text 1"
     },
     {
       "similarity": 0.72,
       "question": "Sample question text 2"
     }
   ]
   ```

   **Error Response**:
   ```json
   {
     "error": "Topic is required."
   }
   ```

---

## Deployment

### Deploy Locally
1. Set the `FLASK_HOST` to `127.0.0.1` in your `.env` file.
2. Run the app with the `python app.py` command.

### Deploy Globally
1. Set the `FLASK_HOST` to `0.0.0.0` in your `.env` file.
2. Choose a deployment platform (e.g., Railway, AWS, Heroku, or DigitalOcean).
3. Upload your project, configure the `.env` variables, and set the deployment parameters.

---

## Preprocessing Steps

1. **Stemming**: Uses the Porter Stemmer to reduce words to their root forms.
2. **Stopwords Removal**: Filters out common words like *the*, *is*, *and* using NLTK's stopwords list.
3. **Vectorization**:
   - Converts text to numerical vectors using **TF-IDF**.
   - Tokenizes text for use in **BM25** ranking.

---

## Troubleshooting

- **Dataset Loading Issues**:
  - Ensure the `DATASET_URL` is a valid CSV file URL.
  - Use tools like Postman to verify the dataset's accessibility.

- **Environment Variables**:
  - Confirm the `.env` file exists and contains correct key-value pairs.

- **Module Errors**:
  - Reinstall dependencies with `pip install -r requirements.txt`.

- **BM25 Error**:
  - Check the tokenized question list for consistency in preprocessing.

---

## Future Enhancements

1. Add user authentication for secured API usage.
2. Extend preprocessing to handle additional text cleaning (e.g., lemmatization).
3. Integrate database support for dynamic dataset updates.
4. Build a front-end interface for user-friendly interactions.

---

## Contributing

Feel free to fork the repository and submit a pull request. Suggestions and bug reports are welcome!

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Let me know if you'd like additional sections or clarifications!