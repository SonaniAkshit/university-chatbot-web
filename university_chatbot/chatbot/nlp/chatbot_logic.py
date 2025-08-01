import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load responses from JSON file
def load_responses():
    with open(os.path.join(os.path.dirname(__file__), 'responses.json'), 'r') as file:
        return json.load(file)

# NLP-based intent matching
def get_response(query, responses):
    # Prepare documents for TF-IDF
    documents = [query.lower()]
    intent_keys = []
    for intent, data in responses.items():
        for keyword in data['keywords']:
            documents.append(keyword.lower())
            intent_keys.append(intent)

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute cosine similarity between query (first document) and keywords
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Find the intent with the highest similarity
    if len(similarities) == 0 or max(similarities) < 0.1:  # Threshold for no match
        return "Sorry, I can only answer questions about the admission process. Try asking about eligibility, application process, deadlines, or documents!"
    
    max_index = np.argmax(similarities)
    best_intent = intent_keys[max_index]
    return responses[best_intent]['response']