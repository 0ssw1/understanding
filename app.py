from flask import Flask, request, jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

app = Flask(__name__)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_embeddings(text_list):
    embeddings = []
    for text in text_list:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        response_data = response.to_dict()
        embeddings.append(response_data['data'][0]['embedding'])
    return embeddings

def load_embeddings(file_path):
    return np.load(file_path, allow_pickle=True).item()

def search_similar_videos(query, data, top_k=10):
    query_embedding = get_embeddings([query])[0]
    similarities = cosine_similarity([query_embedding], data["embeddings"])[0]
    top_results = np.argsort(-similarities)[:top_k]
    return [(data["titles"][idx], data["links"][idx], similarities[idx]) for idx in top_results]

data = load_embeddings('video_embeddings.npy')

@app.route('/search', methods=['POST'])
def search():
    content = request.json
    query = content.get('query')
    top_k = content.get('top_k', 10)
    results = search_similar_videos(query, data, top_k=top_k)
    return jsonify(results)

if __name__ == '__main__':
    app.run()
