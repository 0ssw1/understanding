import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests

# OpenAI API 클라이언트 설정
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 외부 스토리지에서 임베딩 파일 다운로드
def download_embeddings():
    url = os.getenv('EMBEDDINGS_URL')  # 외부 스토리지 서비스의 파일 URL
    response = requests.get(url)
    with open('/tmp/video_embeddings.npy', 'wb') as f:
        f.write(response.content)
    return np.load('/tmp/video_embeddings.npy', allow_pickle=True).item()

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

def search_similar_videos(query, data, top_k=5):
    # 쿼리 문장을 임베딩
    query_embedding = get_embeddings([query])[0]
    # 코사인 유사도 계산
    similarities = cosine_similarity([query_embedding], data["embeddings"])[0]
    # 유사도가 높은 순서대로 정렬
    top_results = np.argsort(-similarities)[:top_k]
    return [(data["titles"][idx], data["links"][idx], similarities[idx]) for idx in top_results]

# 임베딩 파일 다운로드 및 로드
data = download_embeddings()

# 예제 유사도 검색
query = "Learn Python"
results = search_similar_videos(query, data)

for result in results:
    print(f"Title: {result[0]}, Link: {result[1]}, Similarity: {result[2]:.4f}")
