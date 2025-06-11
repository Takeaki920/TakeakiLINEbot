import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# モデルとデータの読み込み
model = SentenceTransformer("all-MiniLM-L6-v2")

# ベクトルインデックスとメタデータ（文書の対応情報）を読み込む
index = faiss.read_index("faiss_index.index")  # faiss保存先
with open("faiss_metadata.pkl", "rb") as f:
    documents = pickle.load(f)

def search_similar_documents(query, top_k=3):
    """ユーザーの質問に対して類似した文書を返す"""
    embedding = model.encode([query])
    distances, indices = index.search(embedding, top_k)
    results = [documents[i] for i in indices[0] if i < len(documents)]
    return "\n---\n".join(results)
