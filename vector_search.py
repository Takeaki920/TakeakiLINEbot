import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔽 index.faiss と index.pkl にあわせてパスを修正
index = faiss.read_index("index.faiss")
with open("index.pkl", "rb") as f:
    documents = pickle.load(f)

def search_similar_documents(query, top_k=3):
    embedding = model.encode([query])
    distances, indices = index.search(embedding, top_k)

    # 🔽 ここからログ出力追加
    print("🔍 クエリ:", query)
    print("🔍 検索結果インデックス:", indices)
    print("🔍 距離スコア:", distances)

    results = [documents[i] for i in indices[0] if i < len(documents)]

    if not results:
        print("⚠ 検索結果が空です！")
    else:
        print("✅ 取得した文書:\n", "\n---\n".join(results))

    return "\n---\n".join(results)

