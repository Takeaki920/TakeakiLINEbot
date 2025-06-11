from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# 軽量モデルを使用（すでに生成済みのベクトルと互換性がある場合）
model = SentenceTransformer("all-MiniLM-L6-v2")

# インデックスとメタデータを読み込み
faiss_index = faiss.read_index("faiss_index.index")
with open("faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def search_similar_documents(query, top_k=3, max_chars=1000):
    query_vec = model.encode([query])
    D, I = faiss_index.search(np.array(query_vec).astype("float32"), top_k)
    
    docs = []
    for idx in I[0]:
        if idx < len(metadata):
            docs.append(metadata[idx])

    combined = "\n\n".join(docs)
    return combined[:max_chars] 
