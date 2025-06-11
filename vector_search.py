import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# モデルのロード（軽量で高速な文書ベクトルモデル）
model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISSインデックスと対応文書の読み込み（app.pyと同じ階層に置く）
index = faiss.read_index("index.faiss")
with open("index.pkl", "rb") as f:
    documents = pickle.load(f)

def search_similar_documents(query, top_k=3):
    """ユーザーのクエリに似た文書を上位から取得"""
    embedding = model.encode([query])
    distances, indices = index.search(np.array(embedding).reshape(1, -1), top_k)

    # 🔍 ログ出力（Renderでも確認可能）
    print("🔍 クエリ:", query)
    print("🔍 検索インデックス:", indices)
    print("🔍 類似スコア:", distances)

    results = [documents[i] for i in indices[0] if i < len(documents)]

    if not results:
        print("⚠ 検索結果なし（results 空）")
        return "（参考文献が見つかりませんでした）"

    print("✅ 取得文書:\n", "\n---\n".join(results))
    return "\n---\n".join(results)
