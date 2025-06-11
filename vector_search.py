from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os # ファイルパスを扱うためにosモジュールを追加

# --- モデルとインデックス、メタデータのロード ---

# 軽量な多言語対応モデルを使用
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# FAISSインデックスとメタデータのファイルパスを設定
# スクリーンショットのファイル名に合わせて修正
FAISS_INDEX_PATH = "index.faiss"  # ここを修正
METADATA_PATH = "index.pkl"       # ここを修正

# インデックスとメタデータを読み込み
try:
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    print("FAISSインデックスとメタデータのロードに成功しました。")
except FileNotFoundError as e:
    print(f"エラー: {e} - FAISSインデックスまたはメタデータファイルが見つかりません。")
    print("ファイルが指定されたパスに存在し、デプロイ時に含まれているか確認してください。")
    faiss_index = None
    metadata = []
except Exception as e:
    print(f"FAISSインデックスまたはメタデータのロード中に予期せぬエラーが発生しました: {e}")
    faiss_index = None
    metadata = []

# --- ベクトル検索関数 ---
def search_similar_documents(query: str, top_k: int = 3, max_chars: int = 1000) -> str:
    if faiss_index is None:
        return "検索システムが現在利用できません。管理者にお問い合わせください。"

    query_vec = model.encode([query]).astype("float32")

    D, I = faiss_index.search(query_vec, top_k)

    docs = []
    for idx in I[0]:
        if idx != -1 and idx < len(metadata):
            docs.append(metadata[idx])

    combined_docs = "\n\n".join(docs)
    return combined_docs[:max_chars]

# --- 使用例（テスト用、LINEbotのコードからは不要） ---
# if __name__ == "__main__":
#     # ... (テストコード) ...
