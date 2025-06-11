from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os # ファイルパスを扱うためにosモジュールを追加

# --- モデルとインデックス、メタデータのロード ---
# LINEbotアプリ起動時に一度だけロードされるように、関数の外で定義します。
# これにより、リクエストごとにモデルがロードされるのを防ぎ、効率的になります。

# 軽量な多言語対応モデルを使用
# 日本語を含む多言語に対応しており、メモリ効率も良いです。
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# FAISSインデックスとメタデータのファイルパスを設定
# Renderにデプロイする際、これらのファイルがPythonスクリプトと同じディレクトリ、
# または指定したサブディレクトリに存在するようにしてください。
# 例: 'data/faiss_index.index' のようにサブディレクトリを指定する場合は適宜修正してください。
FAISS_INDEX_PATH = "faiss_index.index"
METADATA_PATH = "faiss_metadata.pkl"

# インデックスとメタデータを読み込み
# ファイルが存在しない場合はエラーになるので、事前に作成し、デプロイ時に含めること
try:
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    print("FAISSインデックスとメタデータのロードに成功しました。")
except FileNotFoundError as e:
    print(f"エラー: {e} - FAISSインデックスまたはメタデータファイルが見つかりません。")
    print("ファイルが指定されたパスに存在し、デプロイ時に含まれているか確認してください。")
    # エラーハンドリング: ここでアプリを終了させるか、代替処理を行うか検討
    faiss_index = None
    metadata = []
except Exception as e:
    print(f"FAISSインデックスまたはメタデータのロード中に予期せぬエラーが発生しました: {e}")
    faiss_index = None
    metadata = []

# --- ベクトル検索関数 ---
def search_similar_documents(query: str, top_k: int = 3, max_chars: int = 1000) -> str:
    """
    ユーザーのクエリに基づいてFAISSインデックスから類似文書を検索し、結合して返します。

    Args:
        query (str): ユーザーからの検索クエリ。
        top_k (int): 検索する類似文書の数。
        max_chars (int): 返す結合された文書の最大文字数。

    Returns:
        str: 検索された類似文書を結合した文字列。
             インデックスがロードされていない場合はエラーメッセージを返します。
    """
    if faiss_index is None:
        return "検索システムが現在利用できません。管理者にお問い合わせください。"

    # クエリをモデルでベクトル化（float32型に変換）
    query_vec = model.encode([query]).astype("float32")

    # FAISSで類似文書を検索
    D, I = faiss_index.search(query_vec, top_k)
    # D: 距離（Distance）、I: インデックス（Index）

    docs = []
    # 検索結果のインデックス（I）を元に、メタデータから元の文書を取得
    for idx in I[0]:
        # FAISSの検索結果には-1が含まれることがあるので除外
        if idx != -1 and idx < len(metadata):
            # metadataは元の回答テキストなどが格納されているリストまたは辞書を想定
            docs.append(metadata[idx])

    # 検索された文書を結合し、最大文字数で切り詰める
    combined_docs = "\n\n".join(docs)
    return combined_docs[:max_chars]

# --- 使用例（テスト用、LINEbotのコードからは不要） ---
# if __name__ == "__main__":
#     # 実際にFAISSインデックスとメタデータファイルを作成してから実行してください
#     # 例: original_texts = ["これはテスト文章1です。", "これはテスト文章2です。日本語も含まれます。", ...]
#     #     embeddings = model.encode(original_texts).astype("float32")
#     #     index = faiss.IndexFlatL2(embeddings.shape[1])
#     #     index.add(embeddings)
#     #     faiss.write_index(index, FAISS_INDEX_PATH)
#     #     with open(METADATA_PATH, "wb") as f:
#     #         pickle.dump(original_texts, f)

#     if faiss_index is not None:
#         test_query = "LINEボットの作り方について知りたい"
#         result = search_similar_documents(test_query)
#         print(f"クエリ: '{test_query}'\n検索結果:\n{result}")

#         test_query_jp = "返品の方法を教えてください"
#         result_jp = search_similar_documents(test_query_jp)
#         print(f"\nクエリ: '{test_query_jp}'\n検索結果:\n{result_jp}")
#     else:
#         print("\n検索システムが初期化されていないため、テストを実行できません。")
