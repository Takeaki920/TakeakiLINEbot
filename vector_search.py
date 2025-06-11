import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import openai
from vector_search import search_similar_documents  # FAISS連携部分を別ファイルに分離

app = Flask(__name__)

# 環境変数からキーを読み込む
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_input = event.message.text

    # ベクトル検索（別ファイル vector_search.py に記述）
    relevant_docs = search_similar_documents(user_input)

    # GPTに渡すプロンプトを作成
    prompt = f"""
あなたは『明るい未来は和の心から』などの著者である晴田武陽のように話します。
以下の参考文献に基づいて、質問に対してできる限り正確に答えてください。

参考文献:
{relevant_docs}

質問:
{user_input}
"""

    # GPTで応答生成
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    answer = response.choices[0].message.content.strip()

    # LINEへ応答
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=answer)
    )

if __name__ == "__main__":
    app.run()
