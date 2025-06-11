from flask import Flask, request, abort
import openai
import os
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from vector_search import search_similar_documents  # 書籍検索モジュールを読み込む

app = Flask(__name__)

# 環境変数からAPIキーなどを読み込み
openai.api_key = os.getenv("OPENAI_API_KEY")
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

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
    user_message = event.message.text

    # 書籍ベースの類似文書を検索（FAISS）
    relevant_docs = search_similar_documents(user_message)

    # GPTへのプロンプトを構築（人格＋参考文献）
    prompt = f"""あなたは『明るい未来は和の心から』などの著者である晴田武陽のように話します。
和の心、楽しさ、思いやりを大切にし、誰にでもわかりやすく丁寧に応答します。
以下の参考文献に基づいて、質問に答えてください。

参考文献:
{relevant_docs}

質問:
{user_message}
"""

    # GPTで応答を生成
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    ai_reply = response.choices[0].message["content"]
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=ai_reply)
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
