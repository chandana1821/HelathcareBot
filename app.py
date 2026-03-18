from flask import Flask, request, jsonify
from query import search_documents, generate_answer, format_results

app = Flask(__name__)

@app.route("/")
def home():
    return "Healthcare Chatbot Running Successfully 🚀"

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("message")

    matches = search_documents(user_query)

    if not matches:
        return jsonify({"reply": "No relevant information found."})

    context = format_results(matches)
    answer = generate_answer(user_query, context)

    return jsonify({"reply": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)