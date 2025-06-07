from flask import Flask, request, jsonify
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import google.generativeai as genai

# ----------------------------
# Preprocessing & Setup
# ----------------------------
def preprocess(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

print("🚀 Loading model and FAISS index...")
model_st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
index = faiss.read_index("vector_index.faiss")

with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
texts = metadata["texts"]
ids = metadata["ids"]

# Gemini setup
genai.configure(api_key="AIzaSyAii5vu6WjFbMwu4t0gkM0THbryynkgUuk")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)

@app.route("/rag", methods=["POST"])
def rag():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    # Step 1: Vectorize the query
    query_clean = preprocess(query)
    query_vec = model_st.encode([query_clean], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    # Step 2: FAISS similarity search
    D, I = index.search(query_vec, k=5)
    context_chunks = [texts[idx] for idx in I[0]]
    context = "\n\n".join(context_chunks)

    # Step 3: Gemini expert prompt
    prompt = f"""
You are a knowledgeable and helpful assistant. A user has asked a question. Below are documents retrieved from a knowledge base that are relevant to the user's query.

Your task is to read the documents and provide a clear, accurate, and helpful answer. If the answer is not present in the documents, politely say that the information is not available.

---

📌 User Question:
{query}

📚 Relevant Documents:
{context}

---

Please provide a concise and informative answer based only on the above documents.
"""

    try:
        # Step 4: Generate response from Gemini
        response = gemini_model.generate_content(prompt)
        final_answer = response.text.strip()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "answer": final_answer,
        "top_chunks": context_chunks
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
