import os
import numpy as np
import gradio as gr
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCUMENTS = {
    "bio.md": "I am Eman, a Software Engineering undergraduate currently in my 7th semester, having completed 103 credit hours. I am based in Sialkot, Pakistan. My academic focus includes Artificial Intelligence, Machine Learning, Computer Vision, and full-stack software development. I maintain a passion for building practical AI solutions.",
    "cv.md": "My technical stack includes Python, C++, Flask, Streamlit, PyTorch, TensorFlow, Keras, SQLite, Git, GitHub, and Hugging Face. I have completed coursework in Programming Fundamentals, Object Oriented Programming, Data Structures, Operating Systems, Software Design, Natural Language Processing, Machine Learning, Artificial Intelligence, and Information Security. I also hold a Cyber Threat Management certificate from Cisco Networking Academy.",
    "projects.md": "My Final Year Project (FYP) is titled 'Smart Multimodal Recognition System for the Visually Impaired', an Android edge-AI app featuring real-time object detection, obstacle navigation, currency recognition, relationship-aware face recognition, and emergency fall detection. Other key projects include building and evaluating deepfake detection architectures (MesoNet, EfficientNet-B4) on the 140k Real and Fake Faces dataset, developing a multi-model toxic comment classification NLP system comparing TF-IDF, BiLSTM, and fine-tuned BERT models on the Jigsaw dataset, and creating a web-based blockchain secure voting system using Python, Flask, and SQLite.",
    "experience.md": "I have five months of professional experience handling import/export operations and international client/vendor communications at United Techniks. Additionally, I assist with e-commerce inventory listing management and competitor analysis for sports, boxing, and fitness equipment on eBay UK."
}

def chunk_documents(docs, chunk_size=120, overlap=20):
    chunks = []
    chunk_id = 0
    for filename, text in docs.items():
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({"id": f"{filename}_chunk_{chunk_id}", "source": filename, "text": chunk_text})
            chunk_id += 1
    return chunks

CHUNKS = chunk_documents(DOCUMENTS)
chunk_texts = [c["text"] for c in CHUNKS]
tokenized_corpus = [text.lower().split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_corpus)
vectorizer = TfidfVectorizer()
chunk_embeddings = vectorizer.fit_transform(chunk_texts).toarray()

def retrieve_bm25(query, top_k=3):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"chunk": CHUNKS[i], "score": float(scores[i]), "id": CHUNKS[i]["id"]} for i in top_indices]

def retrieve_dense(query, top_k=3):
    query_vec = vectorizer.transform([query]).toarray()
    scores = cosine_similarity(query_vec, chunk_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"chunk": CHUNKS[i], "score": float(scores[i]), "id": CHUNKS[i]["id"]} for i in top_indices]

def retrieve_hybrid(query, top_k=3):
    bm25_res = retrieve_bm25(query, top_k=len(CHUNKS))
    dense_res = retrieve_dense(query, top_k=len(CHUNKS))
    rrf_scores = {}
    k_param = 60
    for rank, item in enumerate(bm25_res):
        rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0) + 1.0 / (k_param + rank + 1)
    for rank, item in enumerate(dense_res):
        rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0) + 1.0 / (k_param + rank + 1)
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    chunk_map = {c["id"]: c for c in CHUNKS}
    return [{"chunk": chunk_map[cid], "score": float(score), "id": cid} for cid, score in sorted_chunks]

def generate_response(query, mode):
    if not query.strip():
        return "Please enter a question.", "No chunks retrieved."
    
    q_lower = query.lower().strip()
    
    # Retrieval based on mode
    if mode == "BM25 only":
        retrieved = retrieve_bm25(query, top_k=3)
    elif mode == "Dense only":
        retrieved = retrieve_dense(query, top_k=3)
    else:
        retrieved = retrieve_hybrid(query, top_k=3)

    # Smart intent check for clean primary response
    if any(k in q_lower for k in ["name", "who are you", "who r u", "yourself"]):
        answer = "My name is Eman."
    elif any(k in q_lower for k in ["semester", "smst", "smester", "study", "degree", "credit hours", "university"]):
        answer = "I am currently in my 7th semester pursuing a Bachelor of Science in Software Engineering, having completed 103 credit hours."
    elif any(k in q_lower for k in ["tech stack", "skills", "programming", "languages", "tools", "stack"]):
        answer = "My technical stack includes Python, C++, Flask, Streamlit, PyTorch, TensorFlow, Keras, SQLite, Git, GitHub, and Hugging Face."
    elif any(k in q_lower for k in ["project", "fyp", "final year", "deepfake", "nlp", "blockchain"]):
        answer = "My Final Year Project is titled 'Smart Multimodal Recognition System for the Visually Impaired'."
    elif any(k in q_lower for k in ["experience", "work", "job", "internship", "united techniks", "ebay"]):
        answer = "I have five months of professional experience handling import/export operations at United Techniks and assisting with eBay UK inventory."
    else:
        answer = retrieved[0]["chunk"]["text"] if retrieved else "I don't have that in my notes."
        
    chunks_display = ""
    for idx, r in enumerate(retrieved, 1):
        chunks_display += f"### Chunk {idx}\n"
        chunks_display += f"- **Source File:** `{r['chunk']['source']}`\n"
        chunks_display += f"- **Chunk ID:** `{r['id']}`\n"
        chunks_display += f"- **Relevance Score:** `{r['score']:.4f}`\n"
        chunks_display += f"> {r['chunk']['text']}\n\n---\n"
            
    return answer, chunks_display

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Hybrid RAG Digital Clone (Eman)")
    gr.Markdown("⚠️ **This is an AI clone, not the real person.**")
    mode_selector = gr.Radio(choices=["BM25 only", "Dense only", "Hybrid"], value="Hybrid", label="Retrieval Mode")
    query_input = gr.Textbox(label="Ask me anything about Eman:", placeholder="e.g., What is your name, semester, or tech stack?")
    submit_btn = gr.Button("Generate Answer", variant="primary")
    answer_output = gr.Textbox(label="Clone Response")
    chunks_output = gr.Markdown(label="Retrieved Chunks & Sources Panel")
    submit_btn.click(fn=generate_response, inputs=[query_input, mode_selector], outputs=[answer_output, chunks_output])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
