import os
import numpy as np
import gradio as gr
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Knowledge Base Documents (Eman's Profile) ---
DOCUMENTS = {
    "bio.md": (
        "I am Eman, a Software Engineering undergraduate currently in my 7th semester, having completed 103 credit hours. "
        "I am based in Sialkot, Pakistan. My academic focus includes Artificial Intelligence, Machine Learning, Computer Vision, "
        "and full-stack software development. I maintain a passion for building practical AI solutions."
    ),
    "cv.md": (
        "My technical stack includes Python, C++, Flask, Streamlit, PyTorch, TensorFlow, Keras, SQLite, Git, GitHub, and Hugging Face. "
        "I have completed coursework in Programming Fundamentals, Object Oriented Programming, Data Structures, Operating Systems, "
        "Software Design, Natural Language Processing, Machine Learning, Artificial Intelligence, and Information Security. "
        "I also hold a Cyber Threat Management certificate from Cisco Networking Academy."
    ),
    "projects.md": (
        "My Final Year Project (FYP) is titled 'Smart Multimodal Recognition System for the Visually Impaired', an Android edge-AI app "
        "featuring real-time object detection, obstacle navigation, currency recognition, relationship-aware face recognition, and emergency fall detection. "
        "Other key projects include building and evaluating deepfake detection architectures (MesoNet, EfficientNet-B4) on the 140k Real and Fake Faces dataset, "
        "developing a multi-model toxic comment classification NLP system comparing TF-IDF, BiLSTM, and fine-tuned BERT models on the Jigsaw dataset, "
        "and creating a web-based blockchain secure voting system using Python, Flask, and SQLite."
    ),
    "experience.md": (
        "I have five months of professional experience handling import/export operations and international client/vendor communications at United Techniks. "
        "Additionally, I assist with e-commerce inventory listing management and competitor analysis for sports, boxing, and fitness equipment on eBay UK."
    ),
    "faq.md": (
        "Q: What is your name?\nA: My name is Eman.\n"
        "Q: What semester are you in?\nA: I am currently in my 7th semester studying Software Engineering.\n"
        "Q: What is your tech stack?\nA: I primarily use Python, PyTorch, TensorFlow, Flask, Streamlit, and Git for my development workflows.\n"
        "Q: Are you an AI clone?\n"
        "A: Yes, I am an AI digital clone of Eman, not the real person. I answer questions based only on my documented notes."
    )
}

# --- 2. Chunking & Indexing Setup ---
def chunk_documents(docs, chunk_size=150, overlap=30):
    chunks = []
    chunk_id = 0
    for filename, text in docs.items():
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "id": f"{filename}_chunk_{chunk_id}",
                "source": filename,
                "text": chunk_text
            })
            chunk_id += 1
    return chunks

CHUNKS = chunk_documents(DOCUMENTS)
chunk_texts = [c["text"] for c in CHUNKS]

# BM25 Setup
tokenized_corpus = [text.lower().split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_corpus)

# Dense Embedding Setup (MiniLM)
model = SentenceTransformer("all-MiniLM-L6-v2")
if os.path.exists("embeddings.npy"):
    chunk_embeddings = np.load("embeddings.npy")
else:
    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=False)
    np.save("embeddings.npy", chunk_embeddings)

# --- 3. Retrieval Functions ---
def retrieve_bm25(query, top_k=3):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "chunk": CHUNKS[idx],
            "score": float(scores[idx]),
            "id": CHUNKS[idx]["id"]
        })
    return results

def retrieve_dense(query, top_k=3):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "chunk": CHUNKS[idx],
            "score": float(scores[idx]),
            "id": CHUNKS[idx]["id"]
        })
    return results

def retrieve_hybrid(query, top_k=3):
    # Reciprocal Rank Fusion (RRF)
    bm25_res = retrieve_bm25(query, top_k=len(CHUNKS))
    dense_res = retrieve_dense(query, top_k=len(CHUNKS))
    
    rrf_scores = {}
    k_param = 60
    
    for rank, item in enumerate(bm25_res):
        cid = item["id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k_param + rank + 1)
        
    for rank, item in enumerate(dense_res):
        cid = item["id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k_param + rank + 1)
        
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    chunk_map = {c["id"]: c for c in CHUNKS}
    results = []
    for cid, score in sorted_chunks:
        results.append({
            "chunk": chunk_map[cid],
            "score": float(score),
            "id": cid
        })
    return results

# --- 4. Generation & Response Logic ---
def generate_response(query, mode):
    if mode == "BM25 only":
        retrieved = retrieve_bm25(query, top_k=3)
    elif mode == "Dense only":
        retrieved = retrieve_dense(query, top_k=3)
    else:
        retrieved = retrieve_hybrid(query, top_k=3)
        
    context_str = "\n\n".join([f"[{r['chunk']['source']} - {r['id']}]: {r['chunk']['text']}" for r in retrieved])
    
    # Grounded first-person generation logic
    q_lower = query.lower()
    answer = ""
    
    if any(k in q_lower for k in ["name", "who are you", "who r u"]):
        answer = "I am an AI digital clone of Eman, a 7th-semester Software Engineering undergraduate."
    elif any(k in q_lower for k in ["semester", "study", "degree"]):
        answer = "I am currently in my 7th semester pursuing a Bachelor of Science in Software Engineering, having completed 103 credit hours."
    elif any(k in q_lower for k in ["tech stack", "skills", "tools", "python"]):
        answer = "My tech stack includes Python, C++, Flask, Streamlit, PyTorch, TensorFlow, Keras, SQLite, Git, GitHub, and Hugging Face."
    elif any(k in q_lower for k in ["project", "fyp", "final year", "deepfake", "nlp"]):
        answer = "My Final Year Project is the 'Smart Multimodal Recognition System for the Visually Impaired'. I have also worked on deepfake detection (EfficientNet-B4) and toxic comment classification using BERT."
    else:
        # Check if context has relevant information
        if len(retrieved) > 0 and retrieved[0]["score"] > 0.05:
            answer = f"Based on my documents: {retrieved[0]['chunk']['text'][:250]}..."
        else:
            answer = "I don't have that in my notes."
            
    # Format retrieved chunks display panel
    sources_text = ""
    for r in retrieved:
        sources_text += f"**Source:** {r['chunk']['source']} | **ID:** {r['id']} | **Score:** {r['score']:.4f}\n> {r['chunk']['text']}\n\n----\n"
        
    return answer, sources_text

# --- 5. Gradio User Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Hybrid RAG Digital Clone (Eman)")
    gr.Markdown("⚠️ **This is an AI clone, not the real person.**")
    
    with gr.Row():
        mode_selector = gr.Radio(
            choices=["BM25 only", "Dense only", "Hybrid"],
            value="Hybrid",
            label="Retrieval Mode"
        )
        
    with gr.Row():
        query_input = gr.Textbox(label="Ask me anything about Eman:", placeholder="e.g., What is your tech stack or final year project?")
        
    submit_btn = gr.Button("Generate Answer", variant="primary")
    
    with gr.Row():
        answer_output = gr.Textbox(label="Clone Response")
        
    with gr.Row():
        chunks_output = gr.Markdown(label="Retrieved Chunks & Sources")
        
    submit_btn.click(
        fn=generate_response,
        inputs=[query_input, mode_selector],
        outputs=[answer_output, chunks_output]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)