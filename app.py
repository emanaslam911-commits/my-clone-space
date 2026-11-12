import os
import difflib
import gradio as gr
import numpy as np

# --- 1. Load Embeddings & Profile Data ---
if os.path.exists("embeddings.npy"):
    embeddings = np.load("embeddings.npy")

# Complete Knowledge Base for Eman
KNOWLEDGE_BASE = [
    {
        "keywords": ["name", "who are you", "who r u", "yourself", "intro", "introduction", "eman"],
        "answer": "Hi! I am Eman, a Software Engineering undergraduate currently in my 7th semester. I specialize in AI, Machine Learning, Computer Vision, and full-stack Python development."
    },
    {
        "keywords": ["degree", "study", "semester", "smester", "academic", "background", "backgroud", "education", "cgpa", "university", "coursework", "courses", "subjects"],
        "answer": "I am pursuing a Bachelor of Science in Software Engineering (currently in my 7th semester, having completed 103 credit hours). My coursework includes Artificial Intelligence, Machine Learning, Natural Language Processing, Data Structures, Operating Systems, Information Security, and Software Design & Architecture."
    },
    {
        "keywords": ["tech stack", "skills", "programming", "languages", "tools", "frameworks", "python", "ml", "ai"],
        "answer": "My core technical stack includes Python, C++, PyTorch, TensorFlow, Keras, Flask, Streamlit, Gradio, SQLite, Git, GitHub, and Hugging Face. I also have strong experience with Google Workspace and MS Office."
    },
    {
        "keywords": ["project", "projects", "final year", "fyp", "blind", "deepfake", "nlp", "blockchain", "wireshark"],
        "answer": (
            "Here are my key projects:\n"
            "1. **Final Year Project (FYP):** Smart Multimodal Recognition System for the Visually Impaired — an Android edge-AI app featuring real-time object detection, obstacle navigation, currency recognition, face recognition, and emergency fall detection.\n"
            "2. **Deepfake Detection:** Evaluated MesoNet and EfficientNet-B4 architectures on the 140k Real and Fake Faces dataset.\n"
            "3. **Toxic Comment Classification (NLP):** Built and compared TF-IDF + Logistic Regression, BiLSTM, and fine-tuned BERT models on the Jigsaw dataset.\n"
            "4. **Blockchain Voting System:** Developed a secure web-based voting platform using Python, Flask, and SQLite."
        )
    },
    {
        "keywords": ["experience", "work", "job", "internship", "import", "export", "ecommerce", "ebay", "certification", "cisco"],
        "answer": "I have 5 months of experience handling import/export operations and international client/vendor communications, as well as assisting with e-commerce inventory management and competitor analysis on eBay UK. Additionally, I hold a Cyber Threat Management certification from Cisco Networking Academy."
    }
]

def match_intent(query):
    words = query.lower().replace("?", "").replace("'", "").split()
    best_score = 0
    best_answer = None

    for item in KNOWLEDGE_BASE:
        score = 0
        for kw in item["keywords"]:
            # Direct substring match
            if kw in query.lower():
                score += 2
            # Fuzzy word match (handles typos like 'smester' or 'backgroud')
            for w in words:
                similarity = difflib.SequenceMatcher(None, w, kw).ratio()
                if similarity >= 0.80:
                    score += similarity
        if score > best_score:
            best_score = score
            best_answer = item["answer"]

    if best_score >= 0.80:
        return best_answer
    return None

def generate_rag_response(message, history):
    answer = match_intent(message)
    if answer:
        return answer
    return (
        "I'm Eman's AI Digital Clone! I didn't quite catch that, but you can ask me about:\n"
        "- **Identity:** Who I am & my current semester\n"
        "- **Academic Background:** My degree & completed coursework\n"
        "- **Tech Stack:** Programming languages & AI frameworks\n"
        "- **Projects:** My FYP, Deepfake Detection, NLP, or Blockchain work\n"
        "- **Experience:** Work history & certifications"
    )

# --- 2. Gradio User Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Hybrid RAG Digital Clone")
    gr.Markdown("A personalized AI digital clone powered by markdown files and vector embeddings.")
    
    gr.ChatInterface(
        fn=generate_rag_response,
        title="Digital Clone Assistant",
        description="Ask me anything about Eman's identity, academic profile, projects, or professional background!"
    )

# --- 3. Railway Port Binding & Server Launch ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())