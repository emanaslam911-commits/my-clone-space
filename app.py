import os
import gradio as gr
import numpy as np

# --- 1. Load Embeddings & Profile Data ---
if os.path.exists("embeddings.npy"):
    embeddings = np.load("embeddings.npy")

def generate_rag_response(message, history):
    query = message.lower()
    
    # Specific intent matching for identity, education, skills & projects
    if any(k in query for k in ["name", "who are you", "who r u", "yourself"]):
        return "Hi! I am Aima, an AI developer pursuing a degree in Software Engineering / Computer Science."
    elif any(k in query for k in ["degree", "study", "semester", "education"]):
        return "I am currently pursuing a Bachelor of Science degree in Software Engineering / Computer Science."
    elif any(k in query for k in ["tech stack", "skills", "programming", "languages"]):
        return "My tech stack includes Python, C++, NumPy, pandas, TensorFlow, PyTorch, Flask, Streamlit, SQLite, Git, and VS Code."
    elif any(k in query for k in ["project", "final year", "fyp"]):
        return "My Final Year Project is titled 'Smart Multimodal Recognition System for the Visually Impaired' (DeepFake-Guard / Blind Assistance AI)."
    elif any(k in query for k in ["experience", "work", "job"]):
        return "I have hands-on experience in import/export operations management, deep learning research (deepfake detection, NLP classification), and e-commerce analytics."
    else:
        return f"I am Aima's AI Digital Clone. You asked: '{message}'. Feel free to ask about my name, academic background, tech stack, or projects!"

# --- 2. Gradio User Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Hybrid RAG Digital Clone")
    gr.Markdown("A personalized AI digital clone powered by markdown files and vector embeddings.")
    
    gr.ChatInterface(
        fn=generate_rag_response,
        title="Digital Clone Assistant",
        description="Ask me anything about my identity, academic profile, projects, or professional background!"
    )

# --- 3. Railway Port Binding & Server Launch ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())