import os
import gradio as gr
import numpy as np

# --- 1. Load Embeddings & Profile Data ---
if os.path.exists("embeddings.npy"):
    embeddings = np.load("embeddings.npy")

def generate_rag_response(message, history):
    """
    RAG Logic / Persona Knowledge Base for your Digital Clone.
    This handles user queries based on your professional profile.
    """
    query = message.lower()
    
    # Matching keywords to provide relevant personal/professional answers
    if "degree" in query or "study" in query or "semester" in query:
        return "I am currently pursuing a Bachelor of Science degree in Software Engineering / Computer Science, studying in my final stages."
    elif "tech stack" in query or "skills" in query or "programming" in query:
        return "My core tech stack includes Python, C++, NumPy, pandas, TensorFlow, PyTorch, Flask, Streamlit, SQLite, Git, and VS Code."
    elif "project" in query or "final year" in query:
        return "My Final Year Project is titled 'Smart Multimodal Recognition System for the Visually Impaired' (Blind Assistance AI / DeepFake-Guard), focusing on advanced deep learning models like EfficientNet-B4."
    elif "experience" in query or "work" in query:
        return "I have hands-on experience managing import and export operations, digital marketing analysis, and conducting deep learning research for computer vision and NLP pipelines (like toxic comment classification and deepfake detection)."
    else:
        return f"Hello! I am your AI Digital Clone. You asked: '{message}'. Feel free to ask about my academic profile, technical skills, projects, or professional background!"

# --- 2. Gradio User Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Hybrid RAG Digital Clone")
    gr.Markdown("A personalized AI digital clone powered by markdown files and vector embeddings.")
    
    gr.ChatInterface(
        fn=generate_rag_response,
        title="Digital Clone Assistant",
        description="Ask me anything about my academic profile, projects, or professional background!"
    )

# --- 3. Railway Port Binding & Server Launch ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())