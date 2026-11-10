import os
import gradio as gr
import numpy as np

# --- 1. RAG Setup & Data Loading ---
# Load your pre-built embeddings and data if applicable
if os.path.exists("embeddings.npy"):
    embeddings = np.load("embeddings.npy")

def generate_rag_response(message, history):
    """
    Core Hybrid RAG response generation logic.
    Replace or integrate your retrieval and LLM logic here.
    """
    # Example response placeholder
    response = f"Received your query: '{message}'. Your Digital Clone is online and active!"
    return response

# --- 2. Gradio User Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Hybrid RAG Digital Clone")
    gr.Markdown("A personalized AI digital clone powered by markdown files and vector embeddings.")
    
    gr.ChatInterface(
        fn=generate_rag_response,
        title="Digital Clone Assistant",
        description="Ask me anything about my academic profile, projects, or professional background!"
    )

# --- 3. Railway Port Binding & Server Launch ---
if __name__ == "__main__":
    # Fetch the dynamic port provided by Railway, defaulting to 7860 if local
    port = int(os.environ.get("PORT", 7860))
    
    # Bind to 0.0.0.0 to make it publicly accessible externally through Railway's proxy
    demo.launch(server_name="0.0.0.0", server_port=port)