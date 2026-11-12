import os
import glob
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def load_and_chunk_docs(data_folder="data"):
    chunks = []
    chunk_id = 0
    
    file_paths = glob.glob(os.path.join(data_folder, "*.md")) + glob.glob(os.path.join(data_folder, "*.txt"))
    print(f"Found {len(file_paths)} files in '{data_folder}' folder.")
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_word_count = 0
        
        for para in paragraphs:
            words = para.split()
            if not words:
                continue
            current_chunk.append(para)
            current_word_count += len(words)
            
            if current_word_count >= 180:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "id": chunk_id,
                    "source": file_name,
                    "content": chunk_text
                })
                chunk_id += 1
                current_chunk = []
                current_word_count = 0
                
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "id": chunk_id,
                "source": file_name,
                "content": chunk_text
            })
            chunk_id += 1
            
    return chunks

# Chunks banayein
chunks = load_and_chunk_docs()
print(f"Total Chunks Created: {len(chunks)}")

# Embeddings generate karein
print("Loading sentence-transformers model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_texts = [c["content"] for c in chunks]
print("Generating embeddings for chunks...")
embeddings = embed_model.encode(chunk_texts, show_progress_bar=True)

# .npy file save karein
np.save("embeddings.npy", embeddings)
print("Success! 'embeddings.npy' file generated and saved successfully.")