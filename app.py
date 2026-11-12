# --- 4. Generation & Response Logic ---
def generate_response(query, mode):
    if mode == "BM25 only":
        retrieved = retrieve_bm25(query, top_k=3)
    elif mode == "Dense only":
        retrieved = retrieve_dense(query, top_k=3)
    else:
        retrieved = retrieve_hybrid(query, top_k=3)
        
    # Grounded generation from the top retrieved chunk
    if len(retrieved) > 0 and retrieved[0]["score"] > 0.0:
        top_chunk_text = retrieved[0]["chunk"]["text"]
        # Generate answer directly grounded in the retrieved context
        answer = f"Based on my notes: {top_chunk_text}"
    else:
        answer = "I don't have that in my notes."
            
    # Format retrieved chunks display panel
    sources_text = ""
    for r in retrieved:
        sources_text += f"**Source:** {r['chunk']['source']} | **ID:** {r['id']} | **Score:** {r['score']:.4f}\n> {r['chunk']['text']}\n\n----\n"
        
    return answer, sources_text