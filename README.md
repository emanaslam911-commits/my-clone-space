python -c "
content = '''# Hybrid RAG Digital Clone (Eman)

A public Hybrid RAG chatbot representing my profile, built with Python, Gradio, BM25, and dense embeddings, deployed live.

⚠️ **This is an AI clone, not the real person.**

---

## 1. Knowledge Base Chosen & Why
My knowledge base consists of key structured documents covering my professional and academic background:
- **`bio.md`**: Core biographical details, location (Sialkot, Pakistan), and undergraduate status (Software Engineering, 7th semester, 103 credit hours).
- **`cv.md`**: Technical skills, programming languages (Python, C++, Flask, Streamlit, PyTorch, TensorFlow), and coursework details.
- **`projects.md`**: Key technical projects including my Final Year Project (\"Smart Multimodal Recognition System for the Visually Impaired\"), deepfake detection, and NLP systems.
- **`experience.md`**: Professional background covering international client/vendor communications and e-commerce inventory management.

These documents provide a balanced mix of factual data, personal background, and technical project write-ups, ensuring the clone speaks accurately in my voice.

---

## 2. Test Questions Evaluation Table

| Question | BM25 Result | Dense Result | Hybrid Result | Best Mode and Why |
| :--- | :--- | :--- | :--- | :--- |
| **Q1 (Exact Keyword):** \"What is your Final Year Project titled?\" | Retrieved `projects.md` highlighting the exact FYP title. | Retrieved project section successfully via semantic match. | Successfully fused rankings with top score. | **Hybrid / BM25:** Exact keyword match (\"Final Year Project\") allows BM25 to score high precision. |
| **Q2 (Paraphrased):** \"Tell me about your coding languages and tools.\" | Missed some context due to lack of direct keywords. | Retrieved `cv.md` via semantic embedding similarity. | Combined both for a comprehensive response. | **Dense:** Paraphrased wording shares no direct words with the text, making dense search superior. |
| **Q3:** \"Where are you based?\" | Retrieved `bio.md` with location details. | Retrieved location via general context. | Cleanest top chunk selected. | **Hybrid:** Combines keyword precision and semantic scoring robustly. |
| **Q4:** \"Do you have work experience?\" | Retrieved `experience.md` details. | Retrieved professional background chunks. | Ranked experience documents accurately. | **Hybrid:** RRF fusion ensures the most relevant professional chunk stays on top. |
| **Q5:** \"What coursework have you completed?\" | Matched technical terms. | Retrieved relevant curriculum info. | Structured precise context. | **Dense:** Handles conceptual course queries effectively through embeddings. |

---

## 3. Failure Case
- **Query:** \"What is your favorite programming language for machine learning?\"
- **Behavior:** The clone retrieved relevant technical files (`cv.md`), but because my documents listed multiple tools (PyTorch, TensorFlow, Python) without explicitly labeling a \"favorite\", the generation layer gave a generalized tech stack response instead of a personal preference.
- **Explanation:** RAG systems rely strictly on documented context. Since personal subjective preferences (\"favorite\") were not explicitly stated in the source text, the model could not invent or hallucinate a preference, which aligns with the safety constraints.

---

## 4. Future Improvement
Next steps for enhancement include implementing a cross-encoder reranker over hybrid search results to further improve chunk relevance precision for complex or nuanced personal queries.
'''
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)
print('README.md created successfully!')
"