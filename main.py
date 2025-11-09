# main.py
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# =========================
# 1️⃣ Load your JSON files
# =========================
DOCS_PATH = 'docs'
all_chunks = []

for filename in os.listdir(DOCS_PATH):
    if filename.endswith('.json'):
        with open(os.path.join(DOCS_PATH, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for chunk in data['chunks']:
                all_chunks.append({
                    "movie": data['title'],
                    "type": chunk['type'],
                    "text": chunk['text']
                })

print(f"Loaded {len(all_chunks)} chunks from {len(os.listdir(DOCS_PATH))} movies")

# =========================
# 2️⃣ Create embeddings
# =========================
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [chunk['text'] for chunk in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)

print("Embeddings created!")

# =========================
# 3️⃣ Store embeddings in FAISS
# =========================
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
embeddings_np = np.array(embeddings).astype('float32')
index.add(embeddings_np)

print(f"FAISS index contains {index.ntotal} vectors")

# =========================
# 4️⃣ Query your system
# =========================
def query_system(question, top_k=2):
    query_emb = model.encode([question]).astype('float32')
    D, I = index.search(query_emb, k=top_k)
    results = []
    for i in I[0]:
        results.append({
            "movie": all_chunks[i]['movie'],
            "type": all_chunks[i]['type'],
            "text": all_chunks[i]['text']
        })
    return results

# Example usage
question = "Who improvised a line in Inception?"
results = query_system(question)
for r in results:
    print(f"Movie: {r['movie']} | Type: {r['type']}")
    print(r['text'])
    print("-----")

# ---- RAG LLM setup (use Mistral 7B instruct for reliable local demo) ----


# Use Mistral 7B instruct (open) instead of gated LLaMA for now
model_name = "mistralai/mistral-7b-instruct"

print("Loading LLM:", model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
# use device_map="auto" if you have a GPU; remove it if you want pure CPU (but slower)
llama_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",       # set to "auto" if you have a GPU or accelerate installed
    torch_dtype=torch.float16,  # float16 for GPU; use float32 for CPU
    use_auth_token=True
)
# ensure model in eval mode
llama_model.eval()

def rag_answer(question, top_k=3):
    retrieved = query_system(question, top_k=top_k)
    # Build context - keep short so LLM doesn't get confused
    context = "\n\n".join([f"[{r['movie']} | {r['type']}]\n{r['text']}" for r in retrieved])
    prompt = f"Use the context to answer the question concisely.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(llama_model.device)
    outputs = llama_model.generate(**inputs, max_new_tokens=150, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Trim repeated prompt if present
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return answer
