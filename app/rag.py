import os
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


""" Paths """
INDEX_DIR = "data/faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "metadata.pkl")


""" Device """
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


""" EMBEDDING MODEL (for retrieval)
    Purpose: convert text → vectors """
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=DEVICE
)


""" LLM (for answer generation)
    Purpose: convert context + question → answer"""
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base"
).to(DEVICE)


"""Load FAISS index + metadata"""
def load_index():
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError("FAISS index not found. Upload documents first.")

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


"""RETRIEVAL STEP"""
def retrieve_chunks(query: str, top_k: int = 3):
    """
    1. Embed query
    2. Search FAISS
    3. Return top-k text chunks
    """
    index, metadata = load_index()

    # Embed the user query
    query_vector = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    # Similarity search
    distances, indices = index.search(query_vector, top_k)

    # Collect retrieved chunks
    retrieved_chunks = []
    for idx in indices[0]:
        retrieved_chunks.append(metadata[idx]["text"])

    return retrieved_chunks



""" GENERATION STEP"""
def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Use retrieved chunks to generate final answer
    """
    context = "\n".join(context_chunks)

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {question}
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    outputs = llm.generate(
        **inputs,
        max_new_tokens=150
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
