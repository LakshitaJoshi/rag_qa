import os
from typing import List

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import time

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100 

DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
os.makedirs(INDEX_DIR, exist_ok=True)

INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "metadata.pkl")


""" Loading Embedding Model"""
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")


""" Text Extraction """
def extract_text(file_path: str)->str:
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="igmore") as f:
            return f.read()
        
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text=""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    
    raise ValueError("Unsupported file type")


""" Chunking (character based) """
def chunk_text(text: str)->List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start= end - CHUNK_OVERLAP

    return chunks


""" FAISS index """
def load_faiss():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    
    index = None
    metadata = []
    return index, metadata


def save_faiss(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)



""" Ingestion """
def ingest_document(file_path:str):
    print("INGESTION STARTED FOR:", file_path)
    start_time = time.time()

    text = extract_text(file_path)
    chunks = chunk_text(text)

    embeddings = model.encode(chunks, convert_to_numpy=True)

    index, metadata = load_faiss()
    
    if index is None:
        index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)

    for chunk in chunks:
        metadata.append(
            {
                "text": chunk,
                "source": os.path.basename(file_path),
            }
        )

    save_faiss(index, metadata)

    duration = time.time() - start_time

    print(
        f"[INGESTION COMPLETED] file = {file_path}"
        f"chunks={len(chunks)} time={duration:.2f}s"
    )