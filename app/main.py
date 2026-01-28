import time
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from app.schemas import QueryRequest, QueryResponse
from app.ingest import ingest_document
from app.rag import retrieve_chunks, generate_answer

import os

app = FastAPI(title="RAG QA API")

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    background_tasks.add_task(ingest_document, file_path)

    return {"message": "File uploaded successfully", "filename": file.filename}



@app.post("/query", response_model=QueryResponse)
def query_docs(request: QueryRequest):
    start_time = time.time()

    chunks = retrieve_chunks(request.question)
    answer = generate_answer(request.question, chunks)

    latency = time.time() - start_time
    print(f"[QUERY] latency={latency:.2f}s chunks={len(chunks)}")

    return QueryResponse(answer=answer)
