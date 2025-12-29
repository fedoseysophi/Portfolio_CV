from fastapi import FastAPI, Request
from pydantic import BaseModel, validator
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import numpy as np
import orjson
import asyncpg
from rapidfuzz import fuzz
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache
import os
import threading
import psutil
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Optimized Corporate Documents QA API",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json"
)

# --- Configuration ---
DB_CONFIG = {
    "dbname": "YOUR_DB_NAME",
    "user": "YOUR_USER",
    "password": "YOUR_PASSWORD",
    "host": "YOUR_HOST",
    "port": "5432"
}

# --- Global variables ---
e5_model = None
mistral = None
executor = ThreadPoolExecutor(max_workers=4)
documents: List[dict] = []
documents_dict: dict = {}
document_embeddings: dict = {}

# --- Model/document loading on startup ---
@app.on_event("startup")
async def startup_event():
    global e5_model, mistral, documents, documents_dict, document_embeddings

    logger.info("Loading models...")
    with ThreadPoolExecutor() as startup_executor:
        future_e5 = startup_executor.submit(
            SentenceTransformer,
            'intfloat/multilingual-e5-large-instruct'
        )
        future_mistral = startup_executor.submit(
            Llama,
            model_path="/path/to/your/mistral-model.gguf",
            n_ctx=1024,
            n_gpu_layers=24,
            n_threads=8,
            f16_kv=True,
            low_vram=True,
            use_mmap=True,
            use_mlock=True,
            n_batch=128
        )
        e5_model = future_e5.result()
        mistral = future_mistral.result()

    logger.info("Loading documents and embeddings...")
    try:
        with open("/path/to/grouped_document_embeddings_filled.json", "rb") as f:
            documents = orjson.loads(f.read())
        documents_dict = {doc['file_name']: doc for doc in documents}
        document_embeddings = {}
        for doc in documents:
            file_name = doc['file_name']
            if doc.get('combined_embedding'):
                document_embeddings[file_name] = np.array(doc['combined_embedding'])
            else:
                chunk_embs = [np.array(chunk['combined_embedding']) for chunk in doc.get('chunks', []) if chunk.get('combined_embedding')]
                if chunk_embs:
                    document_embeddings[file_name] = np.mean(chunk_embs, axis=0)
        logger.info(f"Loaded {len(documents)} documents and {len(document_embeddings)} embeddings")
    except Exception as e:
        logger.error(f"Document loading error: {e}")
        raise

# --- Helper functions ---
@lru_cache(maxsize=1000)
def encode_text_cached(text: str):
    return e5_model.encode(text, normalize_embeddings=True)

async def encode_text_async(text: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, encode_text_cached, text)

def get_document_by_name(file_name: str):
    return documents_dict.get(file_name)

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    tokens = mistral.tokenize(text.encode("utf-8"))
    if len(tokens) <= max_tokens:
        return text
    truncated = mistral.detokenize(tokens[:max_tokens])
    return truncated.decode("utf-8", errors="ignore")

def find_most_relevant_chunks(document: dict, query_embedding: np.ndarray, top_k: int = 2) -> str:
    # Return the most relevant document chunks for the question embedding
    if not document.get("chunks"):
        logger.warning(f"Document {document.get('file_name')} contains no chunks")
        return ""
    query_norm = np.linalg.norm(query_embedding)
    chunks_with_scores = []
    for chunk in document.get('chunks', []):
        if not chunk.get('combined_embedding'):
            continue
        try:
            emb = np.array(chunk['combined_embedding'])
            if emb.shape[0] != query_embedding.shape[0]:
                continue
            sim = np.dot(query_embedding, emb) / (query_norm * np.linalg.norm(emb))
            chunks_with_scores.append((chunk['text'], float(sim)))
        except Exception as e:
            logger.error(f"Chunk processing error: {e}")
    if not chunks_with_scores:
        return ""
    chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
    return "\n\n".join([text for text, _ in chunks_with_scores[:top_k]])

def generate_answer(document: str, question: str, max_total_tokens: int = 1024) -> str:
    # Generate a concise answer to the question using only the provided document context
    question_tokens = mistral.tokenize(question.encode("utf-8"))
    prompt_prefix = """You are a professional corporate assistant. Answer the question using ONLY the information provided. If the answer is not found in the document, reply "Cannot answer".

Document:
"""
    prompt_suffix = f"\n\nQuestion: {question}\n\nAnswer (max 50 words):"
    prompt_overhead_tokens = len(mistral.tokenize((prompt_prefix + prompt_suffix).encode("utf-8")))
    max_doc_tokens = max_total_tokens - len(question_tokens) - prompt_overhead_tokens - 100
    truncated_doc = truncate_by_tokens(document, max_tokens=max_doc_tokens)
    prompt = f"{prompt_prefix}{truncated_doc}{prompt_suffix}"
    output = mistral(
        prompt,
        max_tokens=100,
        temperature=0.1,
        top_p=0.5,
        repeat_penalty=1.2
    )
    return output['choices'][0]['text'].strip()

async def generate_answer_async(document: str, question: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_answer, document, question)

# --- Query models ---
class SearchQuery(BaseModel):
    query: str
    similarity_threshold: float = 0.5

class FileSelection(BaseModel):
    file_name: str
    original_query: str
    @validator('original_query')
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

class FollowupQuestion(BaseModel):
    file_name: str
    question: str
    @validator('question')
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

# --- API Endpoints ---
@app.post("/search")
async def search_documents(query: SearchQuery):
    # Returns top relevant documents for the query
    try:
        query_emb = await encode_text_async(query.query)
        q_np = np.array(query_emb)
        results = []
        for fname, doc in documents_dict.items():
            doc_emb = document_embeddings.get(fname)
            if doc_emb is None:
                continue
            sim = np.dot(q_np, doc_emb) / (np.linalg.norm(q_np) * np.linalg.norm(doc_emb))
            if sim >= query.similarity_threshold:
                results.append({
                    "file_name": fname,
                    "file_path": doc['file_path'],
                    "similarity": float(sim)
                })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return {"results": results[:5]}
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"error": "Internal server error"}, 500

@app.post("/initial-answer")
async def initial_answer(selection: FileSelection):
    # Generates a context-based answer to an initial question
    try:
        doc = get_document_by_name(selection.file_name)
        if not doc:
            return {"error": "Document not found"}
        q_emb = await encode_text_async(selection.original_query)
        relevant = find_most_relevant_chunks(doc, np.array(q_emb))
        if not relevant:
            return {"answer": "Cannot find relevant information in the document"}
        answer = await generate_answer_async(relevant, selection.original_query)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        return {"error": "Internal server error"}, 500

@app.post("/followup")
async def followup_question(followup: FollowupQuestion):
    # Generates an answer to a followup question based on the document context
    try:
        doc = get_document_by_name(followup.file_name)
        if not doc:
            return {"error": "Document not found"}
        q_emb = await encode_text_async(followup.question)
        relevant = find_most_relevant_chunks(doc, np.array(q_emb))
        if not relevant:
            return {"answer": "Cannot find relevant information in the document"}
        answer = await generate_answer_async(relevant, followup.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Followup question error: {e}")
        return {"error": "Internal server error"}, 500

@app.get("/profile")
async def profile_info():
    # Returns process and resource usage info for monitoring
    return {
        "active_threads": threading.active_count(),
        "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "load_avg": os.getloadavg()
    }

