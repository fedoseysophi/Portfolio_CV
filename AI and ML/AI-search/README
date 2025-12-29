# Service-for-searching-files-in-a-corporate-database
Corporate Documents QA API
This repository contains a set of Python services for building and running a Question Answering (QA) API over a corporate document collection. The system enables users to search for relevant documents and receive concise, grounded answers generated from your organization’s PDF and text files.

Components
corporate_qa_fastapi.py
The main FastAPI-based REST API. This service:

- Loads pre-processed corporate document embeddings.
- Uses a SentenceTransformer model for semantic document retrieval.
- Uses a Llama-based (Mistral) model for controlled answer generation.
- Provides endpoints for document search (/search), context-based answer generation (/initial-answer), follow-up answers, and monitoring system health (/profile).

# first_step.py
Utility for uploading and initially preprocessing source documents. Handles basic parsing and chunking of PDFs/Word files in preparation for embedding.

# files_preprocessing.py
Advanced document processing pipeline, including text normalization, chunking, and embedding generation. Prepares the grouped corpus ready for inference.

# second_step.py
Responsible for final embedding aggregation and possibly vector index updates or export.

# How it Works
Data Ingestion & Preprocessing:
Use first_step.py and files_preprocessing.py to prepare your raw documents, segment content, and generate semantic embeddings.

API Startup:
Launch corporate_qa_fastapi.py (e.g., with uvicorn corporate_qa_fastapi:app).
On startup, the service loads all precomputed embeddings and initializes both retrieval and generative models.

User Workflow:
The user queries the API with a natural language question.
The /search endpoint returns the top-N relevant documents.
The /initial-answer endpoint generates a concise answer using only the selected document's content.
The system enforces tight answer grounding: if information cannot be found, the model clearly replies "Cannot answer".
/followup supports clarification questions based on the context already provided.

Monitoring:
The /profile endpoint exposes process/thread count, memory usage, and OS load for monitoring and debugging.

Use Cases
Internal corporate knowledge bases
Automated HR or legal FAQ systems
Documented process and compliance Q&A
Any domain where factual, constrained answer generation from regulated files is required

Stack
FastAPI, Pydantic – REST API and validation
sentence-transformers – Semantic retrieval (E5-based multilingual model)
llama-cpp-python – On-premises LLM-based answer generation (Mistral)
asyncpg – (Optional) PG database for metadata/usage logging
NumPy, orjson, threading – Performance/scalability support
Getting Started
Prepare your document corpus and embeddings using pipeline scripts.
Configure models and path locations as needed.
Start the API server:

bash
uvicorn corporate_qa_fastapi:app --host 0.0.0.0 --port 8000
Use any HTTP client (curl, requests, etc) to interact with the endpoints.
