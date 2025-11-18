EndoSuite RAG QA Agent
A modular, production-ready Retrieval-Augmented Generation (RAG) system with intent-aware routing, deployed via FastAPI and Fly.io.

Overview:
This project demonstrates a hybrid AI agent that performs autonomous medical QA using Retrieval-Augmented Generation (RAG) with symbolic intent classification and neural inference.
It was developed for EndoSuite, a surgical training simulator, to enable contextually grounded question-answering and document navigation over simulation manuals.

Key Features:
Intent-Aware Reasoning: Automatically detects query types (count, list, details, or general) to optimize response flow.

Efficient Routing: Structured questions are served from a local cache, while open-ended ones trigger retrieval and model inference.

Vector Search with FAISS: Embeds Markdown documents using BAAI/bge-base-en-v1.5 and retrieves semantically relevant chunks.

Remote Model Inference: Uses a Hugging Face Inference Endpoint for scalable GPU-backed LLM responses (Mistral-7B-Instruct).

FastAPI Deployment: Clean REST API (/ask) for interactive or programmatic use.

Fly.io Configuration: Includes Docker and fly.toml for cloud deployment with persistent FAISS cache volumes.

System Architecture:
User Query → Intent Classifier → (Structured Query → Cache Lookup) or (Open-Ended Query → FAISS Retriever → Hugging Face Endpoint) → Response Cleaner → FastAPI Endpoint (/ask)

File Structure:
demo_main.py - Main FastAPI app with RAG pipeline
pdf_to_markdown_endosuite.ipynb - PDF to Markdown preprocessor
endo_suite_pdfs.zip - Source documents
requirements.txt - Dependencies
Dockerfile - Container configuration
fly.toml - Fly.io deployment settings
README.txt - This file

Local Setup:
* Clone repository
git clone https://github.com/glockkm/endosuite-intent-aware-rag-agent.git


* cd endosuite-intent-aware-rag-agent

* Create virtual environment
python -m venv venv
source venv/bin/activate

* Install dependencies
pip install -r requirements.txt

** Note: the following is a general example of the deploymnet pipeline that I used for work and is not a valiud pipeline for public usage. Please set up your own pipeline and adjust the python script as needed. 

* Run API locally
uvicorn demo_main:app --reload

* Then open http://127.0.0.1:8000/docs
 to test the /ask endpoint.

Deployment (Fly.io):
* Authenticate with Fly:
fly auth login

* Deploy:
fly launch

Set secrets (for Hugging Face endpoint access):
* flyctl secrets set HF_ENDPOINT_URL=https://endpoint-url

* flyctl secrets set HF_API_TOKEN=your_hf_token

Example Query:
curl -X POST "https://your-fly-app.fly.dev/ask
"
-H "Content-Type: application/json"
-d '{"question": "List all tasks in the Endovascular module."}'

Response:
{
"answer": "Task 1: Catheter insertion\nTask 2: Wire navigation\n...",
"sources": [{"source": "Endovascular Module.md", "page": 4}]
}

Tech Stack:
Language: Python 3.11
Framework: FastAPI
Retrieval: FAISS
Embeddings: SentenceTransformers (BAAI/bge-base-en-v1.5)
LLM: Mistral-7B-Instruct (via Hugging Face Endpoint)
Deployment: Fly.io / Docker
Environment: Linux / Cloud

Author:
Kimberly Glock, M.S.
Data Scientist | ML/AI Engineer | Medical Simulation and Robotics
linkedin.com/in/kimberly-glock

License:
MIT License — educational and demonstrative use only.
