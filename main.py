# Added RemoteHFEndpointLLM and updated get_llm() to use HF Inference Endpoint via HF_ENDPOINT_URL + HF_API_TOKEN (from Fly secrets)

# Local AutoModelForCausalLM loading is no longer used (but left in imports)

# Long request timeout (600s) for first cold start on HF

# HF endpoint# === IMPORTS ===
import os
import re
import glob
import json
from typing import Any, List, Dict, Union, Optional
from fastapi import FastAPI

# LangChain & HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from langchain_core.documents import Document
import requests  # <-- ADDED

# API Framework
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# === CONFIG ===
MARKDOWN_FOLDER = "converted_pdfs"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#INDEX_PATH = "faiss_index"
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")

# UPDATED: read HF token from environment (set via `flyctl secrets set HUGGINGFACE_HUB_TOKEN=...`)
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", None)

MODEL_NAME = os.getenv("MODEL_NAME", "mistral7b")
MODEL_MAP = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}
LLM_MODEL = MODEL_MAP.get(MODEL_NAME, "meta-llama/Meta-Llama-3-8B-Instruct")
CONTEXT_LEN = 8192 if MODEL_NAME == "llama3" else 4096

print(f"=== Using MODEL_NAME={MODEL_NAME}, mapped to HuggingFace ID: {LLM_MODEL} ===")
print("=== HF token detected? ", "yes" if HF_TOKEN else "no", "===")

# Pipeline hyperparameters
hyperparams = {
    "chunk_size": 2400,
    "chunk_overlap": 160,
    "retriever_k": 8,
    "rerank_top_n": 4,
    "max_new_tokens": 768 #256  
}

# === FASTAPI APP ===
app = FastAPI(
    title="Endosuite RAG QA API",
    description=f"Medical QA powered by FAISS + HuggingFace ({MODEL_NAME})",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str = Field(..., example="Please type your question here.")


# === UTILS ===
def get_embedding_dim(emb_model_name):
    model = SentenceTransformer(emb_model_name)
    return model.get_sentence_embedding_dimension()

def faiss_index_dim(index_path):
    import faiss
    idx_file = os.path.join(index_path, "index.faiss")
    if not os.path.exists(idx_file):
        return None
    return faiss.read_index(idx_file).d

def chunk_params_match(index_path, chunk_size, chunk_overlap, emb_model_name):
    path = os.path.join(index_path, "chunk_params.json")
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        params = json.load(f)
    return (
        params.get("chunk_size") == chunk_size and
        params.get("chunk_overlap") == chunk_overlap and
        params.get("embedding_model") == emb_model_name
    )

def save_chunk_params(index_path, chunk_size, chunk_overlap, emb_model_name):
    os.makedirs(index_path, exist_ok=True)
    with open(os.path.join(index_path, "chunk_params.json"), "w") as f:
        json.dump({
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": emb_model_name
        }, f)

def load_markdown_docs(md_folder):
    docs = []
    for md_path in glob.glob(os.path.join(md_folder, "*.md")):
        with open(md_path, encoding="utf-8") as f:
            content = f.read()
        pages = re.split(r"(?=\n+# Page \d+\n)", content)
        for i, page_text in enumerate(pages):
            if page_text.strip():
                docs.append(Document(
                    page_content=page_text.strip(),
                    metadata={"source": os.path.basename(md_path), "page": i+1}
                ))
    return docs


# === DEFERRED / LAZY INITIALIZATION (prevents 502 on Fly at boot) ===
_vectorstore: Optional[FAISS] = None
_retriever = None
_tokenizer = None
_model = None
_llm = None
_cross_encoder = None

# def _hf_auth_kwargs(token: Optional[str]) -> Dict[str, str]:
#     kw: Dict[str, str] = {}
#     if token:
#         kw["token"] = token           # newer transformers
#         kw["use_auth_token"] = token  # backward compatibility
#     return kw

def _hf_auth_kwargs(token: Optional[str]) -> Dict[str, str]:
    return {"token": token} if token else {}


def get_retriever():
    global _vectorstore, _retriever
    if _retriever is not None:
        return _retriever

    print(f"=== Initializing retriever with model: {MODEL_NAME} ({LLM_MODEL}) ===")
    need_rebuild = False
    expected_dim = get_embedding_dim(EMBED_MODEL)

    if os.path.exists(INDEX_PATH):
        faiss_dim = faiss_index_dim(INDEX_PATH)
        params_ok = chunk_params_match(INDEX_PATH, hyperparams["chunk_size"], hyperparams["chunk_overlap"], EMBED_MODEL)
        if faiss_dim != expected_dim or not params_ok:
            import shutil
            shutil.rmtree(INDEX_PATH)
            need_rebuild = True
    else:
        need_rebuild = True

    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if need_rebuild:
        docs = load_markdown_docs(MARKDOWN_FOLDER)
        for d in docs:
            d.page_content = re.sub(r"\s+", " ", d.page_content).strip()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=hyperparams["chunk_size"],
            chunk_overlap=hyperparams["chunk_overlap"]
        )
        chunks = splitter.split_documents(docs)
        _vectorstore = FAISS.from_documents(chunks, emb)
        _vectorstore.save_local(INDEX_PATH)
        save_chunk_params(INDEX_PATH, hyperparams["chunk_size"], hyperparams["chunk_overlap"], EMBED_MODEL)
    else:
        _vectorstore = FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)

    _retriever = _vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": hyperparams["retriever_k"]})
    return _retriever

class DirectHFModel(LLM):
    model: Any
    tokenizer: Any
    device: Any
    max_new_tokens: int
    class Config:
        arbitrary_types_allowed = True
    def _llm_type(self) -> str:
        return "direct-hf-model"
    def _call(self, prompt: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": "You are a precise medical training QA assistant."},
            {"role": "user", "content": prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        output_ids = self.model.generate(
            inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id
        )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        print("\n=== RAW MODEL OUTPUT ===")
        print(text)
        print("=== END RAW OUTPUT ===\n")
        if "ANSWER:" in text:
            text = text.split("ANSWER:")[-1].strip()
        return text or "Not found in documents."

# === ADDED: Remote HF Inference Endpoint LLM ===
class RemoteHFEndpointLLM(LLM):
    endpoint_url: str
    hf_token: str
    max_new_tokens: int

    def _llm_type(self) -> str:
        return "remote-hf-endpoint"

    def _call(self, prompt: str, **kwargs) -> str:
        # Wrap prompt for Mistral-Instruct style (works fine with TGI endpoints)
        payload = {
            "inputs": f"[INST] {prompt} [/INST]",
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.2,
                "return_full_text": False
            }
        }
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        # Large timeout to accommodate scale-from-zero or cold starts
        r = requests.post(self.endpoint_url, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()

        # Handle common response shapes from TGI / endpoints
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"].strip()
        elif isinstance(data, dict) and "generated_text" in data:
            text = data["generated_text"].strip()
        else:
            text = (data.get("outputs", "") or "").strip() if isinstance(data, dict) else ""

        print("\n=== RAW MODEL OUTPUT (remote) ===")
        print(text)
        print("=== END RAW OUTPUT ===\n")

        if "ANSWER:" in text:
            text = text.split("ANSWER:")[-1].strip()
        return text or "Not found in documents."

def get_llm():
    global _llm, _model, _tokenizer, _cross_encoder
    if _llm is not None:
        return _llm

    # Prefer a dedicated HF_API_TOKEN, fall back to your existing HUGGINGFACE_HUB_TOKEN
    HF_API_TOKEN = os.getenv("HF_API_TOKEN", HF_TOKEN or "")
    HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL", "")

    if not HF_API_TOKEN or not HF_ENDPOINT_URL:
        raise RuntimeError("Missing HF_API_TOKEN or HF_ENDPOINT_URL for remote GPU inference.")

    # Build remote LLM (no local model load)
    _llm = RemoteHFEndpointLLM(
        endpoint_url=HF_ENDPOINT_URL.rstrip("/"),
        hf_token=HF_API_TOKEN,
        max_new_tokens=hyperparams["max_new_tokens"],
    )

    # Optional CrossEncoder (CPU). Safe to skip if unused.
    try:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device='cpu')
    except Exception as _:
        _cross_encoder = None

    return _llm


# === HELPERS ===
def clean_answer(answer: str) -> Union[str, list]:
    if not answer:
        return ""
    answer = re.sub(r"\[/?INST\]", "", answer, flags=re.IGNORECASE)
    stop_markers = ["surgicalscience", "[TABLE]", "Figure", "Page", "Chapter"]
    for marker in stop_markers:
        idx = answer.find(marker)
        if idx != -1:
            answer = answer[:idx]
            break
    answer = answer.strip()
    answer = re.sub(r"•\s*", "- ", answer)
    return answer


# === MODULE CACHE (cases, tasks, guides) ===
CASE_HEADER_RE = re.compile(r'(?mi)^(?:#{1,6}\s*)?Case\s+(\d+)\b')
TASK_RE = re.compile(r'(?mi)^(?:Landmark|Task)\s+(\d+):?\s*(.*)')

def _clean_line(s: str) -> str:
    s = re.sub(r'^[\-\*\u2022•>\|]\s*', '', s.strip())
    s = re.sub(r'\s+', ' ', s)
    return s.strip(" :;-")

def extract_cases_from_text(text: str) -> Dict[str, Dict[str, str]]:
    lines = text.splitlines()
    cases: Dict[str, Dict[str, str]] = {}
    current_num, buffer = None, []

    def flush():
        nonlocal current_num, buffer
        if current_num is None:
            return
        block = "\n".join(buffer).strip()
        header_line = buffer[0] if buffer else ""
        m = re.match(r'(?i)Case\s+(\d+)[:\-\s]*(.*)', header_line)
        if m and m.group(2).strip():
            title = _clean_line(m.group(2))
        else:
            title = ""
            for ln in buffer[1:]:
                if not ln.strip():
                    continue
                if CASE_HEADER_RE.match(ln):
                    break
                title = _clean_line(ln)
                break
        cases[f"Case {current_num}"] = {
            "title": f"Case {current_num}: {title}" if title else f"Case {current_num}",
            "text": block
        }
        current_num, buffer = None, []

    for ln in lines:
        m = CASE_HEADER_RE.match(ln)
        if m:
            flush()
            current_num = int(m.group(1))
            buffer = [ln]
        elif current_num is not None:
            buffer.append(ln)
    flush()
    return cases

def extract_tasks_from_text(text: str) -> Dict[str, Dict[str, str]]:
    tasks = {}
    lines = text.splitlines()
    current_num, current_title, body = None, None, []

    def flush():
        nonlocal current_num, current_title, body
        if current_num is not None:
            tasks[f"Task {current_num}"] = {
                "title": f"Task {current_num}: {current_title.strip() if current_title else ''}",
                "text": "\n".join(body).strip()
            }
        current_num, current_title, body = None, None, []

    for idx, ln in enumerate(lines):
        m = TASK_RE.match(ln)
        if m:
            flush()
            current_num = int(m.group(1))
            current_title = m.group(2).strip()
            if not current_title:
                # fallback: peek ahead to next non-empty line
                if idx + 1 < len(lines) and lines[idx + 1].strip():
                    current_title = _clean_line(lines[idx + 1])
        else:
            if current_num is not None:
                body.append(ln)

    flush()
    return tasks

def list_available_modules(md_folder: str) -> Dict[str, str]:
    module_map = {}
    for md_path in glob.glob(os.path.join(md_folder, "*.md")):
        fname = os.path.basename(md_path)
        base = fname.lower().replace(" module book.md", "").replace(".md", "").strip()
        module_map[base] = fname
    return module_map

def build_module_cache(md_folder: str) -> Dict[str, Dict]:
    module_map = list_available_modules(md_folder)
    cache = {}
    for key, fname in module_map.items():
        file_path = os.path.join(md_folder, fname)
        try:
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            if re.search(r'(?mi)\bCase\s+\d+', text):
                parsed = extract_cases_from_text(text)
                cache[key] = {
                    "type": "case",
                    "count": len(parsed),
                    "titles": [v["title"] for v in parsed.values()],
                    "texts": {k: v["text"] for k, v in parsed.items()}
                }
            elif re.search(r'(?mi)\b(Task|Landmark)\s+\d+', text):
                parsed = extract_tasks_from_text(text)
                cache[key] = {
                    "type": "task",
                    "count": len(parsed),
                    "titles": [v["title"] for v in parsed.values()],
                    "texts": {k: v["text"] for k, v in parsed.items()}
                }
            else:
                cache[key] = {"type": "guide", "count": 0, "titles": [], "texts": {}}

        except Exception:
            cache[key] = {"type": "guide", "count": 0, "titles": [], "texts": {}}
    return cache

module_cache = build_module_cache(MARKDOWN_FOLDER)
print(f"=== Module cache built: {{k:v['count'] for k,v in module_cache.items()}} ===")


# === INTENT CLASSIFIER ===
def classify_intent(query: str) -> str:
    q = query.lower()
    if re.search(r'\b(how many|number of|total)\s+(cases|tasks|landmarks)\b', q):
        return "count"
    if re.search(r'\b(list|what are|which|names?)\b.*\b(cases|tasks|landmarks)\b', q):
        return "list"
    if re.search(r'\b(show|give|display|open|present)\b.*\b(case|task|landmark)\s+\d+\b', q):
        return "details"
    return "general"


# === QUERY HELPERS ===
def extract_module_from_query(query: str, module_map: Dict[str, str]) -> Optional[str]:
    query = query.lower()
    for key in module_map.keys():
        if key in query:
            return key
    return None


# === MAIN QA FUNCTION ===
def rag_qa(query: str, retriever, llm) -> Dict[str, Any]:
    intent = classify_intent(query)
    print(f"🔎 Intent classified as: {intent}")

    module_map = list_available_modules(MARKDOWN_FOLDER)
    module_name = extract_module_from_query(query, module_map)

    if module_name and module_name in module_cache:
        mod = module_cache[module_name]
        if intent == "count":
            return {
                "answer": f"There are {mod['count']} {mod['type']}s.",
                "sources": [{"source": module_map[module_name]}]
            }
        if intent == "list":
            return {
                "answer": "\n".join(mod["titles"]) if mod["titles"] else f"No {mod['type']}s found.",
                "sources": [{"source": module_map[module_name]}]
            }
        if intent == "details":
            m = re.search(rf"{mod['type']}\s+(\d+)", query, flags=re.IGNORECASE)
            if m:
                idx = m.group(1)
                key = f"{mod['type'].capitalize()} {idx}"
                details = mod["texts"].get(key)
                if details:
                    return {
                        "answer": details,
                        "sources": [{"source": module_map[module_name]}]
                    }
                return {"answer": f"{key} not found.", "sources": []}

    # === General RAG fallback ===
    initial_docs = retriever.get_relevant_documents(query)
    seen, unique_docs = set(), []
    for d in initial_docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    context = "\n\n".join([d.page_content for d in unique_docs])
    system_instructions = (
        "You are a precise medical training QA assistant.\n"
        "Answer ONLY the given QUESTION using the provided CONTEXT.\n"
        "If the question asks for case/task names or counts, use the extracted cache values.\n"
        "Do not mix content from other modules.\n"
        "Do not answer more than one question.\n"
        "Do not repeat the question.\n"
        "Do not include greetings, filler, or company information.\n"
        "Provide only the final concise answer.\n"
        "If nothing relevant is found, reply exactly: Not found in documents."
    )
    prompt = f"{system_instructions}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"
    raw_answer = llm(prompt)
    final_answer = clean_answer(raw_answer)

    sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in unique_docs]
    return {"answer": final_answer, "sources": sources}


# === API ENDPOINTS ===

# Lightweight health check so Fly knows the app is alive
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=dict)
async def ask_question(request: QuestionRequest):
    try:
        # lazy-load heavy components on first real request
        retriever = get_retriever()
        llm = get_llm()
        result = rag_qa(request.question, retriever, llm)
        return result
    except Exception as e:
        return {"error": str(e)}





# # === IMPORTS ===
# import os
# import re
# import glob
# import json
# from typing import Any, List, Dict, Union, Optional
# from fastapi import FastAPI

# # LangChain & HuggingFace
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms.base import LLM
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from sentence_transformers import CrossEncoder, SentenceTransformer, util
# from langchain_core.documents import Document

# # API Framework
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field

# # === CONFIG ===
# MARKDOWN_FOLDER = "converted_pdfs"
# EMBED_MODEL = "BAAI/bge-base-en-v1.5"
# CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# INDEX_PATH = "faiss_index"

# # UPDATED: read HF token from environment (set via `flyctl secrets set HUGGINGFACE_HUB_TOKEN=...`)
# HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", None)

# MODEL_NAME = os.getenv("MODEL_NAME", "mistral7b")
# MODEL_MAP = {
#     "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
#     "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
#     "mixtral8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1"
# }
# LLM_MODEL = MODEL_MAP.get(MODEL_NAME, "meta-llama/Meta-Llama-3-8B-Instruct")
# CONTEXT_LEN = 8192 if MODEL_NAME == "llama3" else 4096

# print(f"=== Using MODEL_NAME={MODEL_NAME}, mapped to HuggingFace ID: {LLM_MODEL} ===")
# print("=== HF token detected? ", "yes" if HF_TOKEN else "no", "===")

# # Pipeline hyperparameters
# hyperparams = {
#     "chunk_size": 2400,
#     "chunk_overlap": 160,
#     "retriever_k": 8,
#     "rerank_top_n": 4,
#     "max_new_tokens": 768,
# }

# # === FASTAPI APP ===
# app = FastAPI(
#     title="Endosuite RAG QA API",
#     description=f"Medical QA powered by FAISS + HuggingFace ({MODEL_NAME})",
#     version="1.0.0",
# )
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QuestionRequest(BaseModel):
#     question: str = Field(..., example="Please type your question here.")


# # === UTILS ===
# def get_embedding_dim(emb_model_name):
#     model = SentenceTransformer(emb_model_name)
#     return model.get_sentence_embedding_dimension()

# def faiss_index_dim(index_path):
#     import faiss
#     idx_file = os.path.join(index_path, "index.faiss")
#     if not os.path.exists(idx_file):
#         return None
#     return faiss.read_index(idx_file).d

# def chunk_params_match(index_path, chunk_size, chunk_overlap, emb_model_name):
#     path = os.path.join(index_path, "chunk_params.json")
#     if not os.path.exists(path):
#         return False
#     with open(path, "r") as f:
#         params = json.load(f)
#     return (
#         params.get("chunk_size") == chunk_size and
#         params.get("chunk_overlap") == chunk_overlap and
#         params.get("embedding_model") == emb_model_name
#     )

# def save_chunk_params(index_path, chunk_size, chunk_overlap, emb_model_name):
#     os.makedirs(index_path, exist_ok=True)
#     with open(os.path.join(index_path, "chunk_params.json"), "w") as f:
#         json.dump({
#             "chunk_size": chunk_size,
#             "chunk_overlap": chunk_overlap,
#             "embedding_model": emb_model_name
#         }, f)

# def load_markdown_docs(md_folder):
#     docs = []
#     for md_path in glob.glob(os.path.join(md_folder, "*.md")):
#         with open(md_path, encoding="utf-8") as f:
#             content = f.read()
#         pages = re.split(r"(?=\n+# Page \d+\n)", content)
#         for i, page_text in enumerate(pages):
#             if page_text.strip():
#                 docs.append(Document(
#                     page_content=page_text.strip(),
#                     metadata={"source": os.path.basename(md_path), "page": i+1}
#                 ))
#     return docs


# # === LOAD VECTORSTORE ===
# print(f"=== Initializing retriever with model: {MODEL_NAME} ({LLM_MODEL}) ===")
# need_rebuild = False
# expected_dim = get_embedding_dim(EMBED_MODEL)

# if os.path.exists(INDEX_PATH):
#     faiss_dim = faiss_index_dim(INDEX_PATH)
#     params_ok = chunk_params_match(INDEX_PATH, hyperparams["chunk_size"], hyperparams["chunk_overlap"], EMBED_MODEL)
#     if faiss_dim != expected_dim or not params_ok:
#         import shutil
#         shutil.rmtree(INDEX_PATH)
#         need_rebuild = True
# else:
#     need_rebuild = True

# emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# if need_rebuild:
#     docs = load_markdown_docs(MARKDOWN_FOLDER)
#     for d in docs:
#         d.page_content = re.sub(r"\s+", " ", d.page_content).strip()
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=hyperparams["chunk_size"],
#         chunk_overlap=hyperparams["chunk_overlap"]
#     )
#     chunks = splitter.split_documents(docs)
#     vectorstore = FAISS.from_documents(chunks, emb)
#     vectorstore.save_local(INDEX_PATH)
#     save_chunk_params(INDEX_PATH, hyperparams["chunk_size"], hyperparams["chunk_overlap"], EMBED_MODEL)
# else:
#     vectorstore = FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)

# retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": hyperparams["retriever_k"]})


# # === LLM WRAPPER ===
# class DirectHFModel(LLM):
#     model: Any
#     tokenizer: Any
#     device: Any
#     max_new_tokens: int

#     class Config:
#         arbitrary_types_allowed = True

#     def _llm_type(self) -> str:
#         return "direct-hf-model"

#     def _call(self, prompt: str, **kwargs) -> str:
#         messages = [
#             {"role": "system", "content": "You are a precise medical training QA assistant."},
#             {"role": "user", "content": prompt}
#         ]
#         inputs = self.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         ).to(self.device)

#         output_ids = self.model.generate(
#             inputs,
#             max_new_tokens=self.max_new_tokens,
#             do_sample=False,
#             eos_token_id=self.tokenizer.eos_token_id
#         )

#         text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

#         print("\n=== RAW MODEL OUTPUT ===")
#         print(text)
#         print("=== END RAW OUTPUT ===\n")

#         if "ANSWER:" in text:
#             text = text.split("ANSWER:")[-1].strip()

#         return text or "Not found in documents."


# # === MODEL INIT ===
# # FIX: pass ONLY `token` (not both `token` and `use_auth_token`)
# def _hf_auth_kwargs(token: Optional[str]) -> Dict[str, str]:
#     return {"token": token} if token else {}

# _auth = _hf_auth_kwargs(HF_TOKEN)

# tokenizer = AutoTokenizer.from_pretrained(
#     LLM_MODEL,
#     use_fast=False,
#     **_auth
# )
# model = AutoModelForCausalLM.from_pretrained(
#     LLM_MODEL,
#     device_map="auto",
#     torch_dtype="auto",
#     **_auth
# )
# device = next(model.parameters()).device
# model.eval()

# llm = DirectHFModel(model=model, tokenizer=tokenizer, device=device, max_new_tokens=hyperparams["max_new_tokens"])
# cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device='cuda')

# # Zero-shot intent classifier
# intent_classifier = pipeline("text-classification", model="facebook/bart-large-mnli")


# # === HELPERS ===
# def clean_answer(answer: str) -> Union[str, list]:
#     if not answer:
#         return ""
#     answer = re.sub(r"\[/?INST\]", "", answer, flags=re.IGNORECASE)
#     stop_markers = ["surgicalscience", "[TABLE]", "Figure", "Page", "Chapter"]
#     for marker in stop_markers:
#         idx = answer.find(marker)
#         if idx != -1:
#             answer = answer[:idx]
#             break
#     answer = answer.strip()
#     answer = re.sub(r"•\s*", "- ", answer)
#     return answer


# # === MODULE CACHE (cases, tasks, guides) ===
# CASE_HEADER_RE = re.compile(r'(?mi)^(?:#{1,6}\s*)?Case\s+(\d+)\b')
# TASK_RE = re.compile(r'(?mi)^(?:Landmark|Task)\s+(\d+):?\s*(.*)')

# def _clean_line(s: str) -> str:
#     s = re.sub(r'^[\-\*\u2022•>\|]\s*', '', s.strip())
#     s = re.sub(r'\s+', ' ', s)
#     return s.strip(" :;-")

# def extract_cases_from_text(text: str) -> Dict[str, Dict[str, str]]:
#     lines = text.splitlines()
#     cases: Dict[str, Dict[str, str]] = {}
#     current_num, buffer = None, []

#     def flush():
#         nonlocal current_num, buffer
#         if current_num is None:
#             return
#         block = "\n".join(buffer).strip()
#         header_line = buffer[0] if buffer else ""
#         m = re.match(r'(?i)Case\s+(\d+)[:\-\s]*(.*)', header_line)
#         if m and m.group(2).strip():
#             title = _clean_line(m.group(2))
#         else:
#             title = ""
#             for ln in buffer[1:]:
#                 if not ln.strip():
#                     continue
#                 if CASE_HEADER_RE.match(ln):
#                     break
#                 title = _clean_line(ln)
#                 break
#         cases[f"Case {current_num}"] = {
#             "title": f"Case {current_num}: {title}" if title else f"Case {current_num}",
#             "text": block
#         }
#         current_num, buffer = None, []

#     for ln in lines:
#         m = CASE_HEADER_RE.match(ln)
#         if m:
#             flush()
#             current_num = int(m.group(1))
#             buffer = [ln]
#         elif current_num is not None:
#             buffer.append(ln)
#     flush()
#     return cases

# def extract_tasks_from_text(text: str) -> Dict[str, Dict[str, str]]:
#     tasks = {}
#     lines = text.splitlines()
#     current_num, current_title, body = None, None, []

#     def flush():
#         nonlocal current_num, current_title, body
#         if current_num is not None:
#             tasks[f"Task {current_num}"] = {
#                 "title": f"Task {current_num}: {current_title.strip() if current_title else ''}",
#                 "text": "\n".join(body).strip()
#             }
#         current_num, current_title, body = None, None, []

#     for idx, ln in enumerate(lines):
#         m = TASK_RE.match(ln)
#         if m:
#             flush()
#             current_num = int(m.group(1))
#             current_title = m.group(2).strip()
#             if not current_title:
#                 # fallback: peek ahead to next non-empty line
#                 if idx + 1 < len(lines) and lines[idx + 1].strip():
#                     current_title = _clean_line(lines[idx + 1])
#         else:
#             if current_num is not None:
#                 body.append(ln)

#     flush()
#     return tasks

# def list_available_modules(md_folder: str) -> Dict[str, str]:
#     module_map = {}
#     for md_path in glob.glob(os.path.join(md_folder, "*.md")):
#         fname = os.path.basename(md_path)
#         base = fname.lower().replace(" module book.md", "").replace(".md", "").strip()
#         module_map[base] = fname
#     return module_map

# def build_module_cache(md_folder: str) -> Dict[str, Dict]:
#     module_map = list_available_modules(md_folder)
#     cache = {}
#     for key, fname in module_map.items():
#         file_path = os.path.join(md_folder, fname)
#         try:
#             with open(file_path, encoding="utf-8") as f:
#                 text = f.read()

#             if re.search(r'(?mi)\bCase\s+\d+', text):
#                 parsed = extract_cases_from_text(text)
#                 cache[key] = {
#                     "type": "case",
#                     "count": len(parsed),
#                     "titles": [v["title"] for v in parsed.values()],
#                     "texts": {k: v["text"] for k, v in parsed.items()}
#                 }
#             elif re.search(r'(?mi)\b(Task|Landmark)\s+\d+', text):
#                 parsed = extract_tasks_from_text(text)
#                 cache[key] = {
#                     "type": "task",
#                     "count": len(parsed),
#                     "titles": [v["title"] for v in parsed.values()],
#                     "texts": {k: v["text"] for k, v in parsed.items()}
#                 }
#             else:
#                 cache[key] = {"type": "guide", "count": 0, "titles": [], "texts": {}}

#         except Exception:
#             cache[key] = {"type": "guide", "count": 0, "titles": [], "texts": {}}
#     return cache

# module_cache = build_module_cache(MARKDOWN_FOLDER)
# print(f"=== Module cache built: {{k:v['count'] for k,v in module_cache.items()}} ===")


# # === INTENT CLASSIFIER ===
# def classify_intent(query: str) -> str:
#     q = query.lower()
#     if re.search(r'\b(how many|number of|total)\s+(cases|tasks|landmarks)\b', q):
#         return "count"
#     if re.search(r'\b(list|what are|which|names?)\b.*\b(cases|tasks|landmarks)\b', q):
#         return "list"
#     if re.search(r'\b(show|give|display|open|present)\b.*\b(case|task|landmark)\s+\d+\b', q):
#         return "details"
#     return "general"


# # === QUERY HELPERS ===
# def extract_module_from_query(query: str, module_map: Dict[str, str]) -> Optional[str]:
#     query = query.lower()
#     for key in module_map.keys():
#         if key in query:
#             return key
#     return None


# # === MAIN QA FUNCTION ===
# def rag_qa(query: str, retriever, llm) -> Dict[str, Any]:
#     intent = classify_intent(query)
#     print(f"🔎 Intent classified as: {intent}")

#     module_map = list_available_modules(MARKDOWN_FOLDER)
#     module_name = extract_module_from_query(query, module_map)

#     if module_name and module_name in module_cache:
#         mod = module_cache[module_name]
#         if intent == "count":
#             return {
#                 "answer": f"There are {mod['count']} {mod['type']}s.",
#                 "sources": [{"source": module_map[module_name]}]
#             }
#         if intent == "list":
#             return {
#                 "answer": "\n".join(mod["titles"]) if mod["titles"] else f"No {mod['type']}s found.",
#                 "sources": [{"source": module_map[module_name]}]
#             }
#         if intent == "details":
#             m = re.search(rf"{mod['type']}\s+(\d+)", query, flags=re.IGNORECASE)
#             if m:
#                 idx = m.group(1)
#                 key = f"{mod['type'].capitalize()} {idx}"
#                 details = mod["texts"].get(key)
#                 if details:
#                     return {
#                         "answer": details,
#                         "sources": [{"source": module_map[module_name]}]
#                     }
#                 return {"answer": f"{key} not found.", "sources": []}

#     # === General RAG fallback ===
#     initial_docs = retriever.get_relevant_documents(query)
#     seen, unique_docs = set(), []
#     for d in initial_docs:
#         key = (d.metadata.get("source"), d.metadata.get("page"))
#         if key not in seen:
#             seen.add(key)
#             unique_docs.append(d)

#     context = "\n\n".join([d.page_content for d in unique_docs])
#     system_instructions = (
#         "You are a precise medical training QA assistant.\n"
#         "Answer ONLY the given QUESTION using the provided CONTEXT.\n"
#         "If the question asks for case/task names or counts, use the extracted cache values.\n"
#         "Do not mix content from other modules.\n"
#         "Do not answer more than one question.\n"
#         "Do not repeat the question.\n"
#         "Do not include greetings, filler, or company information.\n"
#         "Provide only the final concise answer.\n"
#         "If nothing relevant is found, reply exactly: Not found in documents."
#     )
#     prompt = f"{system_instructions}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"
#     raw_answer = llm(prompt)
#     final_answer = clean_answer(raw_answer)

#     sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in unique_docs]
#     return {"answer": final_answer, "sources": sources}


# # === API ENDPOINT ===
# @app.post("/ask", response_model=dict)
# async def ask_question(request: QuestionRequest):
#     try:
#         result = rag_qa(request.question, retriever, llm)
#         return result
#     except Exception as e:
#         return {"error": str(e)}







# Lazy-loading of FAISS + model (get_retriever(), get_llm()) so the server binds to 0.0.0.0:8080 quickly and avoids Fly’s 502 “not listening” warning.

# HF token from env is respected everywhere via _hf_auth_kwargs.

# Health endpoint (GET /) so Fly can get a fast 200 during rollout.

################################ USED THIS BEFORE ADDING HF ENDPOINT TO CODE
# === IMPORTS ===
# import os
# import re
# import glob
# import json
# from typing import Any, List, Dict, Union, Optional
# from fastapi import FastAPI

# # LangChain & HuggingFace
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms.base import LLM
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from sentence_transformers import CrossEncoder, SentenceTransformer, util
# from langchain_core.documents import Document

# # API Framework
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field

# # === CONFIG ===
# MARKDOWN_FOLDER = "converted_pdfs"
# EMBED_MODEL = "BAAI/bge-base-en-v1.5"
# CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# #INDEX_PATH = "faiss_index"
# INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")

# # UPDATED: read HF token from environment (set via `flyctl secrets set HUGGINGFACE_HUB_TOKEN=...`)
# HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", None)

# MODEL_NAME = os.getenv("MODEL_NAME", "mistral7b")
# MODEL_MAP = {
#     "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
#     "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
#     "mixtral8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1"
# }
# LLM_MODEL = MODEL_MAP.get(MODEL_NAME, "meta-llama/Meta-Llama-3-8B-Instruct")
# CONTEXT_LEN = 8192 if MODEL_NAME == "llama3" else 4096

# print(f"=== Using MODEL_NAME={MODEL_NAME}, mapped to HuggingFace ID: {LLM_MODEL} ===")
# print("=== HF token detected? ", "yes" if HF_TOKEN else "no", "===")

# # Pipeline hyperparameters
# hyperparams = {
#     "chunk_size": 2400,
#     "chunk_overlap": 160,
#     "retriever_k": 8,
#     "rerank_top_n": 4,
#     "max_new_tokens": 256 # 768,
# }

# # === FASTAPI APP ===
# app = FastAPI(
#     title="Endosuite RAG QA API",
#     description=f"Medical QA powered by FAISS + HuggingFace ({MODEL_NAME})",
#     version="1.0.0",
# )
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QuestionRequest(BaseModel):
#     question: str = Field(..., example="Please type your question here.")


# # === UTILS ===
# def get_embedding_dim(emb_model_name):
#     model = SentenceTransformer(emb_model_name)
#     return model.get_sentence_embedding_dimension()

# def faiss_index_dim(index_path):
#     import faiss
#     idx_file = os.path.join(index_path, "index.faiss")
#     if not os.path.exists(idx_file):
#         return None
#     return faiss.read_index(idx_file).d

# def chunk_params_match(index_path, chunk_size, chunk_overlap, emb_model_name):
#     path = os.path.join(index_path, "chunk_params.json")
#     if not os.path.exists(path):
#         return False
#     with open(path, "r") as f:
#         params = json.load(f)
#     return (
#         params.get("chunk_size") == chunk_size and
#         params.get("chunk_overlap") == chunk_overlap and
#         params.get("embedding_model") == emb_model_name
#     )

# def save_chunk_params(index_path, chunk_size, chunk_overlap, emb_model_name):
#     os.makedirs(index_path, exist_ok=True)
#     with open(os.path.join(index_path, "chunk_params.json"), "w") as f:
#         json.dump({
#             "chunk_size": chunk_size,
#             "chunk_overlap": chunk_overlap,
#             "embedding_model": emb_model_name
#         }, f)

# def load_markdown_docs(md_folder):
#     docs = []
#     for md_path in glob.glob(os.path.join(md_folder, "*.md")):
#         with open(md_path, encoding="utf-8") as f:
#             content = f.read()
#         pages = re.split(r"(?=\n+# Page \d+\n)", content)
#         for i, page_text in enumerate(pages):
#             if page_text.strip():
#                 docs.append(Document(
#                     page_content=page_text.strip(),
#                     metadata={"source": os.path.basename(md_path), "page": i+1}
#                 ))
#     return docs


# # === DEFERRED / LAZY INITIALIZATION (prevents 502 on Fly at boot) ===
# _vectorstore: Optional[FAISS] = None
# _retriever = None
# _tokenizer = None
# _model = None
# _llm = None
# _cross_encoder = None

# # def _hf_auth_kwargs(token: Optional[str]) -> Dict[str, str]:
# #     kw: Dict[str, str] = {}
# #     if token:
# #         kw["token"] = token           # newer transformers
# #         kw["use_auth_token"] = token  # backward compatibility
# #     return kw

# def _hf_auth_kwargs(token: Optional[str]) -> Dict[str, str]:
#     return {"token": token} if token else {}


# def get_retriever():
#     global _vectorstore, _retriever
#     if _retriever is not None:
#         return _retriever

#     print(f"=== Initializing retriever with model: {MODEL_NAME} ({LLM_MODEL}) ===")
#     need_rebuild = False
#     expected_dim = get_embedding_dim(EMBED_MODEL)

#     if os.path.exists(INDEX_PATH):
#         faiss_dim = faiss_index_dim(INDEX_PATH)
#         params_ok = chunk_params_match(INDEX_PATH, hyperparams["chunk_size"], hyperparams["chunk_overlap"], EMBED_MODEL)
#         if faiss_dim != expected_dim or not params_ok:
#             import shutil
#             shutil.rmtree(INDEX_PATH)
#             need_rebuild = True
#     else:
#         need_rebuild = True

#     emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

#     if need_rebuild:
#         docs = load_markdown_docs(MARKDOWN_FOLDER)
#         for d in docs:
#             d.page_content = re.sub(r"\s+", " ", d.page_content).strip()
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=hyperparams["chunk_size"],
#             chunk_overlap=hyperparams["chunk_overlap"]
#         )
#         chunks = splitter.split_documents(docs)
#         _vectorstore = FAISS.from_documents(chunks, emb)
#         _vectorstore.save_local(INDEX_PATH)
#         save_chunk_params(INDEX_PATH, hyperparams["chunk_size"], hyperparams["chunk_overlap"], EMBED_MODEL)
#     else:
#         _vectorstore = FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)

#     _retriever = _vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": hyperparams["retriever_k"]})
#     return _retriever

# class DirectHFModel(LLM):
#     model: Any
#     tokenizer: Any
#     device: Any
#     max_new_tokens: int
#     class Config:
#         arbitrary_types_allowed = True
#     def _llm_type(self) -> str:
#         return "direct-hf-model"
#     def _call(self, prompt: str, **kwargs) -> str:
#         messages = [
#             {"role": "system", "content": "You are a precise medical training QA assistant."},
#             {"role": "user", "content": prompt}
#         ]
#         inputs = self.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         ).to(self.device)
#         output_ids = self.model.generate(
#             inputs,
#             max_new_tokens=self.max_new_tokens,
#             do_sample=False,
#             eos_token_id=self.tokenizer.eos_token_id
#         )
#         text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
#         print("\n=== RAW MODEL OUTPUT ===")
#         print(text)
#         print("=== END RAW OUTPUT ===\n")
#         if "ANSWER:" in text:
#             text = text.split("ANSWER:")[-1].strip()
#         return text or "Not found in documents."

# def get_llm():
#     global _llm, _model, _tokenizer, _cross_encoder
#     if _llm is not None:
#         return _llm
#     _auth = _hf_auth_kwargs(HF_TOKEN)
#     _tokenizer = AutoTokenizer.from_pretrained(
#         LLM_MODEL,
#         use_fast=False,
#         **_auth
#     )
#     _model = AutoModelForCausalLM.from_pretrained(
#         LLM_MODEL,
#         device_map="auto",
#         torch_dtype="auto",
#         **_auth
#     )
#     device = next(_model.parameters()).device
#     _model.eval()
#     _llm = DirectHFModel(model=_model, tokenizer=_tokenizer, device=device, max_new_tokens=hyperparams["max_new_tokens"])
#     # Lazy CrossEncoder on CPU to avoid GPU issues (not used in your current flow, but kept available)
#     try:
#         _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device='cpu')
#     except Exception as _:
#         _cross_encoder = None
#     return _llm


# # === HELPERS ===
# def clean_answer(answer: str) -> Union[str, list]:
#     if not answer:
#         return ""
#     answer = re.sub(r"\[/?INST\]", "", answer, flags=re.IGNORECASE)
#     stop_markers = ["surgicalscience", "[TABLE]", "Figure", "Page", "Chapter"]
#     for marker in stop_markers:
#         idx = answer.find(marker)
#         if idx != -1:
#             answer = answer[:idx]
#             break
#     answer = answer.strip()
#     answer = re.sub(r"•\s*", "- ", answer)
#     return answer


# # === MODULE CACHE (cases, tasks, guides) ===
# CASE_HEADER_RE = re.compile(r'(?mi)^(?:#{1,6}\s*)?Case\s+(\d+)\b')
# TASK_RE = re.compile(r'(?mi)^(?:Landmark|Task)\s+(\d+):?\s*(.*)')

# def _clean_line(s: str) -> str:
#     s = re.sub(r'^[\-\*\u2022•>\|]\s*', '', s.strip())
#     s = re.sub(r'\s+', ' ', s)
#     return s.strip(" :;-")

# def extract_cases_from_text(text: str) -> Dict[str, Dict[str, str]]:
#     lines = text.splitlines()
#     cases: Dict[str, Dict[str, str]] = {}
#     current_num, buffer = None, []

#     def flush():
#         nonlocal current_num, buffer
#         if current_num is None:
#             return
#         block = "\n".join(buffer).strip()
#         header_line = buffer[0] if buffer else ""
#         m = re.match(r'(?i)Case\s+(\d+)[:\-\s]*(.*)', header_line)
#         if m and m.group(2).strip():
#             title = _clean_line(m.group(2))
#         else:
#             title = ""
#             for ln in buffer[1:]:
#                 if not ln.strip():
#                     continue
#                 if CASE_HEADER_RE.match(ln):
#                     break
#                 title = _clean_line(ln)
#                 break
#         cases[f"Case {current_num}"] = {
#             "title": f"Case {current_num}: {title}" if title else f"Case {current_num}",
#             "text": block
#         }
#         current_num, buffer = None, []

#     for ln in lines:
#         m = CASE_HEADER_RE.match(ln)
#         if m:
#             flush()
#             current_num = int(m.group(1))
#             buffer = [ln]
#         elif current_num is not None:
#             buffer.append(ln)
#     flush()
#     return cases

# def extract_tasks_from_text(text: str) -> Dict[str, Dict[str, str]]:
#     tasks = {}
#     lines = text.splitlines()
#     current_num, current_title, body = None, None, []

#     def flush():
#         nonlocal current_num, current_title, body
#         if current_num is not None:
#             tasks[f"Task {current_num}"] = {
#                 "title": f"Task {current_num}: {current_title.strip() if current_title else ''}",
#                 "text": "\n".join(body).strip()
#             }
#         current_num, current_title, body = None, None, []

#     for idx, ln in enumerate(lines):
#         m = TASK_RE.match(ln)
#         if m:
#             flush()
#             current_num = int(m.group(1))
#             current_title = m.group(2).strip()
#             if not current_title:
#                 # fallback: peek ahead to next non-empty line
#                 if idx + 1 < len(lines) and lines[idx + 1].strip():
#                     current_title = _clean_line(lines[idx + 1])
#         else:
#             if current_num is not None:
#                 body.append(ln)

#     flush()
#     return tasks

# def list_available_modules(md_folder: str) -> Dict[str, str]:
#     module_map = {}
#     for md_path in glob.glob(os.path.join(md_folder, "*.md")):
#         fname = os.path.basename(md_path)
#         base = fname.lower().replace(" module book.md", "").replace(".md", "").strip()
#         module_map[base] = fname
#     return module_map

# def build_module_cache(md_folder: str) -> Dict[str, Dict]:
#     module_map = list_available_modules(md_folder)
#     cache = {}
#     for key, fname in module_map.items():
#         file_path = os.path.join(md_folder, fname)
#         try:
#             with open(file_path, encoding="utf-8") as f:
#                 text = f.read()

#             if re.search(r'(?mi)\bCase\s+\d+', text):
#                 parsed = extract_cases_from_text(text)
#                 cache[key] = {
#                     "type": "case",
#                     "count": len(parsed),
#                     "titles": [v["title"] for v in parsed.values()],
#                     "texts": {k: v["text"] for k, v in parsed.items()}
#                 }
#             elif re.search(r'(?mi)\b(Task|Landmark)\s+\d+', text):
#                 parsed = extract_tasks_from_text(text)
#                 cache[key] = {
#                     "type": "task",
#                     "count": len(parsed),
#                     "titles": [v["title"] for v in parsed.values()],
#                     "texts": {k: v["text"] for k, v in parsed.items()}
#                 }
#             else:
#                 cache[key] = {"type": "guide", "count": 0, "titles": [], "texts": {}}

#         except Exception:
#             cache[key] = {"type": "guide", "count": 0, "titles": [], "texts": {}}
#     return cache

# module_cache = build_module_cache(MARKDOWN_FOLDER)
# print(f"=== Module cache built: {{k:v['count'] for k,v in module_cache.items()}} ===")


# # === INTENT CLASSIFIER ===
# def classify_intent(query: str) -> str:
#     q = query.lower()
#     if re.search(r'\b(how many|number of|total)\s+(cases|tasks|landmarks)\b', q):
#         return "count"
#     if re.search(r'\b(list|what are|which|names?)\b.*\b(cases|tasks|landmarks)\b', q):
#         return "list"
#     if re.search(r'\b(show|give|display|open|present)\b.*\b(case|task|landmark)\s+\d+\b', q):
#         return "details"
#     return "general"


# # === QUERY HELPERS ===
# def extract_module_from_query(query: str, module_map: Dict[str, str]) -> Optional[str]:
#     query = query.lower()
#     for key in module_map.keys():
#         if key in query:
#             return key
#     return None


# # === MAIN QA FUNCTION ===
# def rag_qa(query: str, retriever, llm) -> Dict[str, Any]:
#     intent = classify_intent(query)
#     print(f"🔎 Intent classified as: {intent}")

#     module_map = list_available_modules(MARKDOWN_FOLDER)
#     module_name = extract_module_from_query(query, module_map)

#     if module_name and module_name in module_cache:
#         mod = module_cache[module_name]
#         if intent == "count":
#             return {
#                 "answer": f"There are {mod['count']} {mod['type']}s.",
#                 "sources": [{"source": module_map[module_name]}]
#             }
#         if intent == "list":
#             return {
#                 "answer": "\n".join(mod["titles"]) if mod["titles"] else f"No {mod['type']}s found.",
#                 "sources": [{"source": module_map[module_name]}]
#             }
#         if intent == "details":
#             m = re.search(rf"{mod['type']}\s+(\d+)", query, flags=re.IGNORECASE)
#             if m:
#                 idx = m.group(1)
#                 key = f"{mod['type'].capitalize()} {idx}"
#                 details = mod["texts"].get(key)
#                 if details:
#                     return {
#                         "answer": details,
#                         "sources": [{"source": module_map[module_name]}]
#                     }
#                 return {"answer": f"{key} not found.", "sources": []}

#     # === General RAG fallback ===
#     initial_docs = retriever.get_relevant_documents(query)
#     seen, unique_docs = set(), []
#     for d in initial_docs:
#         key = (d.metadata.get("source"), d.metadata.get("page"))
#         if key not in seen:
#             seen.add(key)
#             unique_docs.append(d)

#     context = "\n\n".join([d.page_content for d in unique_docs])
#     system_instructions = (
#         "You are a precise medical training QA assistant.\n"
#         "Answer ONLY the given QUESTION using the provided CONTEXT.\n"
#         "If the question asks for case/task names or counts, use the extracted cache values.\n"
#         "Do not mix content from other modules.\n"
#         "Do not answer more than one question.\n"
#         "Do not repeat the question.\n"
#         "Do not include greetings, filler, or company information.\n"
#         "Provide only the final concise answer.\n"
#         "If nothing relevant is found, reply exactly: Not found in documents."
#     )
#     prompt = f"{system_instructions}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"
#     raw_answer = llm(prompt)
#     final_answer = clean_answer(raw_answer)

#     sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in unique_docs]
#     return {"answer": final_answer, "sources": sources}


# # === API ENDPOINTS ===

# # Lightweight health check so Fly knows the app is alive
# @app.get("/")
# def health():
#     return {"status": "ok"}

# @app.post("/ask", response_model=dict)
# async def ask_question(request: QuestionRequest):
#     try:
#         # lazy-load heavy components on first real request
#         retriever = get_retriever()
#         llm = get_llm()
#         result = rag_qa(request.question, retriever, llm)
#         return result
#     except Exception as e:
#         return {"error": str(e)}


# # === IMPORTS ===
# import os
# import re
# import glob
# import json
# from typing import Any, List, Dict, Union, Optional
# from fastapi import FastAPI

# # LangChain & HuggingFace
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms.base import LLM
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from sentence_transformers import CrossEncoder, SentenceTransformer, util
# from langchain_core.documents import Document

# # API Framework
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field

# # === CONFIG ===
# MARKDOWN_FOLDER = "converted_pdfs"
# EMBED_MODEL = "BAAI/bge-base-en-v1.5"
# CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# INDEX_PATH = "faiss_index"

# # UPDATED: read HF token from environment (set via `flyctl secrets set HUGGINGFACE_HUB_TOKEN=...`)
# HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", None)

# MODEL_NAME = os.getenv("MODEL_NAME", "mistral7b")
# MODEL_MAP = {
#     "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
#     "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
#     "mixtral8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1"
# }
# LLM_MODEL = MODEL_MAP.get(MODEL_NAME, "meta-llama/Meta-Llama-3-8B-Instruct")
# CONTEXT_LEN = 8192 if MODEL_NAME == "llama3" else 4096

# print(f"=== Using MODEL_NAME={MODEL_NAME}, mapped to HuggingFace ID: {LLM_MODEL} ===")
# print("=== HF token detected? ", "yes" if HF_TOKEN else "no", "===")

# # Pipeline hyperparameters
# hyperparams = {
#     "chunk_size": 2400,
#     "chunk_overlap": 160,
#     "retriever_k": 8,
#     "rerank_top_n": 4,
#     "max_new_tokens": 768,
# }

# # === FASTAPI APP ===
# app = FastAPI(
#     title="Endosuite RAG QA API",
#     description=f"Medical QA powered by FAISS + HuggingFace ({MODEL_NAME})",
#     version="1.0.0",
# )
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QuestionRequest(BaseModel):
#     question: str = Field(..., example="Please type your question here.")


# # === UTILS ===
# def get_embedding_dim(emb_model_name):
#     model = SentenceTransformer(emb_model_name)
#     return model.get_sentence_embedding_dimension()

# def faiss_index_dim(index_path):
#     import faiss
#     idx_file = os.path.join(index_path, "index.faiss")
#     if not os.path.exists(idx_file):
#         return None
#     return faiss.read_index(idx_file).d

# def chunk_params_match(index_path, chunk_size, chunk_overlap, emb_model_name):
#     path = os.path.join(index_path, "chunk_params.json")
#     if not os.path.exists(path):
#         return False
#     with open(path, "r") as f:
#         params = json.load(f)
#     return (
#         params.get("chunk_size") == chunk_size and
#         params.get("chunk_overlap") == chunk_overlap and
#         params.get("embedding_model") == emb_model_name
#     )

# def save_chunk_params(index_path, chunk_size, chunk_overlap, emb_model_name):
#     os.makedirs(index_path, exist_ok=True)
#     with open(os.path.join(index_path, "chunk_params.json"), "w") as f:
#         json.dump({
#             "chunk_size": chunk_size,
#             "chunk_overlap": chunk_overlap,
#             "embedding_model": emb_model_name
#         }, f)

# def load_markdown_docs(md_folder):
#     docs = []
#     for md_path in glob.glob(os.path.join(md_folder, "*.md")):
#         with open(md_path, encoding="utf-8") as f:
#             content = f.read()
#         pages = re.split(r"(?=\n+# Page \d+\n)", content)
#         for i, page_text in enumerate(pages):
#             if page_text.strip():
#                 docs.append(Document(
#                     page_content=page_text.strip(),
#                     metadata={"source": os.path.basename(md_path), "page": i+1}
#                 ))
#     return docs


# # === LOAD VECTORSTORE ===
# print(f"=== Initializing retriever with model: {MODEL_NAME} ({LLM_MODEL}) ===")
# need_rebuild = False
# expected_dim = get_embedding_dim(EMBED_MODEL)

# if os.path.exists(INDEX_PATH):
#     faiss_dim = faiss_index_dim(INDEX_PATH)
#     params_ok = chunk_params_match(INDEX_PATH, hyperparams["chunk_size"], hyperparams["chunk_overlap"], EMBED_MODEL)
#     if faiss_dim != expected_dim or not params_ok:
#         import shutil
#         shutil.rmtree(INDEX_PATH)
#         need_rebuild = True
# else:
#     need_rebuild = True

# emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# if need_rebuild:
#     docs = load_markdown_docs(MARKDOWN_FOLDER)
#     for d in docs:
#         d.page_content = re.sub(r"\s+", " ", d.page_content).strip()
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=hyperparams["chunk_size"],
#         chunk_overlap=hyperparams["chunk_overlap"]
#     )
#     chunks = splitter.split_documents(docs)
#     vectorstore = FAISS.from_documents(chunks, emb)
#     vectorstore.save_local(INDEX_PATH)
#     save_chunk_params(INDEX_PATH, hyperparams["chunk_size"], hyperparams["chunk_overlap"], EMBED_MODEL)
# else:
#     vectorstore = FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)

# retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": hyperparams["retriever_k"]})


# # === LLM WRAPPER ===
# class DirectHFModel(LLM):
#     model: Any
#     tokenizer: Any
#     device: Any
#     max_new_tokens: int

#     class Config:
#         arbitrary_types_allowed = True

#     def _llm_type(self) -> str:
#         return "direct-hf-model"

#     def _call(self, prompt: str, **kwargs) -> str:
#         messages = [
#             {"role": "system", "content": "You are a precise medical training QA assistant."},
#             {"role": "user", "content": prompt}
#         ]
#         inputs = self.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         ).to(self.device)

#         output_ids = self.model.generate(
#             inputs,
#             max_new_tokens=self.max_new_tokens,
#             do_sample=False,
#             eos_token_id=self.tokenizer.eos_token_id
#         )

#         text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

#         print("\n=== RAW MODEL OUTPUT ===")
#         print(text)
#         print("=== END RAW OUTPUT ===\n")

#         if "ANSWER:" in text:
#             text = text.split("ANSWER:")[-1].strip()

#         return text or "Not found in documents."


# # === MODEL INIT ===
# # UPDATED: pass HF token via env secret (supports both new 'token' and legacy 'use_auth_token')
# def _hf_auth_kwargs(token: Optional[str]) -> Dict[str, str]:
#     kw: Dict[str, str] = {}
#     if token:
#         kw["token"] = token           # newer transformers
#         kw["use_auth_token"] = token  # backward compatibility
#     return kw

# _auth = _hf_auth_kwargs(HF_TOKEN)

# tokenizer = AutoTokenizer.from_pretrained(
#     LLM_MODEL,
#     use_fast=False,
#     **_auth
# )
# model = AutoModelForCausalLM.from_pretrained(
#     LLM_MODEL,
#     device_map="auto",
#     torch_dtype="auto",
#     **_auth
# )
# device = next(model.parameters()).device
# model.eval()

# llm = DirectHFModel(model=model, tokenizer=tokenizer, device=device, max_new_tokens=hyperparams["max_new_tokens"])
# cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device='cuda')

# # Zero-shot intent classifier
# intent_classifier = pipeline("text-classification", model="facebook/bart-large-mnli")


# # === HELPERS ===
# def clean_answer(answer: str) -> Union[str, list]:
#     if not answer:
#         return ""
#     answer = re.sub(r"\[/?INST\]", "", answer, flags=re.IGNORECASE)
#     stop_markers = ["surgicalscience", "[TABLE]", "Figure", "Page", "Chapter"]
#     for marker in stop_markers:
#         idx = answer.find(marker)
#         if idx != -1:
#             answer = answer[:idx]
#             break
#     answer = answer.strip()
#     answer = re.sub(r"•\s*", "- ", answer)
#     return answer


# # === MODULE CACHE (cases, tasks, guides) ===
# CASE_HEADER_RE = re.compile(r'(?mi)^(?:#{1,6}\s*)?Case\s+(\d+)\b')
# TASK_RE = re.compile(r'(?mi)^(?:Landmark|Task)\s+(\d+):?\s*(.*)')

# def _clean_line(s: str) -> str:
#     s = re.sub(r'^[\-\*\u2022•>\|]\s*', '', s.strip())
#     s = re.sub(r'\s+', ' ', s)
#     return s.strip(" :;-")

# def extract_cases_from_text(text: str) -> Dict[str, Dict[str, str]]:
#     lines = text.splitlines()
#     cases: Dict[str, Dict[str, str]] = {}
#     current_num, buffer = None, []

#     def flush():
#         nonlocal current_num, buffer
#         if current_num is None:
#             return
#         block = "\n".join(buffer).strip()
#         header_line = buffer[0] if buffer else ""
#         m = re.match(r'(?i)Case\s+(\d+)[:\-\s]*(.*)', header_line)
#         if m and m.group(2).strip():
#             title = _clean_line(m.group(2))
#         else:
#             title = ""
#             for ln in buffer[1:]:
#                 if not ln.strip():
#                     continue
#                 if CASE_HEADER_RE.match(ln):
#                     break
#                 title = _clean_line(ln)
#                 break
#         cases[f"Case {current_num}"] = {
#             "title": f"Case {current_num}: {title}" if title else f"Case {current_num}",
#             "text": block
#         }
#         current_num, buffer = None, []

#     for ln in lines:
#         m = CASE_HEADER_RE.match(ln)
#         if m:
#             flush()
#             current_num = int(m.group(1))
#             buffer = [ln]
#         elif current_num is not None:
#             buffer.append(ln)
#     flush()
#     return cases

# def extract_tasks_from_text(text: str) -> Dict[str, Dict[str, str]]:
#     tasks = {}
#     lines = text.splitlines()
#     current_num, current_title, body = None, None, []

#     def flush():
#         nonlocal current_num, current_title, body
#         if current_num is not None:
#             tasks[f"Task {current_num}"] = {
#                 "title": f"Task {current_num}: {current_title.strip() if current_title else ''}",
#                 "text": "\n".join(body).strip()
#             }
#         current_num, current_title, body = None, None, []

#     for idx, ln in enumerate(lines):
#         m = TASK_RE.match(ln)
#         if m:
#             flush()
#             current_num = int(m.group(1))
#             current_title = m.group(2).strip()
#             if not current_title:
#                 # fallback: peek ahead to next non-empty line
#                 if idx + 1 < len(lines) and lines[idx + 1].strip():
#                     current_title = _clean_line(lines[idx + 1])
#         else:
#             if current_num is not None:
#                 body.append(ln)

#     flush()
#     return tasks

# def list_available_modules(md_folder: str) -> Dict[str, str]:
#     module_map = {}
#     for md_path in glob.glob(os.path.join(md_folder, "*.md")):
#         fname = os.path.basename(md_path)
#         base = fname.lower().replace(" module book.md", "").replace(".md", "").strip()
#         module_map[base] = fname
#     return module_map

# def build_module_cache(md_folder: str) -> Dict[str, Dict]:
#     module_map = list_available_modules(md_folder)
#     cache = {}
#     for key, fname in module_map.items():
#         file_path = os.path.join(md_folder, fname)
#         try:
#             with open(file_path, encoding="utf-8") as f:
#                 text = f.read()

#             if re.search(r'(?mi)\bCase\s+\d+', text):
#                 parsed = extract_cases_from_text(text)
#                 cache[key] = {
#                     "type": "case",
#                     "count": len(parsed),
#                     "titles": [v["title"] for v in parsed.values()],
#                     "texts": {k: v["text"] for k, v in parsed.items()}
#                 }
#             elif re.search(r'(?mi)\b(Task|Landmark)\s+\d+', text):
#                 parsed = extract_tasks_from_text(text)
#                 cache[key] = {
#                     "type": "task",
#                     "count": len(parsed),
#                     "titles": [v["title"] for v in parsed.values()],
#                     "texts": {k: v["text"] for k, v in parsed.items()}
#                 }
#             else:
#                 cache[key] = {"type": "guide", "count": 0, "titles": [], "texts": {}}

#         except Exception:
#             cache[key] = {"type": "guide", "count": 0, "titles": [], "texts": {}}
#     return cache

# module_cache = build_module_cache(MARKDOWN_FOLDER)
# print(f"=== Module cache built: {{k:v['count'] for k,v in module_cache.items()}} ===")


# # === INTENT CLASSIFIER ===
# def classify_intent(query: str) -> str:
#     q = query.lower()
#     if re.search(r'\b(how many|number of|total)\s+(cases|tasks|landmarks)\b', q):
#         return "count"
#     if re.search(r'\b(list|what are|which|names?)\b.*\b(cases|tasks|landmarks)\b', q):
#         return "list"
#     if re.search(r'\b(show|give|display|open|present)\b.*\b(case|task|landmark)\s+\d+\b', q):
#         return "details"
#     return "general"


# # === QUERY HELPERS ===
# def extract_module_from_query(query: str, module_map: Dict[str, str]) -> Optional[str]:
#     query = query.lower()
#     for key in module_map.keys():
#         if key in query:
#             return key
#     return None


# # === MAIN QA FUNCTION ===
# def rag_qa(query: str, retriever, llm) -> Dict[str, Any]:
#     intent = classify_intent(query)
#     print(f"🔎 Intent classified as: {intent}")

#     module_map = list_available_modules(MARKDOWN_FOLDER)
#     module_name = extract_module_from_query(query, module_map)

#     if module_name and module_name in module_cache:
#         mod = module_cache[module_name]
#         if intent == "count":
#             return {
#                 "answer": f"There are {mod['count']} {mod['type']}s.",
#                 "sources": [{"source": module_map[module_name]}]
#             }
#         if intent == "list":
#             return {
#                 "answer": "\n".join(mod["titles"]) if mod["titles"] else f"No {mod['type']}s found.",
#                 "sources": [{"source": module_map[module_name]}]
#             }
#         if intent == "details":
#             m = re.search(rf"{mod['type']}\s+(\d+)", query, flags=re.IGNORECASE)
#             if m:
#                 idx = m.group(1)
#                 key = f"{mod['type'].capitalize()} {idx}"
#                 details = mod["texts"].get(key)
#                 if details:
#                     return {
#                         "answer": details,
#                         "sources": [{"source": module_map[module_name]}]
#                     }
#                 return {"answer": f"{key} not found.", "sources": []}

#     # === General RAG fallback ===
#     initial_docs = retriever.get_relevant_documents(query)
#     seen, unique_docs = set(), []
#     for d in initial_docs:
#         key = (d.metadata.get("source"), d.metadata.get("page"))
#         if key not in seen:
#             seen.add(key)
#             unique_docs.append(d)

#     context = "\n\n".join([d.page_content for d in unique_docs])
#     system_instructions = (
#         "You are a precise medical training QA assistant.\n"
#         "Answer ONLY the given QUESTION using the provided CONTEXT.\n"
#         "If the question asks for case/task names or counts, use the extracted cache values.\n"
#         "Do not mix content from other modules.\n"
#         "Do not answer more than one question.\n"
#         "Do not repeat the question.\n"
#         "Do not include greetings, filler, or company information.\n"
#         "Provide only the final concise answer.\n"
#         "If nothing relevant is found, reply exactly: Not found in documents."
#     )
#     prompt = f"{system_instructions}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"
#     raw_answer = llm(prompt)
#     final_answer = clean_answer(raw_answer)

#     sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in unique_docs]
#     return {"answer": final_answer, "sources": sources}


# # === API ENDPOINT ===
# @app.post("/ask", response_model=dict)
# async def ask_question(request: QuestionRequest):
#     try:
#         result = rag_qa(request.question, retriever, llm)
#         return result
#     except Exception as e:
#         return {"error": str(e)}
