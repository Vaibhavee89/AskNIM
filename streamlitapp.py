import os
import json
import time
import re
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# LangChain / NVIDIA NIM
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ----------------------------
# Environment & Config
# ----------------------------
load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY", "")

DATA_DIR = os.getenv("MEDICAL_PDF_DIR", "./medical_report")  # folder containing PDFs
MODEL_NAME = os.getenv("NVIDIA_CHAT_MODEL", "openai/gpt-oss-120b")
CHUNK_SIZE = 700
CHUNK_OVERLAP = 80
TOP_K = 8
FETCH_K = 32
LAMBDA_MMR = 0.5  # 0 = diversity, 1 = relevance

# ----------------------------
# Safety: triage & guardrails
# ----------------------------
EMERGENCY_KEYWORDS = [
    "chest pain", "shortness of breath", "stroke", "suicidal", "severe bleeding",
    "loss of consciousness", "seizure", "pregnancy bleeding", "anaphylaxis",
    "vision loss", "poison", "overdose", "intense headache", "worst headache",
    "high fever infant", "trauma", "fainting", "hemorrhage"
]

HIGH_RISK_KEYWORDS = [
    "dosage", "dose", "prescribe", "prescription", "start medication", "how many mg",
    "self medicate", "increase dose", "decrease dose", "drug interaction", "mix alcohol",
]

PHI_PATTERNS = [
    r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", # naive full name pattern (avoid logging)
    r"MRN\s*[:#]?\s*\w+",
    r"\b\d{10}\b",  # phone-like
]


def triage_intent(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in EMERGENCY_KEYWORDS):
        return "emergency"
    if any(k in ql for k in HIGH_RISK_KEYWORDS):
        return "high_risk"
    return "general"


def show_emergency_message():
    st.error(
        "⚠️ This may be an emergency. Please seek immediate medical care or call your local emergency number. "
        "This app cannot provide urgent medical advice."
    )


# ----------------------------
# Index building (cached)
# ----------------------------
@st.cache_resource(show_spinner=True)
def build_index(data_dir: str) -> Dict[str, Any]:
    """Load PDFs, split, and build FAISS index (cached)."""
    embeddings = NVIDIAEmbeddings()
    loader = PyPDFDirectoryLoader(data_dir)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    final_docs = splitter.split_documents(docs)

    # Ensure source metadata present
    for d in final_docs:
        d.metadata.setdefault("source", d.metadata.get("file_path") or d.metadata.get("source", "unknown"))
        if "page" not in d.metadata:
            # Many PDF loaders include 'page' already; if not present, default to 1
            d.metadata["page"] = d.metadata.get("page", 1)

    vector_store = FAISS.from_documents(final_docs, embeddings)
    return {"vs": vector_store, "count": len(final_docs)}


# ----------------------------
# LLM & Prompts
# ----------------------------
llm = ChatNVIDIA(model=MODEL_NAME, temperature=0.1, max_tokens=1024)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a careful clinical information assistant.
ONLY use the provided Context to answer. If the answer is not clearly supported by the Context, respond with
"I don't know based on the provided documents." Do NOT diagnose, prescribe, or provide dosing.
If the user's question implies emergencies or red-flag symptoms, advise seeking medical care promptly.
Use patient-friendly language.

Return strictly valid JSON with keys:
- answer (string; ≤ 120 words)
- red_flags (string; "none" if not applicable)
- confidence (one of: "low", "medium", "high")
- sources (array of objects: {{"file": string, "page": number}})

Context:
{context}

Question: {input}

JSON:
    """
)

# ----------------------------
# Utilities
# ----------------------------

def parse_llm_json(s: str) -> Dict[str, Any]:
    """Parse the LLM's JSON output robustly."""
    # Extract the first JSON block
    try:
        # Try direct parse
        return json.loads(s)
    except Exception:
        pass
    try:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    # Fallback minimal structure
    return {
        "answer": s.strip(),
        "red_flags": "unknown",
        "confidence": "low",
        "sources": []
    }


def render_sources(ctx_docs: List[Any], json_sources: List[Dict[str, Any]]):
    # Prefer structured sources from the model; fall back to context docs
    if json_sources:
        st.subheader("Sources")
        for s in json_sources:
            file = os.path.basename(str(s.get("file", "unknown")))
            page = s.get("page", "?")
            st.write(f"• {file} — p.{page}")
    else:
        st.subheader("Sources")
        for doc in ctx_docs:
            src = os.path.basename(str(doc.metadata.get("source", "unknown")))
            page = doc.metadata.get("page", "?")
            st.write(f"• {src} — p.{page}")


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="NVIDIA NIM Medical RAG", page_icon="🩺", layout="wide")
st.title("🩺 NVIDIA NIM Medical RAG (Safer & Grounded)")
st.caption("**Educational use only. Not a substitute for professional care.**")

with st.sidebar:
    st.header("Setup")
    st.write("**PDF Folder:**", DATA_DIR)
    build = st.button("📚 Build / Load Vector DB", type="primary")
    st.markdown("""
    **Notes**
    - Place your medical PDFs in the folder above (or set `MEDICAL_PDF_DIR`).
    - Requires `NVIDIA_API_KEY` in `.env`.
    """)

if build:
    if not os.path.isdir(DATA_DIR):
        st.error(f"Data directory not found: {DATA_DIR}")
    else:
        with st.spinner("Building index (first time may take a bit)..."):
            out = build_index(DATA_DIR)
            st.session_state.vdb = out["vs"]
            st.success(f"Vector store ready. Indexed chunks: {out['count']}")

# Main input
user_q = st.text_input("Ask about your medical documents", placeholder="e.g., What does my MRI report say about L4-L5?")
ask_clicked = st.button("🔎 Answer from documents", type="primary", disabled=not user_q)

# Guard: emergency / high-risk
if ask_clicked and user_q:
    intent = triage_intent(user_q)
    if intent == "emergency":
        show_emergency_message()
        st.stop()
    elif intent == "high_risk":
        st.warning("⚖️ This appears to involve medications or dosing. The assistant will provide only general information and will not prescribe or provide dosing.")

    if "vdb" not in st.session_state:
        st.warning("Please build the vector DB from the sidebar first.")
        st.stop()

    # Retrieval with MMR
    retriever = st.session_state.vdb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": LAMBDA_MMR},
    )

    # Chain
    doc_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
    chain = create_retrieval_chain(retriever, doc_chain)

    t0 = time.process_time()
    result = chain.invoke({"input": user_q})
    dt = time.process_time() - t0

    raw_answer = result.get("answer", "")
    ctx_docs = result.get("context", [])

    parsed = parse_llm_json(raw_answer)

    # Render
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Answer")
        st.write(parsed.get("answer", ""))
        rf = parsed.get("red_flags", "none")
        if rf and rf.lower() != "none":
            st.error(f"Red flags: {rf}")

        conf = parsed.get("confidence", "low").lower()
        if conf == "high":
            st.success("Confidence: high")
        elif conf == "medium":
            st.info("Confidence: medium")
        else:
            st.warning("Confidence: low")

    with col2:
        st.metric("Response time (s)", f"{dt:.2f}")
        render_sources(ctx_docs, parsed.get("sources", []))

    with st.expander("Show relevant context chunks"):
        for i, doc in enumerate(ctx_docs, start=1):
            src = os.path.basename(str(doc.metadata.get("source", "unknown")))
            page = doc.metadata.get("page", "?" )
            st.markdown(f"**Chunk {i}: {src} — p.{page}**")
            # Avoid printing potential PHI-heavy lines in logs; show in UI only
            st.write(doc.page_content)
            st.write("\n---\n")

# Footer disclaimer
st.caption(
    "This assistant summarizes information from your uploaded documents only and may be incomplete or outdated. "
    "Always consult a qualified clinician for diagnosis and treatment."
)
