import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Knowledge Base", page_icon="📚", layout="wide")

# Modern UI Styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1e3a8a;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1e40af;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    .stNumberInput>div>div>input {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def get_inngest_client() -> inngest.Inngest:
    # Match the app_id in main.py
    # If INNGEST_EVENT_KEY is in .env, assume we want to talk to the Cloud
    is_prod = os.getenv("RENDER") is not None or os.getenv("INNGEST_EVENT_KEY") is not None
    return inngest.Inngest(app_id="my-rag-application", is_production=is_prod)

def save_uploaded_pdf(file) -> str:
    backend_url = os.getenv("FASTAPI_BACKEND_URL", "http://127.0.0.1:10000").rstrip("/")
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    resp = requests.post(f"{backend_url}/upload", files=files)
    resp.raise_for_status()
    return resp.json()["path"]

def query_backend(question: str, top_k: int, source_id: str) -> dict:
    backend_url = os.getenv("FASTAPI_BACKEND_URL", "http://127.0.0.1:10000").rstrip("/")
    resp = requests.post(
        f"{backend_url}/query",
        json={
            "question": question,
            "top_k": top_k,
            "source_id": source_id,
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()

async def send_rag_ingest_event(pdf_path: str, source_id: str) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": pdf_path,
                "source_id": source_id,
            },
        )
    )

st.title("📚 RAG Knowledge Base")
st.markdown("Upload PDFs to build your knowledge base, and ask questions to get AI-powered answers based on the documents.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload a PDF")
    st.markdown("Add a new document to the vector database.")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=False)

    if uploaded is not None:
        with st.spinner("Uploading and triggering ingestion..."):
            # Check if we already ingested this exact file during this session
            if st.session_state.get("uploaded_filename") != uploaded.name:
                # Generate a unique source_id for this upload
                import uuid
                source_id = f"{uploaded.name}_{uuid.uuid4().hex[:8]}"
                st.session_state["source_id"] = source_id
                st.session_state["uploaded_filename"] = uploaded.name
                
                path = save_uploaded_pdf(uploaded)
                # Kick off the event and block until the send completes
                asyncio.run(send_rag_ingest_event(path, source_id))
                # Small pause for user feedback continuity
                time.sleep(0.3)
        st.success(f"✅ Triggered ingestion for: {uploaded.name}")
        st.caption("This document and its data will automatically be deleted in 1 hour.")

async def send_rag_query_event(question: str, top_k: int, source_id: str) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            # Correct event name matching main.py
            name="rag/query_pdf",
            data={
                "question": question,
                "top_k": top_k,
                "source_id": source_id,
            },
        )
    )
    return result[0]

def _inngest_api_base() -> str:
    is_prod = os.getenv("RENDER") is not None or os.getenv("INNGEST_EVENT_KEY") is not None
    if is_prod:
        return "https://api.inngest.com/v1"
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    headers = {}
    
    is_prod = os.getenv("RENDER") is not None or os.getenv("INNGEST_EVENT_KEY") is not None
    if is_prod:
        rest_api_key = os.getenv("INNGEST_REST_API_KEY")
        if rest_api_key:
            headers["Authorization"] = f"Bearer {rest_api_key}"

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])

def get_run(run_id: str) -> dict:
    url = f"{_inngest_api_base()}/runs/{run_id}"
    headers = {}
    is_prod = os.getenv("RENDER") is not None or os.getenv("INNGEST_EVENT_KEY") is not None
    if is_prod:
        rest_api_key = os.getenv("INNGEST_REST_API_KEY")
        if rest_api_key:
            headers["Authorization"] = f"Bearer {rest_api_key}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json().get("data", {})
    return {}

def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                output_val = run.get("output")
                
                # If output is missing from list endpoint, fetch the specific run
                if not output_val:
                    run_id = run.get("id") or run.get("run_id")
                    if run_id:
                        full_run = get_run(run_id)
                        output_val = full_run.get("output")
                
                output_val = output_val or {}
                
                if isinstance(output_val, str):
                    import json
                    try:
                        output_val = json.loads(output_val)
                    except Exception:
                        pass
                        
                # Sometimes Inngest wraps output in a list
                if isinstance(output_val, list) and len(output_val) > 0:
                    output_val = output_val[0]
                
                # Ensure it's a dict before returning
                if not isinstance(output_val, dict):
                    output_val = {"raw_output": output_val}
                
                # Since the run is Completed, the output is final. Return it immediately.
                return output_val
            
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        
        if time.time() - start > timeout_s:
            # Instead of crashing, just return empty so the UI says (No answer) gracefully
            return {}
            
        time.sleep(poll_interval_s)

with col2:
    st.subheader("❓ Ask a Question")
    st.markdown("Retrieve information from the ingested PDFs.")
    
    with st.form("rag_query_form"):
        question = st.text_input("What would you like to know?", placeholder="e.g. What is the main topic of the document?")
        top_k = st.number_input("Number of context chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
        submitted = st.form_submit_button("Generate Answer")

        if submitted and question.strip():
            answer = ""
            sources = []
            source_id = st.session_state.get("source_id")
            if not source_id:
                st.error("Please upload a PDF first before asking a question!")
            else:
                try:
                    with st.spinner("🧠 Thinking..."):
                        output = query_backend(question.strip(), int(top_k), source_id)
                    answer = output.get("answer", "")
                    sources = output.get("sources", [])
                except requests.HTTPError as exc:
                    st.error(f"Query failed: {exc.response.text}")
                except requests.RequestException as exc:
                    st.error(f"Could not reach the FastAPI backend: {exc}")

            st.markdown("### Answer")
            st.info(answer or "(No answer)")
            
            if sources:
                st.markdown("**Sources:**")
                for s in sources:
                    st.markdown(f"- 📄 `{s}`")
