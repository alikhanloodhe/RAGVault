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
    return inngest.Inngest(app_id="my-rag-application", is_production=False)

def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path

async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
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
            path = save_uploaded_pdf(uploaded)
            # Kick off the event and block until the send completes
            asyncio.run(send_rag_ingest_event(path))
            # Small pause for user feedback continuity
            time.sleep(0.3)
        st.success(f"✅ Triggered ingestion for: {path.name}")
        st.caption("You can upload another PDF if you like.")

async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            # Correct event name matching main.py
            name="rag/query_pdf",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return result[0]

def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])

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
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)

with col2:
    st.subheader("❓ Ask a Question")
    st.markdown("Retrieve information from the ingested PDFs.")
    
    with st.form("rag_query_form"):
        question = st.text_input("What would you like to know?", placeholder="e.g. What is the main topic of the document?")
        top_k = st.number_input("Number of context chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
        submitted = st.form_submit_button("Generate Answer")

        if submitted and question.strip():
            with st.spinner("🧠 Thinking..."):
                # Fire-and-forget event to Inngest for observability/workflow
                event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
                # Poll the local Inngest API for the run's output
                output = wait_for_run_output(event_id)
                answer = output.get("answer", "")
                sources = output.get("sources", [])

            st.markdown("### Answer")
            st.info(answer or "(No answer)")
            
            if sources:
                st.markdown("**Sources:**")
                for s in sources:
                    st.markdown(f"- 📄 `{s}`")
