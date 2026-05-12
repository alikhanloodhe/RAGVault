# In this file, we set up our FastAPI application and integrate it with Inngest for event handling.
# importing necessary libraries and modules for our application.
import logging 
from fastapi import FastAPI, UploadFile, File
import shutil
import inngest
import inngest.fast_api 
from inngest.experimental import ai
from dotenv import load_dotenv
import os
import uuid
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custome_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAQQueryResult


load_dotenv()  # Load environment variables from .env file

inngest_client = inngest.Inngest(
    app_id = "my-rag-application",
    logger = logging.getLogger("uvicorn"),
    is_production=os.getenv("RENDER") is not None,
    serializer= inngest.PydanticSerializer()

)  # Initialize Inngest client

# For AI heavy work we will wrap it using inngest to maintain log and observability
# Inngest provides a decorator to wrap our functions and track their execution as events.
# Decorator Function
@inngest_client.create_function(
    fn_id='RAG: Inngest PDF ',
    trigger= inngest.TriggerEvent(event="rag/ingest_pdf")
)

async def rag_ingest_pdf(ctx: inngest.Context):
    def _get_first(data: dict, *keys: str):
        for key in keys:
            value = data.get(key)
            if value is not None:
                return value
        return None

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = _get_first(ctx.event.data, "pdf_path", "pdf path", "pdfPath")
        if not pdf_path:
            raise ValueError(
                "Missing PDF path in event data. "
                "Send { 'pdf_path': 'your.pdf' } (preferred). "
                f"Got keys: {sorted(list(ctx.event.data.keys()))}"
            )
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        dim = len(vecs[0]) if vecs else None
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage(dim=dim).upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    
    # Wait for 1 hour, then automatically clean up the file and embeddings
    await ctx.step.sleep("wait-1-hour", datetime.timedelta(hours=1))
    
    def _delete_file():
        pdf_path = _get_first(ctx.event.data, "pdf_path", "pdf path", "pdfPath")
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
            
    await ctx.step.run("delete-file", _delete_file)
    await ctx.step.run("delete-embeddings", lambda: QdrantStorage().delete_points(chunks_and_src.source_id))

    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5, source_id: str = None) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, source_id=source_id, top_k=top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    source_id = ctx.event.data.get("source_id")

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k, source_id), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env")

    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    adapter = ai.openai.Adapter(
        auth_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        model=groq_model,
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful RAG assistant. Answer using only the provided context.",
                },
                {"role": "user", "content": user_content},
            ],
        },
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()  # Create a FastAPI application instance

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Return the absolute path on the server so Inngest can find it
    return {"path": os.path.abspath(file_path)}

inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai])  # Integrate Inngest with FastAPI