# 📚 RAGVault

A highly scalable, distributed Retrieval-Augmented Generation (RAG) application. RAGVault allows users to securely upload PDF documents and ask context-aware questions powered by Large Language Models. 

Built with an advanced distributed architecture, RAGVault uses serverless orchestration, cloud-native vector databases, and ephemeral storage to ensure infinite scalability and tenant isolation.

## 🚀 Key Features

* **Distributed File Processing:** A lightweight Streamlit frontend streams uploaded documents directly to a remote FastAPI backend via network POST requests.
* **Serverless Orchestration (Inngest):** Heavy, long-running tasks like document chunking, embedding generation, and vector indexing are offloaded to background workers using Inngest.
* **Memory-Optimized Embeddings:** Uses the Hugging Face Inference API (`sentence-transformers/all-MiniLM-L6-v2`) to generate embeddings over the network, completely bypassing strict PaaS RAM limits (e.g., Render Free Tier OOM crashes).
* **Tenant Isolation:** Every uploaded document is assigned a unique `source_id`. All vector searches in the Qdrant database are strictly filtered by this ID using payload indexes, guaranteeing users only query their own data.
* **Self-Healing Storage:** To prevent storage bloat, the Inngest ingestion pipeline sleeps for 10 minutes upon completion, then automatically wakes up to securely delete the raw PDF from the ephemeral disk and purge the associated vector embeddings from Qdrant.

## 🛠️ Architecture & Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (Deployed on Streamlit Community Cloud)
* **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Deployed on Render)
* **Orchestration:** [Inngest Cloud](https://www.inngest.com/)
* **Vector Database:** [Qdrant Cloud](https://qdrant.tech/)
* **Embeddings:** Hugging Face Inference API
* **LLM Provider:** Groq (`llama-3.3-70b-versatile`)
* **Document Processing:** LlamaIndex

## 💻 Local Development

### Prerequisites
* Python 3.11+
* `uv` package manager
* An Inngest Dev Server instance (`npx inngest-cli dev`)

### Installation

1. Clone the repository and install dependencies using `uv`:
   ```bash
   uv sync
   ```

2. Create a `.env` file in the root directory:
   ```env
   # LLM & Models
   GROQ_API_KEY=your_groq_api_key
   HUGGINGFACE_API_KEY=your_huggingface_key
   
   # Vector Database
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=your_qdrant_key
   
   # Networking
   FASTAPI_BACKEND_URL="http://127.0.0.1:10000"
   ```

3. Start the FastAPI backend:
   ```bash
   uv run uvicorn main:app --port 10000
   ```

4. Start the Inngest Dev Server (in a new terminal):
   ```bash
   npx inngest-cli@latest dev -u http://127.0.0.1:10000/api/inngest
   ```

5. Start the Streamlit Frontend (in a new terminal):
   ```bash
   uv run streamlit run streamlit_app.py
   ```

## ☁️ Production Deployment

### Backend (Render)
1. Deploy the repository as a **Web Service** on Render.
2. Set the Start Command to: `uv run uvicorn main:app --host 0.0.0.0 --port 10000`
3. Add your environment variables (`GROQ_API_KEY`, `HUGGINGFACE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `INNGEST_EVENT_KEY`, `INNGEST_SIGNING_KEY`).

### Frontend (Streamlit Community Cloud)
1. Connect your repository to Streamlit Community Cloud.
2. Under "Advanced Settings" > "Secrets", provide the minimal variables required to reach the backend:
   ```toml
   FASTAPI_BACKEND_URL="https://your-render-url.onrender.com"
   INNGEST_EVENT_KEY="your_inngest_event_key"
   INNGEST_REST_API_KEY="your_inngest_rest_api_key"
   ```

## 📄 License
MIT License
