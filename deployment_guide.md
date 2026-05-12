# Deployment Guide for Your RAG Application

Deploying this architecture requires moving each local component to a production-ready environment. The good news is that your architecture separates concerns well, making it easy to deploy using modern cloud platforms.

## Architecture Shift: Local vs. Production

| Component | Local Development | Production Environment |
| :--- | :--- | :--- |
| **Database** | Docker Qdrant | **Qdrant Cloud** (Managed) or self-hosted VM |
| **Orchestration** | `npx inngest-cli dev` | **Inngest Cloud** |
| **Backend (FastAPI)** | `uvicorn main:app` | **Render / Railway / Fly.io** |
| **Frontend (Streamlit)**| `streamlit run` | **Streamlit Community Cloud / Render** |

---

## Step 1: Deploy Qdrant (Vector Database)

Since you are running Qdrant in Docker locally, you have two options for production:

1. **Qdrant Cloud (Recommended & Easiest):**
   - Sign up for [Qdrant Cloud](https://cloud.qdrant.io/) (they have a generous free tier).
   - Create a new cluster and get the **Cluster URL** and **API Key**.
   - Update your `vector_db.py` or `.env` to connect to this remote URL instead of `http://localhost:6333`.

2. **Self-Hosted:**
   - Provision a VM on AWS (EC2) or DigitalOcean.
   - Run the exact same Qdrant Docker command you use locally.
   - Expose the port (6333) and secure it behind a reverse proxy (like Nginx) with HTTPS and basic auth.

## Step 2: Set Up Inngest Cloud

In production, you **do not** run the `npx inngest-cli dev` server. Inngest operates as a managed platform.

1. Create an account at [Inngest.com](https://www.inngest.com/).
2. Create a new environment/project and get your **Inngest Event Key** and **Signing Key**.
3. Inngest Cloud will handle receiving the events and pushing them to your FastAPI app.

## Step 3: Deploy FastAPI Backend (`main.py`)

Your FastAPI app acts as the worker that actually executes the Inngest functions (`rag_ingest_pdf` and `rag_query_pdf_ai`). It needs to be hosted on a public URL so Inngest Cloud can reach it.

1. **Choose a Provider:** Use a Platform-as-a-Service (PaaS) like **Render**, **Railway**, or **Fly.io** (they all support standard Docker or Python deployments).
2. **Setup the Service:**
   - Point the PaaS to your GitHub repository.
   - Set the start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. **Environment Variables needed in your PaaS:**
   - `GROQ_API_KEY`: Your Groq token.
   - `QDRANT_URL`: URL to your Qdrant Cloud instance.
   - `QDRANT_API_KEY`: Your Qdrant Cloud API key.
   - `INNGEST_SIGNING_KEY`: From your Inngest Cloud dashboard (secures communication so only Inngest can trigger your functions).
4. **Link to Inngest:**
   - Once deployed, grab the public URL of your backend (e.g., `https://my-rag-backend.onrender.com`).
   - Go to your Inngest Cloud dashboard, add a new app, and provide the webhook URL: `https://my-rag-backend.onrender.com/api/inngest`.

## Step 4: Deploy Streamlit Frontend (`streamlit_app.py`)

Your Streamlit app is the client. It pushes events to Inngest and polls for the results.

1. **Choose a Provider:** 
   - **Streamlit Community Cloud** (Free, connects directly to GitHub).
   - Alternatively, deploy it on the same PaaS you used for FastAPI (Render, Railway).
2. **Environment Variables needed for Streamlit:**
   - `INNGEST_EVENT_KEY`: Needed so your frontend can push events securely to Inngest Cloud.
   - `INNGEST_API_BASE`: Update this to the production Inngest REST API instead of the local dev server (e.g., `https://api.inngest.com/v1`).
3. **Code Adjustment (`streamlit_app.py`):**
   - Make sure `is_production=True` is set in the frontend when initializing the Inngest client in production.
   - When deploying, ensure `INNGEST_EVENT_KEY` is provided in Streamlit secrets so `inngest.Event()` calls are routed to your production Inngest account.

## Summary Checklist for Go-Live

- [ ] Push code to GitHub.
- [ ] Create Qdrant Cloud cluster -> get URL/Key.
- [ ] Create Inngest Cloud account -> get Signing Key / Event Key.
- [ ] Deploy FastAPI to Render/Railway with `INNGEST_SIGNING_KEY`, `GROQ_API_KEY`, and Qdrant credentials.
- [ ] Register FastAPI URL (`.../api/inngest`) in the Inngest Cloud dashboard.
- [ ] Deploy Streamlit with `INNGEST_EVENT_KEY` and updated `INNGEST_API_BASE`.
