# 🤖 CODDY – THE CHAT BOT

**CODDY** is a production-ready, feature-rich Retrieval-Augmented Generation (RAG) chatbot designed to answer complex questions using documents uploaded to a private knowledge base. Powered by a FastAPI backend and a responsive glassmorphic HTML5/Tailwind CSS frontend, it integrates local LLM execution (via Ollama and DeepSeek-R1) and modular vector databases (ChromaDB and FAISS) to ensure maximum privacy, speed, and versatility.

---

## 🚀 Key Features

*   **Multiformat Document Support:** Upload and index `PDF`, `DOCX`, `TXT`, `CSV`, and `Markdown` documents.
*   **Persistent Hybrid Search:** Combines semantic vector embeddings (Sentence Transformers) and keyword search (BM25) with weighted scoring (Reciprocal Rank Fusion).
*   **Modular Architecture:** Support for switching vector databases (ChromaDB / FAISS) and LLM providers (Ollama / Hugging Face) purely via configuration.
*   **Interactive Citations:** Responses include clickable source citations showing the exact matching segments of the source document in a popup modal.
*   **Short & Long-Term Memory:** Stored session history in database, context-aware RAG pipelines using short-term memory (last 10 messages).
*   **Voice Integration:** Supports Hands-Free voice input (Speech-to-Text) and output (Text-to-Speech) using the native Web Speech API.
*   **Admin Dashboard:** Monitor system health, user counts, storage utilization (uploads and vector store indexes), and active model properties in real-time.
*   **Secure Authentication:** Secure user registration, password hashing (bcrypt), and stateful token validation (JWT).

---

## 📁 Repository Structure

```text
CODDY-THE-CHAT-BOT/
├── backend/
│   ├── api/             # API Router endpoints (auth, chat, docs, admin)
│   ├── auth/            # Security keys and password hashing utilities
│   ├── database/        # Database engines and connection logic
│   ├── models/          # SQLAlchemy Database Models
│   ├── rag/             # RAG logic (extraction, chunking, embeddings, fusion)
│   ├── services/        # LLM Provider abstractions (Ollama & Hugging Face)
│   ├── utils/           # Configuration files and structured logger
│   └── main.py          # FastAPI Entrypoint and Static Asset Mounting
├── frontend/
│   ├── css/             # Custom glassmorphism variables and styling
│   ├── js/              # Application controller (chat streams, voice API, auth)
│   └── index.html       # Landing and Dashboard user interface markup
├── uploads/             # Persistent local uploaded document folder
├── vector_store/        # Persistent Chroma/FAISS directory
├── Dockerfile           # FastAPI + UI single container build schema
├── docker-compose.yml   # Docker Compose orchestration
├── requirements.txt     # Python Dependencies
├── .env.example         # Template configuration settings
└── README.md            # Setup and Deployment Documentation
```

---

## ⚙️ Local Machine Installation

### Prerequisites
*   [Python 3.10+](https://www.python.org/) installed on your machine.
*   [Ollama](https://ollama.com/) installed and running.
    *   Pull the default model locally: `ollama pull deepseek-r1:1.5b`

### Steps
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/CODDY-THE-CHAT-BOT.git
    cd CODDY-THE-CHAT-BOT
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Copy the template variables and adjust if necessary:
    ```bash
    cp .env.example .env
    ```

5.  **Run the Server:**
    Start the FastAPI application:
    ```bash
    python backend/main.py
    ```
    *Alternatively, run with uvicorn directly:*
    ```bash
    uvicorn backend.main:app --reload --port 8000
    ```

6.  **Access the Chatbot:**
    Open your browser and navigate to:
    [http://localhost:8000](http://localhost:8000)

---

## 🐳 Docker Deployment

To launch the entire stack inside Docker, simply run:

```bash
docker-compose up --build
```

### Note on local Ollama connectivity:
*   The `docker-compose.yml` configures `OLLAMA_BASE_URL` as `http://host.docker.internal:11434`. 
*   If you are running Ollama on Windows/macOS, this resolves automatically to your host network. Make sure your local Ollama is configured to bind to all interfaces by setting the environment variable `OLLAMA_HOST=0.0.0.0` before launching the Ollama desktop app.

---

## 🧪 Running Unit Tests

To run the automated test suite verifying auth tokens, chunk splits, extraction pipelines, and vector database indices:

```bash
pytest backend/tests/test_rag.py -v
```

---

## 🌐 Multi-Cloud Deployment Guide

For production environments, configure `DATABASE_URL` to point to a managed SQL database (e.g. PostgreSQL) instead of SQLite.

### 1. AWS (ECS + Fargate + RDS)
*   **Database:** Provision an Amazon RDS PostgreSQL instance.
*   **Deployment:**
    1. Build the Docker image and push it to AWS ECR (Elastic Container Registry).
    2. Create an ECS Task Definition using the Fargate launch type.
    3. Expose port `8000` and pass variables (`DATABASE_URL`, `SECRET_KEY`, `LLM_PROVIDER`) via AWS Secrets Manager.
    4. Attach an Application Load Balancer (ALB) to handle incoming traffic.
    5. Set persistent folders (`/app/uploads`) to mount onto AWS EFS (Elastic File System) if hosting multiple instances.

### 2. Google Cloud Platform (Cloud Run + Cloud SQL)
*   **Database:** Create a Google Cloud SQL PostgreSQL database instance.
*   **Deployment:**
    1. Build and submit your Docker container to Artifact Registry using Google Cloud Build:
       ```bash
       gcloud builds submit --tag gcr.io/your-project-id/coddy-chatbot
       ```
    2. Deploy container to Google Cloud Run:
       ```bash
       gcloud run deploy coddy-chatbot \
         --image gcr.io/your-project-id/coddy-chatbot \
         --platform managed \
         --allow-unauthenticated \
         --set-env-vars="DATABASE_URL=postgresql://user:pass@/dbname?host=/cloudsql/conn-name"
       ```
    3. Connect Cloud Run to Cloud SQL via SQL Connection Name.

### 3. Microsoft Azure (App Service)
*   **Database:** Create an Azure Database for PostgreSQL server.
*   **Deployment:**
    1. Create an Azure App Service choosing "Docker Container" and runtime "Linux".
    2. Configure "Deployment Center" to pull from Docker Hub or a private registry.
    3. Under App Service Configuration, add the Application Settings:
       * `DATABASE_URL`: Your Azure PostgreSQL connection string.
       * `SECRET_KEY`: JWT Signing Key.
       * `VECTOR_STORE_TYPE`: `chroma` (App Service handles local path persistence under standard storage).

### 4. Railway (Fast deployment)
1. Push this repository to GitHub.
2. Link Railway to your GitHub repository.
3. Add the **PostgreSQL Plugin** to your Railway project.
4. Railway automatically populates `DATABASE_URL`. Map the rest of your variables (`SECRET_KEY`, `VECTOR_STORE_TYPE`, `LLM_PROVIDER`) under variables settings.
5. Deployment will compile and run automatically.

### 5. Render
1. Create a new "Web Service" on Render and connect your GitHub repository.
2. Set Environment to "Docker".
3. Under Environment Variables, add:
   * `DATABASE_URL` (provision a Render PostgreSQL instance and copy external connection string).
   * `SECRET_KEY` (your private key).
   * `LLM_PROVIDER` (if connecting to a remote LLM API like Hugging Face, or set to `ollama` if routing to an external endpoint).
4. Add a "Disk" mount at `/app/uploads` and `/app/vector_store` to ensure local index files persist between deploys.
