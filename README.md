# CODDY

A local retrieval-augmented chatbot. Upload documents, ask questions, get answers
grounded in those documents with citations pointing back at the source.

Everything runs on your machine. No hosted API is called: retrieval and generation
both execute locally, and the only network access is the one-time model download.

![status](https://img.shields.io/badge/tests-49%20passing-brightgreen)

---

## What it does

- **Reads** PDF, DOCX, TXT, CSV and Markdown
- **Finds** relevant passages using vector search combined with BM25 keyword search
- **Answers** from a local language model, refusing when the documents do not contain the answer
- **Cites** the specific chunks each answer came from

## Requirements

| | |
|---|---|
| Python | 3.11+ (developed on 3.13) |
| RAM | 8GB |
| GPU | Optional. Strongly recommended — see [Performance](#performance) |
| Disk | ~10GB for the model |

## Setup

```bash
git clone https://github.com/santhosh220z/CODDY-THE-CHAT-BOT.git
cd CODDY-THE-CHAT-BOT

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# torch needs the CUDA index for NVIDIA GPUs
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Configure:

```bash
cp .env.example .env          # Windows: copy .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # paste into SECRET_KEY
```

`SECRET_KEY` must be set. The app refuses to start on the example value, because
anyone knowing it could mint a valid session token for any account.

Download a model (~8GB). Any instruct-tuned causal LM works:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Instruct-2507')"
```

Then set `HF_MODEL_ID` in `.env` to either the downloaded path or the model id.

Run:

```bash
uvicorn backend.main:app --reload
```

Open <http://127.0.0.1:8000> and create the first account — it becomes the
administrator. Everyone after that is a regular user.

## Configuration

All settings live in `.env`. See `.env.example` for the annotated list. The ones
that matter most:

| Setting | Default | Notes |
|---|---|---|
| `HF_MODEL_ID` | `Qwen/Qwen3-4B-Instruct-2507` | Must fit your VRAM at 4-bit |
| `LOAD_IN_4BIT` | `true` | Turning this off needs roughly 2x the VRAM |
| `MAX_NEW_TOKENS` | `1024` | Lower this to shorten answers and speed things up |
| `MODEL_MAX_CONTEXT` | `8192` | Longest prompt the model will see |
| `VECTOR_STORE_TYPE` | `chroma` | Or `faiss` |
| `RAG_TOP_K` | `5` | Chunks passed to the model |
| `BM25_WEIGHT` | `0.4` | 0 = pure vector, 1 = pure keyword |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Overlap must be smaller than size |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Changing it requires re-uploading |

## Performance

Measured on an RTX 3050 Laptop (6GB VRAM):

| | |
|---|---|
| Model load | ~20–35s, once |
| Peak VRAM | 2.8GB of 6GB |
| Time to first token | ~1.5–2s |
| **Throughput** | **~3.2 tok/s** |

That throughput is the honest limitation. A 150-token answer takes around 48
seconds. The cause is `bitsandbytes`: 4-bit nf4 weights are dequantised on every
token, which costs more than the maths itself.

Things that help:

- **Lower `MAX_NEW_TOKENS`.** Halving it roughly halves answer time. This is the
  cheapest lever and needs no extra software.
- **A smaller model.** 1.7B or 2B variants run faster at some cost in quality.
- **Ollama, optionally.** `backend/services/llm.py` also implements an Ollama
  backend, selected with `LLM_PROVIDER=ollama` and no code changes. llama.cpp
  reads pre-quantised weights directly, so it is substantially faster than
  bitsandbytes for the same model. Entirely optional — skip it unless the
  throughput above is a problem for you.

## Tests

```bash
pytest backend/tests -v
```

49 tests. The suite includes regression tests for three vulnerabilities this
project previously had; see below.

## Security notes

**Tenant isolation.** Retrieval is scoped to the uploading user. The vector store's
`search()` takes a required `user_id` rather than an optional filter, because a
filter with a default is a filter that eventually gets omitted — and omitting it
would return every user's documents to anyone who asked. Both the vector index and
the BM25 corpus derive their scope from one function, so they cannot disagree.

**No self-registration as admin.** `role` is not a field on the registration
schema. It previously was, which let anyone POST `{"role": "admin"}`.

**Output is sanitised.** Assistant text can quote an uploaded document, so it is
untrusted input. Everything rendered into the page passes through DOMPurify.
User-controlled values — filenames, session titles — are set with `textContent`.

**Uploads are validated.** Extension *and* magic bytes, so a binary renamed to
`.pdf` is rejected before reaching a parser. Files are streamed to disk with the
size cap enforced during the write, and the stored name is a generated UUID.

**Passwords** are hashed with bcrypt directly (passlib is unmaintained and breaks
against modern bcrypt). Input past 72 bytes is rejected rather than silently
truncated. Sign-in is rate limited to 5 attempts per 15 minutes.

**Bugs found by running this, not by reading it.** Worth recording, because each
was invisible in review:

- History was ordered ascending before taking N, returning the *oldest* messages,
  so long conversations lost their actual context.
- `CHUNK_OVERLAP` was computed and then discarded whenever the combined chunk
  exceeded the size limit — which is almost always. Overlap was effectively zero.
- A stream timeout of 60 seconds meant any slow generation was truncated mid-answer.
- SSE frames split across network reads were silently dropped.
- `pydantic.EmailStr` performs a DNS lookup per registration, rejecting any
  address whose domain does not resolve.

## Limitations

- Single-process. The rate limiter is in-memory and would not be shared across
  workers.
- Admins can read any session. That is intentional and matches the admin's access
  to all documents, but it does mean the admin role is a trusted one.
- Retrieval is keyword + vector. There is no reranker, so highly relevant chunks
  can lose their place to lexically similar irrelevant ones.
- Scanned PDFs need OCR and will be rejected with a message saying so.

## Project layout

```text
backend/
  api/          auth, documents, chat endpoints
  auth/         password hashing and JWT
  database/     engine, session factory, declarative base
  models/       SQLAlchemy models
  rag/          extraction, chunking, embeddings, vector store, pipeline
  services/     llm providers (transformers, ollama)
  tests/        pytest suite
frontend/       plain HTML, CSS and JS - no framework, no CDN
```

## Licence

MIT