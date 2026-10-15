# CODDY

A local retrieval-augmented chatbot. Upload documents, ask questions, get answers
grounded in those documents with citations pointing back at the source.

Everything runs on your machine. No hosted API is called: retrieval and generation
both execute locally, and the only network access is the one-time model download.

![status](https://img.shields.io/badge/tests-104%20passing-brightgreen)

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

Open <http://127.0.0.1:8000>. There is no sign-in step: the server provisions a
single local account on first use and the browser picks up a session from it. The
account owns your documents and cannot be signed into — its password is a random
secret generated once and discarded — so **anyone who can reach the port can read
your documents**. Bind to localhost, as above, and it is reachable only from your
own machine.

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

Measured on an RTX 3050 Laptop (6GB VRAM), Qwen3-4B-Instruct-2507 at 4-bit nf4,
warm page cache, one short document indexed:

| | |
|---|---|
| Heavy library imports | ~8s, once per process, backgrounded at startup |
| Model load | ~18s, once per process, backgrounded at startup |
| Embedding model load | ~6s, once per process, backgrounded at startup |
| Ready to answer | **~32s after launch** |
| First question from cold | ~29s if sent during warm-up, ~7s once warm |
| Question, everything warm | ~5.9s complete |
| Time to first token | ~0.6s warm |
| **Decode throughput** | **8–9 tok/s short bursts, ~5 tok/s sustained** |
| VRAM with both models resident | 4.4GB of 6GB |

The first question of a session is the slow one, and it is slow for a reason worth
knowing: nothing is answered until the 4B model is resident, and reading 7.7GB of
weights off disk takes about 18 seconds. The UI names which wait you are in, so a
slow start is not a mystery. Setting `PREWARM_MODEL=false` moves that cost onto
your first question instead of hiding it in the background.

The embedding model is kept on the GPU, which is why VRAM sits at 4.4GB rather
than 2.6GB. `EMBEDDING_DEVICE=cpu` frees that 1.8GB and speeds up the model load,
at the cost of a few tens of milliseconds per query. Worth doing on a 6GB card.

Decode speed is stable — measurements over a sustained 150-token generation
showed no thermal decay, and prompt length, sampling, `repetition_penalty` and
attention implementation made no measurable difference. The bottleneck is
`bitsandbytes`, which dequantises 4-bit weights on every token.

If you want more headroom:

- **Lower `MAX_NEW_TOKENS`.** Answers usually finish well before the limit, and
  shortening them cuts wall-clock proportionally.
- **A smaller model.** 1.7B or 2B variants run faster at some cost in quality.
- **Ollama, optionally.** `backend/services/llm.py` also implements an Ollama
  backend, selected with `LLM_PROVIDER=ollama` and no code changes. llama.cpp
  reads pre-quantised weights directly. Entirely optional — skip it unless the
  throughput above is a problem for you.

> A previous version of this README claimed ~3 tok/s. That figure divided total
> wall-clock — including an 18s model load and prompt prefill — by token count,
> which charges fixed startup cost to every token. The number above separates
> them.

## Tests

```bash
pytest backend/tests -v
```

82 tests. The suite includes regression tests for three vulnerabilities this
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
- Streaming fed CPU tensors to a CUDA model, so every streamed generation raised
  a device-mismatch error.

A later audit of the running system found a second batch, none of which the
existing suite could see because it never exercised the path:

- A typo'd email address returned **500, not 422**. The validation handler put a
  `ValueError` instance in the JSON body, which raised `TypeError` inside the
  handler. This is the most common client mistake there is.
- The SSE generator wrote through a *different* database session than the one the
  request validated against — a time-of-check/time-of-use gap.
- `split_text` could emit a chunk larger than `CHUNK_SIZE`. A single token over
  the limit (minified JSON, an unbroken string) was appended whole, because the
  hard-slice path only ran in one branch.
- A failed reindex **destroyed the document's chunks**: the delete committed
  before embedding, so `db.rollback()` could not undo it. SQL and the vector
  index were left permanently disagreeing.
- The login rate limiter keyed on `X-Forwarded-For`, which any caller can set —
  a fresh value per request defeated the brute-force limit entirely.
- Two concurrent generations each allocated their own KV cache for
  `MODEL_MAX_CONTEXT` tokens, which does not fit a 6GB card beside resident
  weights. Generation is now serialised.
- Two simultaneous registrations could both observe an empty users table and both
  become administrators.
- `GET /api/health` constructed the LLM provider, so a provider that failed to
  build turned the liveness probe into a 500.

The first two were found by the end-to-end run; the rest by an audit that read
every file and reproduced each finding. Neither the original review nor 64
passing tests had surfaced them.

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