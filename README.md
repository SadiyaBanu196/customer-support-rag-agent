# GigaCorp Customer Support RAG Agent

A conversational customer support assistant for a fictional company,
**GigaCorp**. It answers questions from a mock FAQ knowledge base using
Retrieval-Augmented Generation (RAG), cites the exact section and line it
pulled its answer from, and remembers earlier turns in the conversation.

**Live demo:** _https://customer-support-rag-agent-4gmp8ddbu4qbre6gorbayx.streamlit.app/_

## Architecture

```
User (browser)
   |
   v
Streamlit UI (app.py)
   |
   v
LangChain conversational RAG chain (rag_engine.py)
   |-- 1. History-aware retriever
   |      rewrites follow-up questions ("how much does it cost?") into
   |      standalone questions using chat history, then
   |-- 2. FAISS vector search
   |      over embedded FAQ chunks (local, persisted in faiss_index/)
   |-- 3. Groq LLM (openai/gpt-oss-120b)
          answers using ONLY the retrieved chunks, citing section + line
```

**Backend & orchestration:** Python + LangChain (`create_history_aware_retriever`
+ `create_retrieval_chain`), calling the Groq API (OpenAI-compatible,
free tier, very low latency) as the LLM.

**Knowledge base (RAG):** `data/gigacorp_faq.txt` — a mock FAQ covering
shipping policy, returns/refunds, business hours, and service tiers. Each
line is tagged `Line N:` under a `SECTION` heading; `rag_engine.py` parses
this into one retrievable chunk per line, with `section` and `line`
metadata attached, so every retrieved chunk can be cited precisely (e.g.
*"Shipping Policy, Line 4"*) rather than just naming the file.

**Vector store:** FAISS, built once from the FAQ using free, locally-run
`sentence-transformers/all-MiniLM-L6-v2` embeddings (no embedding API key
needed) and cached to disk in `faiss_index/` so it isn't rebuilt on every
run.

**Conversational memory:** The full chat history (as LangChain
`HumanMessage`/`AIMessage` objects) is stored in Streamlit's
`st.session_state` and passed into the chain on every turn. A
history-aware retriever step rewrites context-dependent follow-ups (e.g.
"Do you ship to India?" → "How much does that cost?") into standalone
questions before hitting the vector store, which is what lets the agent
correctly answer questions that only make sense given earlier turns.

**Sources & citations:** The QA prompt instructs the model to cite
`(Source: <Section>, Line <N>)` for every claim, and the UI additionally
shows a collapsible "Sources" panel under each answer listing the raw
retrieved chunks, so citations are both in the model's own text and
independently verifiable.

## Project structure

```
customer-support-rag/
├── app.py                  # Streamlit chat UI
├── rag_engine.py            # RAG pipeline: loading, embeddings, chain
├── data/
│   └── gigacorp_faq.txt     # Mock knowledge base
├── faiss_index/              # Auto-generated vector store cache (gitignored contents ok to commit or regenerate)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup (local)

1. Clone the repo and enter it:
   ```bash
   git clone <your-repo-url>
   cd customer-support-rag
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Get a **free** Groq API key at https://console.groq.com/keys.
4. Copy `.env.example` to `.env` and paste in your key:
   ```bash
   cp .env.example .env
   ```
   (Or just paste the key directly into the sidebar text box when the app
   is running — it's never written to disk or committed either way.)
5. Run the app:
   ```bash
   streamlit run app.py
   ```
6. Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deploying to Streamlit Community Cloud (free)

1. Push this repo to GitHub (make sure `.env` is **not** committed — it's
   in `.gitignore`).
2. Go to https://share.streamlit.io, sign in with GitHub, and click
   "New app".
3. Select this repository, branch `main`, and set the main file path to
   `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_actual_key_here"
   ```
5. Click **Deploy**. First build takes a couple of minutes (it downloads
   the embedding model). Subsequent runs are fast because the FAISS index
   is cached.

## Example conversation (demonstrating memory)

```
User: Do you ship to India?
Agent: Yes — GigaCorp ships to India via Standard International Shipping,
       which takes 7-10 business days. (Source: Shipping Policy, Line 4)

User: How much does that cost?
Agent: Standard International Shipping to India is free for orders over
       $75 USD. (Source: Shipping Policy, Line 6) If you need it faster,
       Express International Shipping to India takes 3-5 business days
       and costs $34.99. (Source: Shipping Policy, Line 5)
```

The second question ("How much does that cost?") only makes sense because
the agent remembers the first turn was about shipping to India — this is
the conversational memory requirement in action.

## Test credentials

No login is required to use the app. You only need your own free Groq API
key (see Setup step 3). No paid credentials, credit card, or trial account
are needed anywhere in this project.

## Notes on model choice

Groq deprecated `llama-3.3-70b-versatile` in mid-2026. This project uses
`openai/gpt-oss-120b`, Groq's current recommended free-tier general-purpose
model. If you hit free-tier rate limits, swap `GROQ_MODEL_NAME` in
`rag_engine.py` to `llama-3.1-8b-instant` for a faster, higher-rate-limit
alternative.
