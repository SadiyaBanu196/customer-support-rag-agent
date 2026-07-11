"""
rag_engine.py
--------------
Core RAG (Retrieval-Augmented Generation) logic for the GigaCorp Customer
Support Agent.

Responsibilities:
1. Load the mock FAQ knowledge base and tag every chunk with a precise
   line-number citation.
2. Build (or load a cached) local FAISS vector store using free,
   locally-run HuggingFace embeddings (no API key needed for embeddings).
3. Wire up a history-aware conversational retrieval chain using LangChain +
   Groq, so the agent remembers previous turns and cites its sources.
"""

import os
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

# Formats each retrieved chunk WITH its real section/line metadata before it
# reaches the LLM. Without this, create_stuff_documents_chain only shows the
# model raw chunk text with no metadata, and the model ends up inventing
# section names and renumbering lines from scratch instead of citing the
# real source.
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[Section: {section} | Line: {line}]\n{page_content}"
)

DATA_PATH = Path(__file__).parent / "data" / "gigacorp_faq.txt"
INDEX_PATH = Path(__file__).parent / "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Groq model recommended as of mid-2026 after llama-3.3-70b-versatile's
# deprecation. Swap to "llama-3.1-8b-instant" for a lighter/faster free-tier
# option if you hit rate limits.
GROQ_MODEL_NAME = "openai/gpt-oss-120b"


def load_faq_as_documents(path: Path = DATA_PATH) -> list[Document]:
    """
    Parses the FAQ file into one Document per "Line: <text>" entry, storing
    the section heading and line number as metadata. This is what lets the
    agent cite an exact source (e.g. "Section 2, Line 15") instead of a
    vague file name.
    """
    text = path.read_text(encoding="utf-8")

    documents = []
    current_section = "General"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        section_match = re.match(r"^SECTION\s+\d+:\s*(.+)$", line, re.IGNORECASE)
        if section_match:
            current_section = section_match.group(1).strip()
            continue

        line_match = re.match(r"^Line\s+(\d+):\s*(.+)$", line, re.IGNORECASE)
        if line_match:
            line_number, content = line_match.groups()
            documents.append(
                Document(
                    page_content=content.strip(),
                    metadata={
                        "source": path.name,
                        "section": current_section,
                        "line": int(line_number),
                    },
                )
            )

    return documents


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_or_load_vectorstore() -> FAISS:
    """
    Builds the FAISS index from the FAQ on first run and caches it to disk
    under faiss_index/. Subsequent runs (and Streamlit reruns) load the
    cached index instantly instead of re-embedding every chunk.
    """
    embeddings = get_embeddings()

    if INDEX_PATH.exists() and any(INDEX_PATH.iterdir()):
        return FAISS.load_local(
            str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
        )

    documents = load_faq_as_documents()
    vectorstore = FAISS.from_documents(documents, embeddings)
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_PATH))
    return vectorstore


CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question, which might "
    "reference context in the chat history, rewrite it as a standalone "
    "question that can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed, "
    "otherwise return it unchanged."
)

QA_SYSTEM_PROMPT = (
    "You are GigaCorp's customer support assistant. Answer the user's "
    "question ONLY using the retrieved context below, which comes from "
    "GigaCorp's official FAQ document. Each retrieved chunk is tagged with "
    "its real '[Section: ... | Line: ...]' metadata \u2014 you MUST copy that "
    "exact section name and line number verbatim in your citation. NEVER "
    "invent, rename, shorten, or renumber a section or line \u2014 only use the "
    "exact values shown in the [Section: ... | Line: ...] tags.\n\n"
    "Rules:\n"
    "1. Every factual claim must be explicitly cited using the exact tag "
    "values, e.g. '(Source: SHIPPING POLICY, Line 4)'.\n"
    "2. If the answer isn't in the retrieved context, say you don't have "
    "that information and suggest contacting support@gigacorp-example.com "
    "rather than guessing.\n"
    "3. Be concise, warm, and professional \u2014 like a real support agent.\n"
    "4. Use the chat history to resolve follow-up questions "
    "(e.g. 'how much does it cost?' after discussing shipping to India).\n\n"
    "Retrieved context:\n{context}"
)


def build_conversational_rag_chain(groq_api_key: str, vectorstore: FAISS):
    """
    Builds a history-aware retrieval chain:
      1. contextualize_q_chain rewrites follow-up questions into
         standalone questions using chat history (this is what gives the
         agent conversational memory over retrieval).
      2. The rewritten question retrieves relevant FAQ chunks from FAISS.
      3. The QA chain answers using only that retrieved context, citing
         section + line number.
    """
    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=groq_api_key,
        temperature=0.2,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(
        llm, qa_prompt, document_prompt=DOCUMENT_PROMPT
    )

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def format_sources(retrieved_docs) -> str:
    """Turns retrieved Document metadata into a clean citation list for the UI."""
    seen = set()
    lines = []
    for doc in retrieved_docs:
        key = (doc.metadata.get("section"), doc.metadata.get("line"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- **{doc.metadata.get('source', 'gigacorp_faq.txt')}** \u2014 "
            f"{doc.metadata.get('section', 'General')}, Line {doc.metadata.get('line', '?')}: "
            f"\"{doc.page_content}\""
        )
    return "\n".join(lines) if lines else "No specific source retrieved."