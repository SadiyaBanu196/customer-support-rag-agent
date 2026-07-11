"""
app.py
------
Streamlit UI for the GigaCorp Customer Support RAG Agent.

Run locally:
    streamlit run app.py

Requires a free Groq API key set as an environment variable GROQ_API_KEY,
or entered in the sidebar when running the hosted demo.
"""

import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from rag_engine import build_or_load_vectorstore, build_conversational_rag_chain, format_sources

st.set_page_config(page_title="GigaCorp Support Assistant", page_icon="\U0001F4AC", layout="centered")

st.title("\U0001F4AC GigaCorp Support Assistant")
st.caption(
    "Ask me about shipping, returns, business hours, or service tiers. "
    "I answer using GigaCorp's FAQ and cite my sources."
)

# ---------------------------------------------------------------------------
# Sidebar: API key input (kept out of the repo; entered at runtime or via
# Streamlit Cloud "Secrets")
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    default_key = os.environ.get("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Get a free key at https://console.groq.com/keys. "
        "Never commit this key to GitHub — use Streamlit Secrets instead.",
    )
    st.divider()
    st.markdown(
        "**Knowledge base:** `data/gigacorp_faq.txt`\n\n"
        "**Vector store:** FAISS (local, persisted in `faiss_index/`)\n\n"
        "**Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (free, local)\n\n"
        "**LLM:** Groq \u2014 openai/gpt-oss-120b"
    )
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.display_messages = []
        st.rerun()

if not groq_api_key:
    st.info("Enter your free Groq API key in the sidebar to start chatting. Get one at https://console.groq.com/keys")
    st.stop()

# ---------------------------------------------------------------------------
# Cache the vector store across reruns/users (it's read-only shared data)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base into FAISS...")
def get_vectorstore():
    return build_or_load_vectorstore()


vectorstore = get_vectorstore()
rag_chain = build_conversational_rag_chain(groq_api_key, vectorstore)

# ---------------------------------------------------------------------------
# Session state: chat_history is the LangChain-format history fed to the
# chain for conversational memory; display_messages is what's rendered.
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[HumanMessage | AIMessage]
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # list[dict(role, content, sources)]

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("\U0001F4CE Sources"):
                st.markdown(msg["sources"])

user_input = st.chat_input("e.g. Do you ship to India?")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.display_messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag_chain.invoke(
                    {
                        "input": user_input,
                        "chat_history": st.session_state.chat_history,
                    }
                )
                answer = result["answer"]
                sources_md = format_sources(result.get("context", []))
            except Exception as e:
                answer = (
                    "Sorry, I ran into an error reaching the language model. "
                    f"Details: {e}"
                )
                sources_md = ""

        st.markdown(answer)
        if sources_md:
            with st.expander("\U0001F4CE Sources"):
                st.markdown(sources_md)

    st.session_state.display_messages.append(
        {"role": "assistant", "content": answer, "sources": sources_md}
    )
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))