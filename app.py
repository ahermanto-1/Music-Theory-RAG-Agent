"""
app.py — The Harmonic Curator
Streamlit chat interface for the Music Theory RAG Agent.

API keys are read from .env locally.
For deployment (Streamlit Cloud), add keys to .streamlit/secrets.toml instead.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# ── API key handling ──────────────────────────────────────────────────────────
# Supports both local .env and Streamlit Cloud secrets
load_dotenv()

def _load_secrets():
    """Inject Streamlit secrets into env vars if running on Streamlit Cloud."""
    try:
        for key in ("PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "GEMINI_API_KEY"):
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = st.secrets[key]
    except Exception:
        pass  # secrets.toml not present — relying on .env

_load_secrets()

from agent import run  # noqa: E402 — must import after env is populated

# ── Helpers ───────────────────────────────────────────────────────────────────

_SOURCE_LABELS = {
    "retrieve":   ("Knowledge base", "source-badge-retrieve"),
    "web_search": ("Web search",     "source-badge-web"),
    "direct":     ("Direct",         "source-badge-direct"),
}

def _source_badge(source: str) -> str:
    label, css_class = _SOURCE_LABELS.get(source, ("Unknown", "source-badge-direct"))
    return (
        f"<span class='source-badge {css_class}'>{label}</span>"
    )


def _render_assistant_meta(msg: dict) -> None:
    """Render source badge, rewrite notice, and sources expander for an assistant message."""
    source = msg.get("source", "retrieve")
    st.markdown(_source_badge(source), unsafe_allow_html=True)

    if msg.get("rewritten"):
        st.info(f"Query rewritten to: *{msg['rewritten']}*")

    steps = msg.get("steps", [])
    if steps:
        with st.expander("View agent reasoning"):
            for i, step in enumerate(steps, 1):
                st.markdown(
                    f"<p style='font-size:0.82rem;color:#2a3439;line-height:1.6;"
                    f"margin:0.1rem 0;'><span style='color:#a9b4b9;font-weight:500;"
                    f"margin-right:0.4rem;'>{i}.</span>{step}</p>",
                    unsafe_allow_html=True,
                )

    chunks = msg.get("chunks", [])
    if chunks:
        is_web = source == "web_search"
        expander_label = f"View {len(chunks)} web results" if is_web else f"View {len(chunks)} sources used"
        with st.expander(expander_label):
            for i, chunk in enumerate(chunks, 1):
                if is_web:
                    url = chunk["source"]
                    st.markdown(
                        f"<p style='font-size:0.75rem;font-weight:600;"
                        f"color:#455f88;margin-bottom:0.25rem;'>"
                        f"[{i}] <a href='{url}' target='_blank' "
                        f"style='color:#455f88;text-decoration:none;'>{url}</a> "
                        f"<span style='color:#a9b4b9;font-weight:400;'>"
                        f"score {chunk['score']}</span></p>"
                        f"<p style='font-size:0.82rem;color:#2a3439;"
                        f"line-height:1.6;margin-bottom:1rem;'>"
                        f"{chunk['text'][:400]}{'…' if len(chunk['text']) > 400 else ''}</p>",
                        unsafe_allow_html=True,
                    )
                else:
                    label = chunk["source"]
                    if chunk["chapter"]:
                        label += f"  ·  {chunk['chapter']}"
                    st.markdown(
                        f"<p style='font-size:0.75rem;font-weight:600;"
                        f"color:#455f88;margin-bottom:0.25rem;'>"
                        f"[{i}] {label} "
                        f"<span style='color:#a9b4b9;font-weight:400;'>"
                        f"score {chunk['score']}</span></p>"
                        f"<p style='font-size:0.82rem;color:#2a3439;"
                        f"line-height:1.6;margin-bottom:1rem;'>"
                        f"{chunk['text'][:400]}{'…' if len(chunk['text']) > 400 else ''}</p>",
                        unsafe_allow_html=True,
                    )

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Harmonic Curator",
    page_icon="♩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700&family=Inter:wght@300;400;500&display=swap');

/* ── Base ── */
*, *::before, *::after {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Manrope', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"], [data-testid="stHeader"] { display: none; }
[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; }

/* ── App background ── */
.stApp {
    background-color: #f7f9fb;
    color: #2a3439;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #f0f4f7 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.5rem;
}
[data-testid="stSidebarNav"] { display: none; }

/* ── Main content area ── */
.main .block-container {
    padding: 2rem 3rem 6rem 3rem;
    max-width: 860px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.25rem 0;
}

/* Bot bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
[data-testid="stChatMessageContent"] {
    background-color: #f0f4f7;
    color: #2a3439;
    border-radius: 0.75rem 0.75rem 0.75rem 0.125rem;
    padding: 1rem 1.25rem;
    box-shadow: 0 8px 32px 0 rgba(42, 52, 57, 0.06);
    font-size: 0.95rem;
    line-height: 1.7;
    border: none;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
[data-testid="stChatMessageContent"] {
    background-color: #d6e3ff;
    color: #2a3439;
    border-radius: 0.75rem 0.75rem 0.125rem 0.75rem;
    padding: 1rem 1.25rem;
    box-shadow: 0 8px 32px 0 rgba(42, 52, 57, 0.06);
    font-size: 0.95rem;
    line-height: 1.7;
    border: none;
}

/* Avatar icons */
[data-testid="chatAvatarIcon-assistant"],
[data-testid="chatAvatarIcon-user"] {
    background: transparent !important;
    border: none !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background-color: #d9e4ea !important;
    border: none !important;
    border-radius: 0.75rem !important;
    box-shadow: 0 8px 32px 0 rgba(42, 52, 57, 0.06);
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    color: #2a3439 !important;
    border: none !important;
    outline: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #a9b4b9 !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #455f88, #39537c) !important;
    border: none !important;
    border-radius: 0.5rem !important;
}
[data-testid="stChatInput"] button:hover {
    opacity: 0.88;
}

/* ── Expander (sources) ── */
[data-testid="stExpander"] {
    background-color: #ffffff;
    border: none !important;
    border-radius: 0.5rem;
    box-shadow: 0 8px 32px 0 rgba(42, 52, 57, 0.06);
    margin-top: 0.5rem;
}
[data-testid="stExpander"] summary {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #455f88;
    font-weight: 500;
}

/* ── Info / rewrite badge ── */
[data-testid="stAlert"] {
    background-color: #ffffff !important;
    border: none !important;
    border-radius: 0.5rem !important;
    box-shadow: 0 8px 32px 0 rgba(42, 52, 57, 0.06);
    font-size: 0.82rem;
    color: #455f88 !important;
}

/* ── Source badge ── */
.source-badge {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    margin-top: 0.6rem;
}
.source-badge-retrieve  { background: #e8eef5; color: #455f88; }
.source-badge-web       { background: #e6f2ec; color: #3d7a52; }
.source-badge-direct    { background: #f0f0f0; color: #7a8a90; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cfdce3; border-radius: 2px; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-top-color: #455f88 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='font-family:Manrope;font-weight:700;color:#2a3439;"
        "font-size:1.4rem;margin-bottom:0.25rem;'>♩ The Harmonic Curator</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.82rem;color:#a9b4b9;margin-bottom:2rem;'>"
        "Music theory, explored.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='font-family:Manrope;font-size:0.75rem;font-weight:600;"
        "color:#455f88;text-transform:uppercase;letter-spacing:0.08em;"
        "margin-bottom:0.75rem;'>How it works</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<ol style='font-size:0.82rem;color:#2a3439;line-height:1.8;"
        "padding-left:1.1rem;'>"
        "<li>A router classifies your query so that the agent knows where to grab information from - either the knowledge base, from the web, or directly from the model's training data.</li>"
        "<li>If your query requires a knowledge base lookup, a relevance grader checks if the results of the lookup are useful in answering your question. If not, your original query is re-written and is searched against the knowledge base again.</li>"
        "<li>If your query requires a web search, Tavily is used to grab relevant content from the web.</li>"
        "<li>The agent generates an answer, using context from the retrieval process if it was required.</li>"
        "</ol>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border:none;border-top:1px solid #e8eff3;margin:1.5rem 0;'>", unsafe_allow_html=True)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<p style='font-size:0.75rem;color:#a9b4b9;margin-top:auto;padding-top:2rem;'>"
        "Powered by Gemma 3 27B · Pinecone · all-MiniLM-L6-v2</p>",
        unsafe_allow_html=True,
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:Manrope;font-weight:700;font-size:2rem;"
    "color:#2a3439;margin-bottom:0.25rem;'>Ask a music theory question</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-size:0.9rem;color:#a9b4b9;margin-bottom:2rem;'>"
    "Answers are grounded in your knowledge base of music theory texts.</p>",
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # each entry: {role, content, chunks, rewritten, source}

# ── Render history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            _render_assistant_meta(msg)

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("e.g. How do I build a minor 7th chord on the guitar?"):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = run(prompt, history=st.session_state.messages[:-1])
                answer = result.answer
                chunks = result.chunks
                rewritten = result.rewritten
                source = result.source
                steps = result.steps
            except Exception as e:
                answer = f"Something went wrong: {e}"
                chunks, rewritten, source, steps = [], None, "direct", []

        st.write(answer)
        _render_assistant_meta({"source": source, "rewritten": rewritten, "chunks": chunks, "steps": steps})

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks,
        "rewritten": rewritten,
        "source": source,
        "steps": steps,
    })
