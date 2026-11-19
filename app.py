import os
import time
import streamlit as st
from dotenv import load_dotenv
from scraper_rag import build_dynamic_rag
from llm_engine import MiniGeminiEngine

# Load API keys from backend .env silently
load_dotenv()

CONTINUE_TRIGGERS = {
    "yes", "y", "haan", "ha", "han", "aur batao", "or batao",
    "continue", "aage", "aage batao", "next", "more"
}

# ------------------------------------------------------------------
# 1. PAGE CONFIG & PREMIUM DARK UI STYLING
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Mini-Gemini | Live Web Deep Research Agent",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .badge-cloud {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.35);
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-local {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.35);
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. CLEAN SIDEBAR CONTROLS (NO EXPOSED API KEY BOX)
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Research Controls")

    max_sources = st.slider(
        "Web Sources per Query",
        min_value=2,
        max_value=6,
        value=5,
        help="Number of live web sources to scrape and index for retrieval"
    )

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_continuation = None
        st.rerun()

    st.caption(
        "Architecture: Live Web Search → Async Parallel Scraper → "
        "FAISS Vector RAG → High-Speed Cloud / Local Fallback"
    )

# ------------------------------------------------------------------
# 3. CACHED ENGINE INITIALIZATION
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI Research Engine...")
def get_engine(api_key: str):
    return MiniGeminiEngine(use_lora=False, groq_api_key=api_key)

active_groq_key = os.getenv("GROQ_API_KEY", "").strip()
engine = get_engine(active_groq_key)

# ------------------------------------------------------------------
# 4. HEADER & SESSION STATE
# ------------------------------------------------------------------
st.markdown('<div class="hero-title">✨ Mini-Gemini Deep Research Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Real-Time Web Intelligence • Vector RAG Synthesis • Executive-Grade Answers</div>',
    unsafe_allow_html=True
)

if engine.use_cloud:
    st.markdown('<span class="badge-cloud">⚡ Cloud Turbo Active</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="badge-local">💻 Local CPU Mode</span>', unsafe_allow_html=True)

st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_continuation" not in st.session_state:
    st.session_state.pending_continuation = None

def _looks_cut_off(text: str) -> bool:
    stripped = (text or "").strip()
    if len(stripped) < 400:
        return False
    # Check unclosed parentheses, trailing hyphen/colon, or abrupt mid-word/mid-phrase ending
    open_parens = stripped.count("(") - stripped.count(")")
    ends_abruptly = (
        open_parens > 0
        or stripped[-1] not in ".!?\n*`"
        or stripped.endswith(("(", ":", "-", "–", "—", ","))
    )
    return ends_abruptly

# ------------------------------------------------------------------
# 5. RENDER PREVIOUS CHAT HISTORY
# ------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"🔍 Verified Web Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s['title']}]({s['url']})")

# ------------------------------------------------------------------
# 6. USER QUERY PIPELINE (WITH 'YES' CONTINUATION OR NEW QUESTION)
# ------------------------------------------------------------------
user_query = st.chat_input("Ask any deep research question (or type 'yes' to continue the previous answer)...")

if user_query:
    cleaned_input = user_query.strip().lower()
    pending = st.session_state.pending_continuation
    is_continuing = bool(pending and cleaned_input in CONTINUE_TRIGGERS)

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        start_time = time.time()

        if is_continuing:
            target_question = pending["question"]
            retrieved_docs = pending.get("docs", [])
            search_results = pending.get("sources", [])
            prior_text = pending.get("accumulated_text", "")
            st.info(f"↪️ Continuing answer for: **{target_question}**")
        else:
            st.session_state.pending_continuation = None
            target_question = user_query
            prior_text = ""

            with st.status("🌐 Searching & scraping live web intelligence...", expanded=False) as status:
                retriever, search_results = build_dynamic_rag(target_question, max_results=max_sources)
                if retriever:
                    retrieved_docs = retriever.invoke(target_question)
                    status.update(
                        label=f"✅ Indexed {len(search_results)} live sources in {time.time() - start_time:.1f}s",
                        state="complete"
                    )
                else:
                    retrieved_docs = []
                    status.update(label="⚠️ Synthesizing from core model knowledge...", state="complete")

        history_str = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in st.session_state.messages[-4:-1]
        )

        response_stream = engine.stream_answer(
            question=target_question,
            retrieved_docs=retrieved_docs,
            chat_history=history_str,
            continue_from=prior_text
        )
        streamed_chunk = st.write_stream(response_stream)

        full_response = streamed_chunk or ""
        offer_prompt = ""

        if _looks_cut_off(full_response):
            offer_prompt = (
                "\n\n---\n"
                "💡 **Would you like me to continue the rest of this response and remaining sections?**\n"
                "- Type **`yes`** to continue from where it left off.\n"
                "- Or ask a **new question** anytime."
            )
            st.markdown(offer_prompt)
            st.session_state.pending_continuation = {
                "question": target_question,
                "docs": retrieved_docs,
                "sources": search_results,
                "accumulated_text": (prior_text + "\n" + full_response).strip(),
            }
        else:
            st.session_state.pending_continuation = None

        final_assistant_content = full_response + offer_prompt
        elapsed = time.time() - start_time
        st.caption(f"⏱️ Researched & synthesized in {elapsed:.2f}s • {len(search_results)} sources verified")

        if search_results:
            with st.expander(f"🔍 Verified Web Sources ({len(search_results)})"):
                for s in search_results:
                    st.markdown(f"- [{s['title']}]({s['url']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_assistant_content,
        "sources": search_results
    })