# Mini-Gemini Deep Research Agent

  An autonomous web research and synthesis application engineered with [Streamlit](https://streamlit.io), FAISS vector retrieval, asynchronous parallel scraping, and Groq LLM streaming with seamless multi-part continuation.

## Overview

Mini-Gemini Deep Research Agent transforms complex, open-ended questions into structured, source-grounded reports in real time. Instead of relying solely on static model memory, the system retrieves live sources across multiple search tiers, cleans and indexes page content into an in-memory vector store, reranks chunks by semantic similarity and keyword overlap, and streams coherent long-form answers with verifiable citations.

## Use Cases

- **Client & Market Research**: Generate structured technical briefs, competitor comparisons, and project scoping reports in minutes.
- **Technical & Academic Synthesis**: Break down system architectures, engineering workflows, and emerging frameworks with verified references.
- **Product & Startup Roadmaps**: Produce phased execution plans covering MVP architecture, security hardening, deployment pipelines, and scaling strategies.
- **Rapid Fact-Grounded Exploration**: Consolidate insights from multiple live web sources into a single readable synthesis without manual tab switching.

## Key Advantages

- **Live Multi-Tier Web Grounding**: Combines Google News RSS, DuckDuckGo search endpoints, and Wikipedia APIs across ranked entity and topic variants to surface current, high-signal sources.
- **Hybrid Semantic + Lexical Retrieval**: Pairs dense FAISS vector similarity (`all-MiniLM-L6-v2`) with exact keyword-overlap reranking so both conceptual meaning and specific technical terms are preserved.
- **Seamless Long-Form Continuation**: Detects token-limit cutoffs automatically and allows users to continue multi-section reports without losing active context or retrieved sources.
- **High-Speed Async Scraping**: Fetches and strips boilerplate from multiple candidate pages concurrently using `httpx` connection pooling and `BeautifulSoup`.
- **Configurable Research Depth**: Adjust the number of verified web sources per query (`2` to `6`) and manage session state directly from the sidebar controls.

## System Architecture

1. **Query Intelligence**: Extracts named entities, acronyms, and topic keywords to generate targeted search variants.
2. **Multi-Tier Search & Async Scraping**: Queries live search endpoints and scrapes clean text concurrently with snippet fallbacks for restricted pages.
3. **In-Memory Vector RAG**: Splits extracted documents into overlapping chunks (`700` characters, `100` overlap), indexes them in `FAISS`, and reranks top candidates.
4. **Streaming LLM Synthesis**: Streams structured Markdown responses via `MiniGeminiEngine` (`Groq` cloud inference with optional local fallback) and attaches verified source links.

## Repository Structure

- `app.py`: Streamlit user interface, custom dark theme styling, streaming chat loop, and continuation state handler.
- `scraper_rag.py`: Query variant extraction, multi-tier live search, async HTML scraper, FAISS index builder, and hybrid reranker.
- `llm_engine.py`: Inference engine handling streaming generation, prompt formatting, and continuation logic.
- `train_qlora.py`: QLoRA fine-tuning pipeline script for local adapter experimentation.
- `requirements.txt`: Lightweight Python dependencies configured for cloud deployment.
