import asyncio
import concurrent.futures
import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


# ==============================================================================
# 1. GLOBAL CONFIGURATION & EMBEDDING INITIALIZATION
# ==============================================================================
print("[*] Initializing Enterprise Embedding Engine (all-MiniLM-L6-v2)...")
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

BLOCKED_DOMAINS = (
    "chat.openai.com",
    "chatgpt.com",
    "accounts.google.com",
    "login.",
    "signin.",
    "signup.",
    "pinterest.com",
    "tiktok.com",
    "instagram.com",
    "whatsapp.com",
    "quora.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "reddit.com/login",
)

BLOCKED_TITLE_PATTERNS = (
    "us (2019 film)",
    "us group",
    "sign in",
    "log in",
    "page not found",
)

STOP_WORDS = {
    "how", "are", "is", "the", "and", "or", "in", "on", "of", "to", "for",
    "with", "a", "an", "what", "why", "when", "which", "who", "whom", "whose",
    "break", "down", "explain", "analyze", "compare", "their", "them", "they",
    "impacting", "current", "state", "tell", "me", "about", "give", "list",
    "please", "can", "you", "exact", "today", "latest", "top", "from", "into",
    "us",
}

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ==============================================================================
# 2. QUERY INTELLIGENCE, ENTITY EXTRACTION & URL HYGIENE
# ==============================================================================
def _is_clean_url(url: str) -> bool:
    """Filters out malformed, non-HTTP, or low-signal social/login URLs."""
    if not url or not url.startswith(("http://", "https://")):
        return False
    lower_url = url.lower()
    if any(bad in lower_url for bad in BLOCKED_DOMAINS):
        return False
    if lower_url.endswith((".pdf", ".zip", ".exe", ".mp4", ".jpg", ".png", ".webp")):
        return False
    return True


def _is_clean_title(title: str) -> bool:
    """Rejects ambiguous/off-topic titles such as unrelated films or login pages."""
    t_lower = (title or "").strip().lower()
    if not t_lower:
        return False
    return not any(pat in t_lower for pat in BLOCKED_TITLE_PATTERNS)


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _extract_search_variants(query: str) -> List[str]:
    """
    Splits hyphenated compounds (e.g., US-China -> US China) and prioritizes
    specific technical terms and proper nouns for high-signal search results.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9\s\.\$\%]", " ", query)
    cleaned = _normalize_whitespace(cleaned)
    words = cleaned.split()

    entity_tokens: List[str] = []
    topic_tokens: List[str] = []

    for idx, w in enumerate(words):
        wl = w.lower()
        if wl in STOP_WORDS or len(w) <= 1:
            continue
        is_entity = (
            w.isupper()
            or any(ch.isdigit() for ch in w)
            or (w[0].isupper() and idx > 0)
        )
        if is_entity:
            if w not in entity_tokens:
                entity_tokens.append(w)
        else:
            if w not in topic_tokens:
                topic_tokens.append(w)

    ranked_keywords = entity_tokens + [t for t in topic_tokens if t not in entity_tokens]

    variants: List[str] = []
    if ranked_keywords:
        variants.append(" ".join(ranked_keywords[:6]))
        if len(ranked_keywords) > 3:
            variants.append(" ".join(ranked_keywords[:4]))
        if entity_tokens and topic_tokens:
            combo = entity_tokens[:3] + topic_tokens[:2]
            variants.append(" ".join(combo))

    if not variants and cleaned:
        variants.append(" ".join(words[:6]))

    seen = set()
    unique_variants: List[str] = []
    for v in variants:
        norm = _normalize_whitespace(v)
        key = norm.lower()
        if key and key not in seen:
            seen.add(key)
            unique_variants.append(norm)
    return unique_variants


def _is_time_sensitive_query(query: str) -> bool:
    """Detects whether the prompt requires live news, prices, or current events."""
    triggers = (
        "today", "latest", "current", "news", "price", "rate", "exchange",
        "2026", "2025", "stock", "market", "policy", "update", "recent", "live"
    )
    q_lower = query.lower()
    return any(t in q_lower for t in triggers)


def _relevance_score(item: Dict, keywords: List[str]) -> int:
    """Scores a search candidate by how many query keywords appear in title + snippet."""
    haystack = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
    return sum(1 for kw in keywords if kw.lower() in haystack)


# ==============================================================================
# 3. MULTI-TIER LIVE WEB SEARCH ENGINE (GOOGLE NEWS RSS + DDGS + WIKIPEDIA)
# ==============================================================================
def _search_google_news_rss(keyword_query: str, max_results: int = 5) -> List[Dict]:
    """Tier 1: Fetches real-time headlines and domain articles via Google News RSS."""
    collected: List[Dict] = []
    if not keyword_query:
        return collected

    try:
        encoded_q = urllib.parse.quote_plus(keyword_query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
        response = httpx.get(rss_url, headers=BROWSER_HEADERS, timeout=6.0, follow_redirects=True)
        if response.status_code != 200:
            return collected

        root = ET.fromstring(response.text)
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")
            source_el = item.find("source")

            title = _normalize_whitespace(title_el.text if title_el is not None else "")
            url = (link_el.text or "").strip() if link_el is not None else ""
            pub_date = _normalize_whitespace(pub_el.text if pub_el is not None else "")
            publisher = _normalize_whitespace(source_el.text if source_el is not None else "News")

            if not url or not _is_clean_title(title):
                continue

            snippet = f"{title} — Reported by {publisher} ({pub_date}).".strip()
            collected.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "engine": "Google News RSS",
            })
            if len(collected) >= max_results:
                break
    except Exception as exc:
        print(f"[!] Google News RSS warning: {exc}")

    return collected


def _search_ddgs_tier(variants: List[str], max_results: int, use_news: bool = False) -> List[Dict]:
    """Tier 2 & 3: Queries DuckDuckGo Web & News endpoints across keyword variants."""
    collected: List[Dict] = []
    seen_urls = set()

    for q in variants:
        try:
            with DDGS() as ddgs:
                raw_iter = (
                    ddgs.news(q, max_results=max_results + 5)
                    if use_news
                    else ddgs.text(q, max_results=max_results + 7)
                )
                for item in raw_iter or []:
                    url = (item.get("url") or item.get("href") or "").strip()
                    title = _normalize_whitespace(item.get("title") or "Verified Web Source")
                    if not _is_clean_url(url) or not _is_clean_title(title) or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    collected.append({
                        "title": title,
                        "url": url,
                        "snippet": _normalize_whitespace(item.get("body") or item.get("excerpt") or ""),
                        "engine": "DuckDuckGo News" if use_news else "DuckDuckGo Web",
                    })
                    if len(collected) >= max_results:
                        return collected
        except Exception as exc:
            mode = "news" if use_news else "text"
            print(f"[!] DDGS ({mode}) warning for '{q}': {exc}")

        if len(collected) >= max_results:
            break

    return collected


def _search_wikipedia_api(keyword_query: str, max_results: int = 4) -> List[Dict]:
    """Tier 4: Guaranteed encyclopedia fallback via MediaWiki Search API."""
    collected: List[Dict] = []
    if not keyword_query:
        return collected

    try:
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": keyword_query,
            "format": "json",
            "srlimit": max_results + 2,
        }
        headers = {"User-Agent": "MiniGeminiResearchAgent/2.2 (ResearchPipeline)"}
        response = httpx.get(api_url, params=params, headers=headers, timeout=6.0)
        if response.status_code == 200:
            hits = response.json().get("query", {}).get("search", [])
            for hit in hits:
                raw_title = _normalize_whitespace(hit.get("title", ""))
                display_title = f"{raw_title} — Wikipedia"
                if not raw_title or not _is_clean_title(display_title):
                    continue
                raw_snip = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
                snippet = _normalize_whitespace(raw_snip)
                wiki_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(raw_title.replace(' ', '_'))}"
                collected.append({
                    "title": display_title,
                    "url": wiki_url,
                    "snippet": snippet,
                    "engine": "Wikipedia Live API",
                })
                if len(collected) >= max_results:
                    break
    except Exception as exc:
        print(f"[!] Wikipedia API warning: {exc}")

    return collected


def multi_tier_web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Orchestrates 4 search tiers prioritizing real-time domain articles first:
    1. Google News RSS Live Feed (primary & secondary entity variants)
    2. DuckDuckGo News (when time-sensitive)
    3. DuckDuckGo Web Search
    4. Wikipedia Encyclopedia API
    """
    variants = _extract_search_variants(query)
    primary_kw = variants[0] if variants else query
    short_kw = variants[1] if len(variants) > 1 else primary_kw
    keywords = [w for w in primary_kw.split() if len(w) > 1]

    merged: List[Dict] = []
    seen_urls = set()

    def _extend_unique(items: List[Dict]):
        for it in items:
            u = it.get("url", "")
            if u and u not in seen_urls and len(merged) < max_results * 2:
                seen_urls.add(u)
                merged.append(it)

    # 1. Google News RSS first for crisp, real-time domain articles
    _extend_unique(_search_google_news_rss(primary_kw, max_results=max_results))
    if len(merged) < max_results and short_kw != primary_kw:
        _extend_unique(_search_google_news_rss(short_kw, max_results=max_results))

    # 2. DuckDuckGo News & Web
    if _is_time_sensitive_query(query) and len(merged) < max_results:
        _extend_unique(_search_ddgs_tier(variants, max_results=max_results, use_news=True))

    if len(merged) < max_results:
        _extend_unique(_search_ddgs_tier(variants, max_results=max_results, use_news=False))

    # 3. Wikipedia API fallback
    if len(merged) < max_results:
        _extend_unique(_search_wikipedia_api(short_kw, max_results=max_results))

    # Rank highest keyword-matching sources first while preserving tier order on ties
    indexed = list(enumerate(merged))
    indexed.sort(key=lambda pair: (-_relevance_score(pair[1], keywords), pair[0]))
    return [item for _, item in indexed[:max_results]]


# ==============================================================================
# 4. ASYNCHRONOUS PARALLEL CONTENT SCRAPER & CLEANER
# ==============================================================================
def _extract_clean_html_text(html: str) -> str:
    """Strips boilerplate tags and extracts structured paragraphs and headings."""
    soup = BeautifulSoup(html, "html.parser")
    for noise in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form", "iframe", "svg"]):
        noise.decompose()

    container = soup.find("article") or soup.find("main") or soup
    blocks = [
        _normalize_whitespace(el.get_text(" ", strip=True))
        for el in container.find_all(["h1", "h2", "h3", "p", "li", "td"])
    ]

    seen_lines = set()
    clean_lines = []
    for line in blocks:
        if len(line) < 40 or line in seen_lines:
            continue
        seen_lines.add(line)
        clean_lines.append(line)

    return "\n".join(clean_lines)


async def fetch_single_page(client: httpx.AsyncClient, item: Dict) -> Optional[Document]:
    """Asynchronously fetches and cleans a single web page with snippet fallback."""
    url = item["url"]
    title = item.get("title", "Verified Source")
    snippet = item.get("snippet", "")
    engine = item.get("engine", "Web")

    if "news.google.com/rss/" in url and snippet:
        return Document(
            page_content=f"{title}\n{snippet}",
            metadata={"source": url, "title": title, "engine": engine},
        )

    try:
        response = await client.get(url, timeout=5.5, follow_redirects=True)
        if response.status_code != 200:
            if snippet:
                return Document(
                    page_content=f"{title}\n{snippet}",
                    metadata={"source": url, "title": title, "engine": engine},
                )
            return None

        extracted = _extract_clean_html_text(response.text)
        if len(extracted) < 160 and snippet:
            extracted = f"{title}\n{snippet}\n{extracted}".strip()

        if not extracted:
            return None

        return Document(
            page_content=extracted[:9000],
            metadata={"source": url, "title": title, "engine": engine},
        )
    except Exception:
        if snippet:
            return Document(
                page_content=f"{title}\n{snippet}",
                metadata={"source": url, "title": title, "engine": engine},
            )
        return None


async def scrape_urls_parallel(search_results: List[Dict]) -> List[Document]:
    """Scrapes all candidate URLs concurrently using an async HTTPX connection pool."""
    limits = httpx.Limits(max_keepalive_connections=8, max_connections=12)
    async with httpx.AsyncClient(headers=BROWSER_HEADERS, limits=limits, verify=False) as client:
        tasks = [fetch_single_page(client, item) for item in search_results]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    return [doc for doc in results if doc is not None and doc.page_content.strip()]


def _run_async_safely(coro):
    """Executes async coroutines cleanly inside Streamlit's active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


# ==============================================================================
# 5. HYBRID VECTOR RAG BUILDER & RERANKER
# ==============================================================================
class HybridRerankedRetriever:
    """
    Wraps FAISS semantic similarity search with lexical keyword-overlap scoring
    so retrieved chunks match both conceptual meaning and exact entities/numbers.
    """

    def __init__(self, vectorstore: FAISS, query_keywords: List[str], k: int = 5):
        self.vectorstore = vectorstore
        self.query_keywords = [w.lower() for w in query_keywords if len(w) > 1]
        self.k = k

    def invoke(self, query: str) -> List[Document]:
        candidates = self.vectorstore.similarity_search(query, k=max(self.k * 2, 8))
        if not self.query_keywords:
            return candidates[:self.k]

        scored: List[Tuple[int, Document]] = []
        for doc in candidates:
            text_lower = doc.page_content.lower()
            overlap = sum(1 for kw in self.query_keywords if kw in text_lower)
            scored.append((overlap, doc))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in scored[:self.k]]


def build_dynamic_rag(query: str, max_results: int = 5) -> Tuple[Optional[HybridRerankedRetriever], List[Dict]]:
    """
    End-to-End Enterprise Web RAG Pipeline:
    1. Multi-Tier Live Web Search (Google News RSS + DDGS + Wikipedia)
    2. Async Parallel Web Scraping & Boilerplate Removal
    3. Recursive Semantic Chunking & In-Memory FAISS Vector Indexing
    4. Hybrid Semantic + Keyword Reranking Retriever
    """
    search_results = multi_tier_web_search(query, max_results=max_results)
    if not search_results:
        return None, []

    docs = _run_async_safely(scrape_urls_parallel(search_results))
    if not docs:
        return None, search_results

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_documents(docs)
    chunks = [c for c in raw_chunks if c.page_content and len(c.page_content.strip()) > 30]

    if not chunks:
        return None, search_results

    vectorstore = FAISS.from_documents(chunks, EMBEDDINGS)
    variants = _extract_search_variants(query)
    keywords = variants[0].split() if variants else []
    retriever = HybridRerankedRetriever(vectorstore=vectorstore, query_keywords=keywords, k=5)

    return retriever, search_results