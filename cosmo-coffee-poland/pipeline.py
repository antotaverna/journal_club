"""
pipeline.py — arXiv → Notion pipeline (with optional Claude AI enrichment)
---------------------------------------------------------------------------
Usage:
  1. Add your arXiv IDs to papers.py
  2. Set USE_CLAUDE = True or False
  3. Set environment variables (or create a .env file)
  4. Run: python pipeline.py

Required env vars (always):
  NOTION_TOKEN        — from notion.so/my-integrations
  NOTION_DATABASE_ID  — from your Notion database URL

Required env vars (only if USE_CLAUDE = True):
  ANTHROPIC_API_KEY   — from console.anthropic.com

Install dependencies:
  pip install arxiv anthropic requests python-dotenv keybert
"""

import os
import json
import time
import logging
import requests
import arxiv
from datetime import datetime, timezone
from dataclasses import dataclass
from dotenv import load_dotenv

# Load paper IDs from external file
try:
    from load_papers import PAPER_IDS
except ImportError:
    raise ImportError(
        "Could not find papers.py — make sure it exists in the same folder as pipeline.py\n"
        "Create it with a PAPER_IDS list, e.g.:\n\n"
        "  PAPER_IDS = [\n"
        '      "2301.07041",\n'
        '      "2310.06825",\n'
        "  ]"
    )

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. TOGGLE — set to False to use KeyBERT instead of Claude
# ─────────────────────────────────────────────
USE_CLAUDE = False  # ← True = Claude AI  |  False = KeyBERT (free, local)

# ─────────────────────────────────────────────
# 2. CONFIG — reads from .env file
# ─────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY")
NOTION_TOKEN       = os.environ.get("NOTION_TOKEN")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID")

NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

# ─────────────────────────────────────────────
# 3. DATA MODEL
# ─────────────────────────────────────────────
@dataclass
class Paper:
    arxiv_id:   str
    title:      str
    authors:    list[str]
    abstract:   str
    url:        str
    published:  str         # "YYYY-MM-DD"
    categories: list[str]   # e.g. ["cs.LG", "stat.ML"]

# ─────────────────────────────────────────────
# 4. FETCH FROM ARXIV
# ─────────────────────────────────────────────
def fetch_papers(paper_ids: list[str]) -> list[Paper]:
    """Fetch paper metadata from arXiv by ID list."""
    if not paper_ids:
        raise ValueError("PAPER_IDS is empty in papers.py — add at least one arXiv ID.")

    log.info(f"Fetching {len(paper_ids)} papers from arXiv...")
    client = arxiv.Client(delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(id_list=paper_ids)

    papers = []
    for result in client.results(search):
        clean_id = result.entry_id.split("/abs/")[-1]
        papers.append(Paper(
            arxiv_id   = clean_id,
            title      = result.title.strip(),
            authors    = [a.name for a in result.authors],
            abstract   = result.summary.replace("\n", " ").strip(),
            url        = result.entry_id,
            published  = result.published.strftime("%Y-%m-%d"),
            categories = result.categories,
        ))
        log.info(f"  ✓ [{clean_id}] {result.title[:65]}...")

    log.info(f"Fetched {len(papers)}/{len(paper_ids)} papers.")
    return papers

# ─────────────────────────────────────────────
# 5. KEYWORDS — KeyBERT (USE_CLAUDE=False)
# ─────────────────────────────────────────────

# Load KeyBERT model once at startup to avoid reloading on every paper
_keybert_model = None

def get_keybert_model():
    """Lazy-load KeyBERT model (only when needed)."""
    global _keybert_model
    if _keybert_model is None:
        log.info("Loading KeyBERT model (first time only, may take a few seconds)...")
        from keybert import KeyBERT
        _keybert_model = KeyBERT("allenai/scibert_scivocab_uncased")
        log.info("KeyBERT model loaded.")
    return _keybert_model

def extract_keywords_keybert(paper: Paper) -> list[str]:
    """
    Extract keywords from title + abstract using KeyBERT.
    Combines both fields to get better coverage of the paper's topics.
    """
    model = get_keybert_model()

    # Combine title and abstract for richer keyword extraction
    text = f"{paper.title}. {paper.abstract}"

    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),  # single words and pairs (e.g. "dark matter")
        stop_words="english",
        use_mmr=True,                  # MMR reduces redundancy between keywords
        diversity=0.5,                 # balance between relevance and diversity
        top_n=8
    )

    return [kw.lower() for kw, score in keywords]

# ─────────────────────────────────────────────
# 6. ENRICHMENT — with Claude (USE_CLAUDE=True)
# ─────────────────────────────────────────────
def enrich_with_claude(paper: Paper, client) -> dict:
    """Call Claude to generate summary, keywords and category."""
    prompt = f"""You are analyzing a research paper. Return ONLY a valid JSON object with exactly these fields:
- "summary": string — 3 to 5 sentences explaining the paper in plain language, no jargon
- "keywords": array of 5 to 8 strings — key topics or techniques extracted directly from the paper content
- "category": string — pick exactly one from:
  [AI/ML, Physics, Biology, Economics, Chemistry, Mathematics, Medicine, Social Sciences, Engineering, Other]

Paper title: {paper.title}
Authors: {', '.join(paper.authors[:6])}
arXiv categories: {', '.join(paper.categories)}
Abstract: {paper.abstract[:1800]}

Rules:
- Return ONLY the JSON object. No markdown, no backticks, no explanation.
- "keywords" must be lowercase strings extracted from the paper content (not arXiv category codes).
- "category" must be exactly one of the options listed above."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # cheap + fast, great for summaries
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)

# ─────────────────────────────────────────────
# 7. ENRICHMENT — without Claude (USE_CLAUDE=False)
# ─────────────────────────────────────────────
def enrich_without_claude(paper: Paper) -> dict:
    """
    Build enrichment data from arXiv metadata + KeyBERT keyword extraction.
    Keywords come from the paper content, not arXiv category codes.
    """
    category_map = {
        "cs.AI":    "AI/ML",   "cs.LG":   "AI/ML",  "cs.CL":  "AI/ML",
        "cs.CV":    "AI/ML",   "cs.NE":   "AI/ML",  "stat.ML": "AI/ML",
        "math":     "Mathematics",
        "physics":  "Physics", "astro":   "Physics", "cond-mat": "Physics",
        "q-bio":    "Biology",
        "econ":     "Economics", "q-fin":  "Economics",
        "cs.":      "Engineering",
    }

    # Determine category from arXiv tags
    assigned = "Other"
    for cat in paper.categories:
        for prefix, label in category_map.items():
            if cat.startswith(prefix):
                assigned = label
                break
        if assigned != "Other":
            break

    # Extract real keywords from paper content using KeyBERT
    log.info(f"  Extracting keywords with KeyBERT...")
    keywords = extract_keywords_keybert(paper)
    log.info(f"  Keywords found: {keywords}")

    return {
        "summary":  "",
        "keywords": keywords, 
        "category": assigned,
    }

# ─────────────────────────────────────────────
# 8. NOTION — DEDUPLICATION
# ─────────────────────────────────────────────
def get_existing_arxiv_ids() -> set[str]:
    """Return arXiv IDs already stored in Notion (handles pagination)."""
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    existing = set()
    payload = {"page_size": 100}

    while True:
        res = requests.post(url, headers=NOTION_HEADERS, json=payload)
        res.raise_for_status()
        data = res.json()

        for page in data.get("results", []):
            prop = page["properties"].get("arXiv ID", {}).get("rich_text", [])
            if prop:
                existing.add(prop[0]["text"]["content"])

        if not data.get("has_more"):
            break
        payload["start_cursor"] = data["next_cursor"]

    return existing

# ─────────────────────────────────────────────
# 9. NOTION — INSERT
# ─────────────────────────────────────────────
def insert_paper(paper: Paper, enriched: dict, ai_enriched: bool) -> None:
    """Create a new page in the Notion database."""
    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Title": {
                "title": [{"text": {"content": paper.title}}]
            },
            "Authors": {
                "rich_text": [{"text": {"content": ", ".join(paper.authors[:8])}}]
            },
            "Abstract": {
                "rich_text": [{"text": {"content": paper.abstract}}]
            },
            "Summary": {
                "rich_text": [{"text": {"content": enriched.get("summary", "")}}]
            },
            "Keywords": {
                "multi_select": [
                    {"name": kw[:100]} for kw in enriched.get("keywords", [])
                ]
            },
            "Category": {
                "select": {"name": enriched.get("category", "Other")}
            },
            "arXiv ID": {
                "rich_text": [{"text": {"content": paper.arxiv_id}}]
            },
            "URL": {
                "url": paper.url
            },
            "Published": {
                "date": {"start": paper.published}
            },
            "Added On": {
                "date": {"start": datetime.now(timezone.utc).date().isoformat()}
            },
            "arXiv Categories": {
                "multi_select": [{"name": cat} for cat in paper.categories[:5]]
            },
            "AI Enriched": {
                "checkbox": ai_enriched  # True = Claude keywords, False = KeyBERT keywords
            },
        }
    }

    res = requests.post(
        "https://api.notion.com/v1/pages",
        headers=NOTION_HEADERS,
        json=payload
    )
    res.raise_for_status()

# ─────────────────────────────────────────────
# 10. VALIDATION
# ─────────────────────────────────────────────
def validate_config():
    """Check all required env vars are present before starting."""
    missing = []

    for var in ("NOTION_TOKEN", "NOTION_DATABASE_ID"):
        if not os.environ.get(var):
            missing.append(var)

    if USE_CLAUDE and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")

    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing)}\n"
            f"Add them to your .env file."
        )

    if not PAPER_IDS:
        raise ValueError("PAPER_IDS is empty in papers.py — add at least one arXiv ID.")

# ─────────────────────────────────────────────
# 11. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline():
    validate_config()

    mode = "Claude AI (summary + keywords)" if USE_CLAUDE else "KeyBERT keywords + arXiv metadata"
    log.info("=" * 55)
    log.info(f"   arXiv → Notion  |  {mode}")
    log.info(f"   Papers loaded from papers.py: {len(PAPER_IDS)}")
    log.info("=" * 55)

    # Step 1: fetch papers from arXiv
    papers = fetch_papers(PAPER_IDS)

    # Step 2: load existing IDs from Notion (deduplication)
    log.info("Checking existing entries in Notion...")
    existing_ids = get_existing_arxiv_ids()
    log.info(f"Found {len(existing_ids)} existing papers in Notion.")

    # Step 3: set up Claude client only if needed
    ai_client = None
    if USE_CLAUDE:
        import anthropic
        ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        log.info("Claude client initialized.")
    else:
        log.info("Using KeyBERT for keyword extraction (no API cost).")

    # Step 4: enrich + insert each paper
    stats = {"inserted": 0, "skipped": 0, "failed": 0}

    for paper in papers:
        if paper.arxiv_id in existing_ids:
            log.info(f"Skipping duplicate: {paper.arxiv_id}")
            stats["skipped"] += 1
            continue

        try:
            if USE_CLAUDE:
                log.info(f"Enriching with Claude: {paper.title[:60]}...")
                enriched = enrich_with_claude(paper, ai_client)
            else:
                log.info(f"Enriching with KeyBERT: {paper.title[:60]}...")
                enriched = enrich_without_claude(paper)

            log.info(f"Inserting into Notion: {paper.arxiv_id}")
            insert_paper(paper, enriched, ai_enriched=USE_CLAUDE)

            existing_ids.add(paper.arxiv_id)
            stats["inserted"] += 1
            time.sleep(0.5)  # small pause between Notion writes

        except json.JSONDecodeError as e:
            log.error(f"Claude returned invalid JSON for {paper.arxiv_id}: {e}")
            stats["failed"] += 1
        except requests.HTTPError as e:
            log.error(f"Notion API error for {paper.arxiv_id}: {e.response.text}")
            stats["failed"] += 1
        except Exception as e:
            log.error(f"Unexpected error for {paper.arxiv_id}: {e}")
            stats["failed"] += 1

    log.info("=" * 55)
    log.info(f"   Done — inserted: {stats['inserted']}  |  "
             f"skipped: {stats['skipped']}  |  failed: {stats['failed']}")
    log.info("=" * 55)


if __name__ == "__main__":
    run_pipeline()
