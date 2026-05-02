# arXiv → Notion Pipeline

Automated pipeline that fetches research papers from arXiv and stores them in a Notion database, with optional AI-powered keyword extraction and summarization.

---

## How it works

```
papers.py          →       pipeline.py        →     Notion database
(arXiv IDs)            (fetch + enrich)            (one row per paper)
```

1. You add arXiv IDs manually to `load_papers.py`
2. The pipeline fetches metadata from arXiv (title, authors, abstract, etc.)
3. Keywords are extracted — either with **KeyBERT** (free, local) or **Claude AI** (API)
4. Each paper is inserted as a new row in your Notion database
5. Duplicate papers are automatically skipped

---

## Project structure

```
cosmo-coffee-poland/
├── pipeline.py       — main pipeline (do not edit)
├── load_papers.py    — list of arXiv IDs to process (edit this)
├── .env              — secret tokens (never commit to git)
├── .gitignore
└── README.md
```

---

## Setup

### 1. Activate the virtual environment

```bash
source ~/projects/python_envs/arxiv/bin/activate
```

### 2. Install dependencies

```bash
pip install arxiv anthropic requests python-dotenv keybert
```

### 3. Configure your `.env` file

Create a `.env` file in the project folder:

```bash
NOTION_TOKEN=ntn_your_token_here
NOTION_DATABASE_ID=your_database_id_here
ANTHROPIC_API_KEY=sk-ant-your_key_here   # only needed if USE_CLAUDE = True
```

| Variable | Where to get it |
|---|---|
| `NOTION_TOKEN` | notion.so/my-integrations → your integration → Internal Integration Secret |
| `NOTION_DATABASE_ID` | Open your Notion database → copy link → ID before `?v=` |
| `ANTHROPIC_API_KEY` | console.anthropic.com → API Keys |

### 4. Connect Notion integration to your database

```
1. Open Papers_arxiv in Notion
2. Click "..." → Connections → Add connection
3. Select your integration
4. Click Confirm
```

---

## Notion database schema

Create these properties in your `Papers_arxiv` database:

| Property | Type | Filled when |
|---|---|---|
| `Title` | Title | Always |
| `Authors` | Text | Always |
| `Abstract` | Text | Always |
| `Summary` | Text | Only if `USE_CLAUDE = True` |
| `Keywords` | Multi-select | Always (KeyBERT or Claude) |
| `Category` | Select | Always |
| `arXiv ID` | Text | Always (used for deduplication) |
| `URL` | URL | Always |
| `Published` | Date | Always |
| `Added On` | Date | Always |
| `arXiv Categories` | Multi-select | Always |
| `AI Enriched` | Checkbox | Always (True = Claude, False = KeyBERT) |

---

## Usage

### Step 1 — Add papers to `load_papers.py`

Open `load_papers.py` and add the arXiv IDs of papers you find on Bentleyfield:

```python
PAPER_IDS = [
    "2603.29021",   # The Evolution of the Spin Alignments...
    "2301.07041",   # add more IDs here
]
```

### Step 2 — Choose enrichment mode

At the top of `pipeline.py`, set the flag:

```python
USE_CLAUDE   = False  # True = Claude AI  |  False = KeyBERT (free)
NUM_KEYWORDS = 8      # number of keywords to extract per paper
```

| | `USE_CLAUDE = False` | `USE_CLAUDE = True` |
|---|---|---|
| **Abstract** | ✅ Original from arXiv | ✅ Original from arXiv |
| **Summary** | ⬜ empty | ✅ Plain-language summary by Claude |
| **Keywords** | ✅ Extracted by KeyBERT (SciBERT model) | ✅ Extracted by Claude |
| **API cost** | Free | Uses Anthropic credit |

### Step 3 — Run the pipeline

```bash
python pipeline.py
```

Papers already in Notion are automatically skipped — safe to run multiple times.

---

## Notes

- The first run with `USE_CLAUDE = False` downloads the SciBERT model (~500MB) — subsequent runs use the cached version and start instantly.
- The `.env` file must never be committed to git. Add it to `.gitignore`:
  ```bash
  echo ".env" >> .gitignore
  ```
- arXiv rate limiting: the pipeline pauses 3 seconds between arXiv requests and 0.5 seconds between Notion writes — this is intentional and respectful.

