"""
rag.py â€” Retrieval-Augmented Generation for the support inbox.

Two things live here:
  1. A small ingestion pipeline that embeds past tickets (and, later, any
     SOP/FAQ docs you drop in knowledge_base/docs/) into a local Chroma
     vector store.
  2. retrieve_similar() + draft_reply(), used by the /api/draft-reply/
     endpoint to pull relevant context and ask Groq to write a grounded
     first-draft response.

This is intentionally separate from ml_model.py. ml_model.py classifies
(network/hardware/software/access) - a closed-set problem that doesn't need
retrieval. This file is for *generation that needs grounding* - drafting a
reply, which benefits from seeing how similar tickets were handled before.

Setup:
    pip install chromadb
    python manage.py build_kb        # builds/refreshes the vector store
"""

import csv
import os
from pathlib import Path

import chromadb
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root (next to manage.py's parent)
DATASET_CSV = BASE_DIR / "dataset" / "tickets.csv"
DOCS_DIR = BASE_DIR / "django_project" / "knowledge_base" / "docs"
CHROMA_DIR = BASE_DIR / "django_project" / "chroma_db"
COLLECTION_NAME = "support_knowledge"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def _get_collection():
    """Open (or create) the persistent Chroma collection.

    Uses Chroma's bundled default embedding function (all-MiniLM-L6-v2,
    runs locally on CPU, no API key needed) so embedding cost is zero.
    """
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


# â”€â”€ Ingestion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def ingest_tickets_csv(csv_path: Path = DATASET_CSV, batch_size: int = 200) -> int:
    """Embed every row of tickets.csv into the vector store.

    Each ticket becomes one document: "<subject>\n<description>", tagged
    with category/subcategory/device_type as metadata so results can be
    filtered or displayed alongside the draft reply.
    """
    collection = _get_collection()

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    docs, metadatas, ids = [], [], []
    count = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            subject = (row.get("ticket_subject") or "").strip()
            description = (row.get("ticket_description") or "").strip()
            if not subject and not description:
                continue

            text = f"{subject}\n{description}".strip()
            docs.append(text)
            metadatas.append({
                "source": "tickets.csv",
                "category": row.get("category", "") or "unknown",
                "subcategory": row.get("subcategory", "") or "",
                "device_type": row.get("device_type", "") or "",
            })
            ids.append(f"ticket-{i}")

            if len(docs) >= batch_size:
                collection.upsert(documents=docs, metadatas=metadatas, ids=ids)
                count += len(docs)
                docs, metadatas, ids = [], [], []

    if docs:
        collection.upsert(documents=docs, metadatas=metadatas, ids=ids)
        count += len(docs)

    return count


def ingest_docs_folder(docs_dir: Path = DOCS_DIR, chunk_size: int = 800) -> int:
    """Embed any .txt/.md files dropped in knowledge_base/docs/.

    This is where future SOPs, runbooks, or FAQ docs go. Each file is
    split into rough fixed-size chunks (simple char-based chunking -
    good enough for short SOPs; swap for a smarter splitter if your
    docs get long and structured).
    """
    collection = _get_collection()
    docs_dir.mkdir(parents=True, exist_ok=True)

    docs, metadatas, ids = [], [], []
    count = 0

    for path in sorted(docs_dir.glob("*")):
        if path.suffix.lower() not in (".txt", ".md"):
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        for j, chunk in enumerate(chunks):
            docs.append(chunk)
            metadatas.append({"source": path.name, "category": "doc", "subcategory": "", "device_type": ""})
            ids.append(f"doc-{path.stem}-{j}")
            count += 1

    if docs:
        collection.upsert(documents=docs, metadatas=metadatas, ids=ids)

    return count


# â”€â”€ Retrieval + generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def retrieve_similar(query_text: str, k: int = 3) -> list[dict]:
    """Return the k most similar past tickets/docs for a new email."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[query_text], n_results=min(k, collection.count()))

    matches = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        matches.append({"text": doc, "metadata": meta, "distance": dist})
    return matches


def draft_reply(ticket_text: str, category: str, retrieved: list[dict]) -> str:
    """Ask Groq to draft a reply, grounded in the retrieved context.

    This is the "generation" half of RAG - it never answers from the
    model's general knowledge alone, it's told to lean on the retrieved
    similar tickets for how this category of issue is normally resolved.
    """
    if _client is None:
        return "[draft_reply unavailable: GROQ_API_KEY not set]"

    if retrieved:
        context_block = "\n\n".join(
            f"Similar past ticket (category: {m['metadata'].get('category', 'unknown')}):\n{m['text']}"
            for m in retrieved
        )
    else:
        context_block = "(No similar past tickets found - use general best practice for this category.)"

    prompt = f"""You are an IT helpdesk agent drafting a first-pass reply to a support email.

STRICT RULES - follow these exactly:
- Never use placeholders like [agent name], [customer name], [your name] or any text in square brackets.
- Always start the email with exactly: "Hi,"
- Write the full reply with no blanks to fill in.

Ticket category: {category}

New ticket:
\"\"\"{ticket_text}\"\"\"

Context from similar past tickets:
{context_block}

Write a short, helpful, professional draft reply to the customer. Base your
troubleshooting suggestions on the patterns in the similar past tickets where
relevant. Do not invent specific account details. This draft will be reviewed
by a human agent before sending, so it's fine to leave placeholders like
[agent name] where appropriate.

Reply with ONLY the email body. No subject line, no explanation."""

    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
