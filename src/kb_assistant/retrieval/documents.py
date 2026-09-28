"""Loading and chunking the document corpus.

Chunking is structure-aware: one chunk per `## ` section, because a section ("Root Cause",
"Rollback steps") is the unit a person would cite. Sections longer than `max_chars` are split on
paragraph boundaries with overlap. Every chunk starts with a short header naming the document and
section, so a chunk that says "the pool was exhausted" still matches a query about payments-ledger.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import yaml

REQUIRED_KEYS = ("doc_id", "title", "department", "document_type", "access_level", "created_date")
DOCUMENT_TYPES = ("policy", "architecture", "runbook", "incident", "product_spec", "meeting_notes")

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)
_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.M)


@dataclass
class Document:
    doc_id: str
    title: str
    department: str
    document_type: str
    access_level: str
    created_date: str
    tags: list[str]
    body: str
    path: str

    @property
    def created_ts(self) -> int:
        """Dates as a YYYYMMDD integer: Pinecone range filters ($gte/$lte) only work on numbers."""
        return date_to_ts(self.created_date)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    department: str
    document_type: str
    access_level: str
    created_date: str
    created_ts: int
    tags: list[str] = field(default_factory=list)

    def metadata(self) -> dict:
        """Everything except the id, as stored next to the vector (also used for attribution)."""
        data = asdict(self)
        data.pop("chunk_id")
        return data


def date_to_ts(value: str | date) -> int:
    return int(str(value)[:10].replace("-", ""))


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER.match(raw)
    if not match:
        raise ValueError(f"{path}: missing YAML frontmatter")
    meta = yaml.safe_load(match.group(1)) or {}
    missing = [k for k in REQUIRED_KEYS if k not in meta]
    if missing:
        raise ValueError(f"{path}: frontmatter missing {missing}")
    return Document(
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        department=str(meta["department"]),
        document_type=str(meta["document_type"]),
        access_level=str(meta["access_level"]),
        created_date=str(meta["created_date"])[:10],
        tags=[str(t) for t in meta.get("tags") or []],
        body=match.group(2).strip(),
        path=str(path),
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    docs = [parse_document(p) for p in sorted(corpus_dir.rglob("*.md"))]
    seen: set[str] = set()
    for doc in docs:
        if doc.doc_id in seen:
            raise ValueError(f"duplicate doc_id {doc.doc_id}")
        seen.add(doc.doc_id)
    return docs


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "section"


def split_sections(body: str) -> list[tuple[str, str]]:
    """Return (heading, text) pairs. Text before the first heading becomes an 'Overview' section."""
    parts: list[tuple[str, str]] = []
    matches = list(_SECTION.finditer(body))
    preamble = body[: matches[0].start()].strip() if matches else body.strip()
    if preamble:
        parts.append(("Overview", preamble))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[m.end():end].strip()
        if text:
            parts.append((m.group(1).strip(), text))
    return parts


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            pieces.append(current)
            current = current[-overlap:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    # A single paragraph longer than max_chars: fall back to a hard split with overlap.
    out: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            out.append(piece[:max_chars])
            piece = piece[max_chars - overlap:]
        out.append(piece)
    return out


def chunk_document(doc: Document, max_chars: int = 1400, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    used_ids: set[str] = set()
    for heading, text in split_sections(doc.body):
        for n, piece in enumerate(_split_long(text, max_chars, overlap)):
            chunk_id = f"{doc.doc_id}#{_slug(heading)}" + (f"-{n + 1}" if n else "")
            while chunk_id in used_ids:  # two sections with the same heading
                chunk_id += "-x"
            used_ids.add(chunk_id)
            header = f"[{doc.doc_id}] {doc.title} | {doc.document_type} | {doc.department} | {heading}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=heading,
                    text=f"{header}\n{piece}",
                    department=doc.department,
                    document_type=doc.document_type,
                    access_level=doc.access_level,
                    created_date=doc.created_date,
                    created_ts=doc.created_ts,
                    tags=doc.tags,
                )
            )
    return chunks
