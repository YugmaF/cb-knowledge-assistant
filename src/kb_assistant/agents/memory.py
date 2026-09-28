"""Memory.

Three layers, each with a different lifetime and a different owner:

  short-term   The conversation of one session (thread). Stored by the LangGraph checkpointer, so
               it survives turns, API restarts and the human-approval pause. Bounded by a condenser
               (below).
  user context Who is asking: name, role and department from the verified token, plus facts the
               user stated about themselves ("I'm on the payments on-call rota"). Long-term, per user.
  episodic     Past questions and the documents that answered them, across all of this user's
               sessions, recalled by embedding similarity when relevant to the new question.

Condenser rules (learned the hard way on an agent platform: each summary of a summary loses detail,
and a summary once claimed the user had never given an ID they gave 30 events earlier):
  * trigger on estimated tokens, not on message count, so one huge turn cannot blow the window
  * keep the last N turns verbatim
  * user questions are pinned: always kept verbatim, never rewritten by the summariser
  * only assistant answers are condensed, into a running summary

Memory is always scoped by user_id. There is no code path that reads another user's memory.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kb_assistant.retrieval.embeddings import Embedder

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_facts (
    user_id TEXT NOT NULL, fact TEXT NOT NULL, created_at REAL NOT NULL, UNIQUE(user_id, fact));
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, thread_id TEXT NOT NULL, question TEXT NOT NULL,
    answer_summary TEXT NOT NULL, doc_ids TEXT NOT NULL, embedding TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, thread_id TEXT NOT NULL, run_id TEXT NOT NULL,
    score INTEGER NOT NULL, comment TEXT, question TEXT, answer TEXT, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ix_interactions_user ON interactions(user_id);
"""


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # ~4 characters per token for English


@dataclass
class RecalledInteraction:
    question: str
    answer_summary: str
    doc_ids: list[str]
    similarity: float
    thread_id: str


class MemoryStore:
    def __init__(self, path: Path, embedder: Embedder) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._embedder = embedder
        with self._connect() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # sqlite3's own context manager commits but never closes; this one does both.
        db = sqlite3.connect(self._path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    async def _run(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    # --- threads (session ownership) --------------------------------------------------------

    async def claim_thread(self, thread_id: str, user_id: str, title: str) -> bool:
        """Bind a new thread to its creator. Returns False if another user already owns it:
        a guessed thread id must never give access to someone else's conversation."""
        def _claim() -> bool:
            with self._connect() as db:
                row = db.execute("SELECT user_id FROM threads WHERE thread_id=?", (thread_id,)).fetchone()
                if row:
                    return row[0] == user_id
                db.execute("INSERT INTO threads VALUES (?,?,?,?)", (thread_id, user_id, title[:80], time.time()))
                return True
        return await self._run(_claim)

    async def thread_owner(self, thread_id: str) -> str | None:
        def _owner() -> str | None:
            with self._connect() as db:
                row = db.execute("SELECT user_id FROM threads WHERE thread_id=?", (thread_id,)).fetchone()
                return row[0] if row else None
        return await self._run(_owner)

    async def list_threads(self, user_id: str) -> list[dict[str, Any]]:
        def _list() -> list[dict[str, Any]]:
            with self._connect() as db:
                rows = db.execute(
                    "SELECT thread_id, title, created_at FROM threads WHERE user_id=? ORDER BY created_at DESC LIMIT 20",
                    (user_id,)).fetchall()
            return [{"thread_id": r[0], "title": r[1], "created_at": r[2]} for r in rows]
        return await self._run(_list)

    # --- user facts ---------------------------------------------------------------------------

    async def add_facts(self, user_id: str, facts: list[str]) -> list[str]:
        clean = [f.strip()[:200] for f in facts if f and f.strip()]

        def _add() -> list[str]:
            added = []
            with self._connect() as db:
                for fact in clean:
                    cur = db.execute("INSERT OR IGNORE INTO user_facts VALUES (?,?,?)", (user_id, fact, time.time()))
                    if cur.rowcount:
                        added.append(fact)
            return added
        return await self._run(_add)

    async def facts(self, user_id: str, limit: int = 10) -> list[str]:
        def _facts() -> list[str]:
            with self._connect() as db:
                rows = db.execute("SELECT fact FROM user_facts WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                                  (user_id, limit)).fetchall()
            return [r[0] for r in rows]
        return await self._run(_facts)

    # --- episodic -----------------------------------------------------------------------------

    async def remember_interaction(self, user_id: str, thread_id: str, question: str, answer: str,
                                   doc_ids: list[str]) -> None:
        vector = await self._embedder.embed_query(question)

        def _insert() -> None:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO interactions (user_id, thread_id, question, answer_summary, doc_ids, embedding, created_at)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (user_id, thread_id, question[:500], answer[:400], json.dumps(doc_ids[:10]),
                     json.dumps(vector), time.time()))
        await self._run(_insert)

    async def recall(self, user_id: str, question: str, k: int, exclude_thread: str | None = None,
                     min_similarity: float = 0.55) -> list[RecalledInteraction]:
        def _rows() -> list[tuple]:
            with self._connect() as db:
                return db.execute(
                    "SELECT question, answer_summary, doc_ids, embedding, thread_id FROM interactions"
                    " WHERE user_id=? ORDER BY created_at DESC LIMIT 500", (user_id,)).fetchall()
        rows = await self._run(_rows)
        rows = [r for r in rows if r[4] != exclude_thread]
        if not rows:
            return []
        query = np.asarray(await self._embedder.embed_query(question), dtype=np.float32)
        matrix = np.asarray([json.loads(r[3]) for r in rows], dtype=np.float32)
        scores = matrix @ query
        ranked = sorted(zip(scores, rows, strict=True), key=lambda x: x[0], reverse=True)[:k]
        return [RecalledInteraction(r[0], r[1], json.loads(r[2]), float(s), r[4])
                for s, r in ranked if s >= min_similarity]

    # --- feedback -----------------------------------------------------------------------------

    async def add_feedback(self, user_id: str, thread_id: str, run_id: str, score: int, comment: str | None,
                           question: str | None, answer: str | None) -> None:
        def _insert() -> None:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO feedback (user_id, thread_id, run_id, score, comment, question, answer, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (user_id, thread_id, run_id, score, comment, question, answer, time.time()))
        await self._run(_insert)

    async def negative_feedback(self, limit: int = 100) -> list[dict[str, Any]]:
        def _rows() -> list[dict[str, Any]]:
            with self._connect() as db:
                rows = db.execute(
                    "SELECT question, answer, comment, created_at FROM feedback WHERE score < 0"
                    " ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            return [{"question": r[0], "answer": r[1], "comment": r[2], "created_at": r[3]} for r in rows]
        return await self._run(_rows)
