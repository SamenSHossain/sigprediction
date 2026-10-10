"""Identifiers: idempotency keys and deterministic hashes."""

from __future__ import annotations

import hashlib
import uuid


def new_idempotency_key() -> str:
    return uuid.uuid4().hex


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:32]
