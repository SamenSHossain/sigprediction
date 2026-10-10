"""Stance classification with the Claude API (Batch API, sentiment.max_posts_per_batch posts per request,
JSON output {post_id, race_id, party, stance, confidence, is_news_or_poll}); cached by text_hash;
escalates 'unclear' to the escalation model. Writes Stance rows and marks posts classified."""

from __future__ import annotations

raise NotImplementedError("implemented in the news/social workstream")
