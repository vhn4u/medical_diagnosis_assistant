"""Lightweight LLMOps and Responsible-AI layer.

Provides:
  - Structured logging of every RAG query (inputs, retrieved sources, output, latency).
  - A simple retrieval-confidence guardrail (reject / flag low-similarity matches).
  - An emergency-keyword safety filter that prepends urgent-care guidance.
  - A minimal offline evaluation harness (keyword/condition-hit-rate) for the RAG pipeline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "outputs"
LOG_FILE = LOG_DIR / "query_log.jsonl"

EMERGENCY_KEYWORDS = [
    "chest pain", "difficulty breathing", "shortness of breath", "slurred speech",
    "sudden numbness", "severe bleeding", "loss of consciousness", "stroke",
    "suicidal", "can't breathe", "cannot breathe",
]

# Minimum cosine-similarity-derived confidence (0-1) below which we flag
# the retrieval as low-confidence rather than silently answering.
CONFIDENCE_THRESHOLD = 0.35


@dataclass
class QueryLogEntry:
    timestamp: float
    query: str
    retrieved_conditions: list[str]
    top_score: float
    low_confidence: bool
    emergency_flag: bool
    latency_seconds: float
    response_preview: str
    metadata: dict = field(default_factory=dict)


def contains_emergency_keywords(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in EMERGENCY_KEYWORDS)


def score_to_confidence(distance: float) -> float:
    """FAISS returns L2 distance for MiniLM embeddings; convert to a rough
    0-1 confidence score (smaller distance -> higher confidence)."""
    return float(max(0.0, 1.0 - (float(distance) / 2.0)))


def log_query(entry: QueryLogEntry) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def evaluate_rag_pipeline(rag_query_fn, eval_cases: list[dict]) -> dict:
    """Run a small offline evaluation set through the pipeline and compute a
    condition hit-rate: did the top retrieved condition match the expected one?

    eval_cases: [{"symptoms": "...", "expected_condition": "Migraine"}, ...]
    """
    results = []
    hits = 0
    for case in eval_cases:
        start = time.time()
        result = rag_query_fn(case["symptoms"])
        latency = time.time() - start
        top_condition = result["retrieved_conditions"][0] if result["retrieved_conditions"] else None
        hit = top_condition == case["expected_condition"]
        hits += int(hit)
        results.append(
            {
                "symptoms": case["symptoms"],
                "expected": case["expected_condition"],
                "retrieved_top": top_condition,
                "hit": hit,
                "latency_seconds": round(latency, 3),
            }
        )
    return {
        "hit_rate": hits / len(eval_cases) if eval_cases else 0.0,
        "n_cases": len(eval_cases),
        "details": results,
    }
