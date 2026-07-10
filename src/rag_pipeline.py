"""End-to-end RAG pipeline: FAISS retrieval (fine-tuned embeddings) + Ollama
LLM generation (Llama 3.2 by default, falls back to any locally pulled model)
+ advanced prompt engineering + LLMOps guardrails/logging.
"""

from __future__ import annotations

import time
import subprocess

from langchain_ollama import OllamaLLM

from .build_vectorstore import load_vectorstore
from .prompt_templates import build_rag_prompt
from .llmops import (
    QueryLogEntry,
    contains_emergency_keywords,
    score_to_confidence,
    log_query,
    CONFIDENCE_THRESHOLD,
)

PREFERRED_MODELS = ["llama3.2", "gemma2:2b", "qwen3:0.6b"]


def _pick_available_ollama_model() -> str:
    """Pick the first preferred model that is actually pulled locally, so the
    notebook runs out-of-the-box regardless of which model the grader has.
    Falls back to the first preferred name (Ollama will error clearly if missing).
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        installed = result.stdout
        for name in PREFERRED_MODELS:
            if name.split(":")[0] in installed:
                return name
    except Exception:
        pass
    return PREFERRED_MODELS[0]


class MedicalRAGPipeline:
    def __init__(self, model_name: str | None = None, k: int = 3, temperature: float = 0.2):
        self.model_name = model_name or _pick_available_ollama_model()
        self.k = k
        self.vectorstore = load_vectorstore()
        self.llm = OllamaLLM(model=self.model_name, temperature=temperature)

    def retrieve(self, symptoms: str):
        """Return (documents, confidence_scores) sorted by relevance."""
        results = self.vectorstore.similarity_search_with_score(symptoms, k=self.k)
        docs = [r[0] for r in results]
        confidences = [score_to_confidence(r[1]) for r in results]
        return docs, confidences

    def generate(self, symptoms: str, context: str) -> str:
        prompt = build_rag_prompt(context=context, question=symptoms)
        return self.llm.invoke(prompt)

    def query(self, symptoms: str, verbose: bool = False) -> dict:
        start = time.time()
        docs, confidences = self.retrieve(symptoms)
        context = "\n\n".join(d.page_content for d in docs)
        top_score = confidences[0] if confidences else 0.0
        low_confidence = top_score < CONFIDENCE_THRESHOLD
        emergency = contains_emergency_keywords(symptoms)

        response = self.generate(symptoms, context)

        if low_confidence:
            response = (
                "[LOW CONFIDENCE RETRIEVAL — the knowledge base does not contain a "
                "strong match for these symptoms. Treat the answer below as tentative "
                "and consult a clinician.]\n\n" + response
            )
        if emergency:
            response = (
                "EMERGENCY KEYWORDS DETECTED: If this is a medical emergency, call your "
                "local emergency number immediately.\n\n" + response
            )

        latency = time.time() - start
        retrieved_conditions = [d.metadata.get("condition") for d in docs]

        entry = QueryLogEntry(
            timestamp=start,
            query=symptoms,
            retrieved_conditions=retrieved_conditions,
            top_score=top_score,
            low_confidence=low_confidence,
            emergency_flag=emergency,
            latency_seconds=latency,
            response_preview=response[:200],
            metadata={"model": self.model_name},
        )
        log_query(entry)

        result = {
            "symptoms": symptoms,
            "retrieved_conditions": retrieved_conditions,
            "confidences": confidences,
            "low_confidence": low_confidence,
            "emergency_flag": emergency,
            "response": response,
            "latency_seconds": latency,
            "model": self.model_name,
        }
        if verbose:
            print(f"[model={self.model_name}] retrieved={retrieved_conditions} "
                  f"top_conf={top_score:.2f} latency={latency:.2f}s")
        return result
