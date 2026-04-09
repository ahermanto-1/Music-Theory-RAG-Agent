"""
eval_pipeline.py — DeepEval quality gate for the Music Theory RAG Agent.

Runs Faithfulness and Answer Relevancy on 2 synthetic questions using
Gemma 3 27B as the LLM judge. Logs failures (score < 0.8) to failures.csv.
Exits with code 1 if any metric fails — compatible with GitHub Actions.

Usage:
    python3 eval_pipeline.py
"""

import asyncio
import csv
import json
import os
import re
import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from google import genai
from pydantic import BaseModel

from agent import run

# ── Config ────────────────────────────────────────────────────────────────────

JUDGE_MODEL = "gemma-3-27b-it"
PASS_THRESHOLD = 0.8
FAILURES_CSV = "failures.csv"

SYNTHETIC_QUESTIONS = [
    "What intervals make up a major triad and how are they arranged on the guitar fretboard?",
    "How does the circle of fifths help in understanding key signatures and chord progressions?",
]

# ── Gemma judge ───────────────────────────────────────────────────────────────

class GemmaJudge(DeepEvalBaseLLM):
    """Wraps Gemma 3 27B (via Google GenAI) as a DeepEval judge."""

    def __init__(self):
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema: Optional[type[BaseModel]] = None):
        response = self._client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
        )
        text = response.text.strip()

        if schema is not None:
            # Extract JSON from markdown code fences if present
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if match:
                text = match.group(1).strip()
            try:
                return schema.model_validate_json(text)
            except Exception:
                # Fallback: try parsing raw text as JSON
                return schema.model_validate(json.loads(text))

        return text

    async def a_generate(self, prompt: str, schema: Optional[type[BaseModel]] = None):
        # Run the synchronous call in a thread to satisfy the async interface
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return JUDGE_MODEL


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_question(
    question: str,
    judge: GemmaJudge,
) -> list[dict]:
    """
    Run the RAG agent on a question, then score Faithfulness and Answer Relevancy.
    Returns a list of result dicts (one per metric).
    """
    print(f"\n{'─' * 60}")
    print(f"Question: {question}")

    answer, chunks, rewritten = run(question)
    retrieval_context = [c["text"] for c in chunks]

    if rewritten:
        print(f"  Query rewritten to: {rewritten}")

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieval_context,
    )

    results = []
    for metric_cls, name in [
        (FaithfulnessMetric, "Faithfulness"),
        (AnswerRelevancyMetric, "Answer Relevancy"),
    ]:
        metric = metric_cls(
            threshold=PASS_THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        )
        metric.measure(test_case)

        score = round(metric.score, 4)
        passed = score >= PASS_THRESHOLD
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {score:.4f} [{status}]")
        if metric.reason:
            print(f"    Reason: {metric.reason}")

        results.append({
            "question": question,
            "metric": name,
            "score": score,
            "passed": passed,
            "reason": metric.reason or "",
            "rewritten_query": rewritten or "",
            "sources": ", ".join(c["source"] for c in chunks),
        })

    return results


def write_failures(failures: list[dict]) -> None:
    if not failures:
        return

    fieldnames = ["timestamp", "question", "metric", "score", "reason", "rewritten_query", "sources"]
    write_header = not os.path.exists(FAILURES_CSV)

    with open(FAILURES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in failures:
            writer.writerow({
                "timestamp": datetime.utcnow().isoformat(),
                "question": row["question"],
                "metric": row["metric"],
                "score": row["score"],
                "reason": row["reason"],
                "rewritten_query": row["rewritten_query"],
                "sources": row["sources"],
            })

    print(f"\n{len(failures)} failure(s) written to {FAILURES_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"DeepEval Quality Gate")
    print(f"Judge model : {JUDGE_MODEL}")
    print(f"Pass threshold : {PASS_THRESHOLD}")
    print(f"Questions : {len(SYNTHETIC_QUESTIONS)}")

    judge = GemmaJudge()
    all_results: list[dict] = []

    for question in SYNTHETIC_QUESTIONS:
        all_results.extend(evaluate_question(question, judge))

    failures = [r for r in all_results if not r["passed"]]
    write_failures(failures)

    # ── Summary ──
    total = len(all_results)
    passed = total - len(failures)
    print(f"\n{'═' * 60}")
    print(f"Results: {passed}/{total} metrics passed")

    if failures:
        print(f"\nFailed metrics:")
        for f in failures:
            print(f"  - [{f['metric']}] \"{f['question'][:60]}…\" → {f['score']:.4f}")
        print(f"\nQuality gate FAILED — review {FAILURES_CSV}")
        sys.exit(1)  # Non-zero exit fails the GitHub Actions step
    else:
        print("Quality gate PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
