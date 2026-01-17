#!/usr/bin/env python3
"""Focused benchmark: doc2 only, Q10-Q11, comparing temp=0 single vs 20 doers/7 judges ASAP."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from llm_agg.config import RunConfig, ModelRow, BenchmarkConfig, ScorerConfig
from llm_agg.runner import run_pipeline
from llm_agg.cli import _load_doc, _parse_questions_md

GEMINI_MODEL = "google/gemini-2.0-flash-001"
DETAILED_DATA_FILE = Path("benchmark_focused_data.jsonl")
MAX_CONCURRENCY = 1000
NUM_ITERATIONS = 100

def load_test_data():
    """Load only doc2 and questions 10, 11."""
    test_data_path = Path("TestData")

    # Load only doc2
    doc_path = test_data_path / "doc2.pdf"
    doc = _load_doc(str(doc_path), "doc2")
    docs = [doc]

    # Load all questions, then filter to Q10 and Q11 (indices 9, 10)
    questions_path = test_data_path / "Questions.md"
    all_questions, all_ground_truths = _parse_questions_md(str(questions_path))

    # Only Q10 and Q11 (0-indexed: 9, 10)
    questions = [all_questions[9], all_questions[10]]
    ground_truths = [all_ground_truths[9], all_ground_truths[10]]

    return docs, questions, ground_truths

def create_single_model_config(docs, questions, ground_truths):
    """Single model: temp=0, 1 call."""
    return RunConfig(
        questions=questions,
        docs=docs,
        doers=[ModelRow(model_id=GEMINI_MODEL, timeout_s=60, n_calls=1, temperature=0.0)],
        judges=[],
        final_judges=[],
        cap_total_calls=500,
        max_output_tokens=300,
        retries=1,
        max_concurrency=MAX_CONCURRENCY,
        benchmark=BenchmarkConfig(
            enabled=True,
            mode="llm",
            strip_whitespace=True,
            scorer=ScorerConfig(model_id=GEMINI_MODEL, timeout_s=30, temperature=0.0),
            ground_truths=ground_truths,
        ),
    )

def create_asap_config(docs, questions, ground_truths):
    """ASAP: 20 doers (temp 0.7), 7 judges (temp 0.3), 1 final (temp 0.0)."""
    return RunConfig(
        questions=questions,
        docs=docs,
        doers=[ModelRow(model_id=GEMINI_MODEL, timeout_s=60, n_calls=20, temperature=0.7)],
        judges=[ModelRow(model_id=GEMINI_MODEL, timeout_s=60, n_calls=7, temperature=0.3)],
        final_judges=[ModelRow(model_id=GEMINI_MODEL, timeout_s=60, n_calls=1, temperature=0.0)],
        send_doc_to_judges=True,
        send_doc_to_final_judges=True,
        send_doer_responses_to_judges=True,
        send_doer_outputs_to_final_judges=True,
        send_judge_outputs_to_final_judges=True,
        cap_total_calls=5000,
        max_output_tokens=300,
        retries=1,
        max_concurrency=MAX_CONCURRENCY,
        benchmark=BenchmarkConfig(
            enabled=True,
            mode="llm",
            strip_whitespace=True,
            scorer=ScorerConfig(model_id=GEMINI_MODEL, timeout_s=30, temperature=0.0),
            ground_truths=ground_truths,
        ),
    )

def save_detailed_record(benchmark_type, iteration, results, scores, questions, ground_truths):
    """Save detailed record to JSONL file."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_type": benchmark_type,
        "iteration": iteration,
        "results": [],
        "scores": scores,
    }
    for item in results:
        result_entry = {
            "doc_id": item["doc_id"],
            "q_index": item["q_index"],
            "question": item["question"],
            "ground_truth": ground_truths[item["q_index"]] if item["q_index"] < len(ground_truths) else None,
            "doer_responses": [{"model_id": d["model_id"], "call_index": d["call_index"], "text": d["text"], "status": d["status"]} for d in item["doers"]],
            "judge_responses": [{"model_id": j["model_id"], "call_index": j["call_index"], "text": j["text"], "status": j["status"]} for j in item["judges"]],
            "final_responses": [{"model_id": f["model_id"], "call_index": f["call_index"], "text": f["text"], "status": f["status"]} for f in item["finals"]],
            "primary_output": item["primary_outputs"][0]["text"] if item["primary_outputs"] else None,
        }
        record["results"].append(result_entry)
    with open(DETAILED_DATA_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

async def run_single_iteration(config, run_id, iteration, benchmark_type, questions, ground_truths):
    """Run a single iteration."""
    results, attempts, scores = await run_pipeline(config, run_id)
    save_detailed_record(benchmark_type, iteration, results, scores, questions, ground_truths)
    return scores

async def main():
    print("=" * 60)
    print("Focused Benchmark: doc2, Q10-Q11")
    print("Single Model: temp=0")
    print("ASAP: 20 doers, 7 judges, 1 final")
    print(f"Max concurrency: {MAX_CONCURRENCY}")
    print("=" * 60)

    # Clear previous data
    if DETAILED_DATA_FILE.exists():
        DETAILED_DATA_FILE.unlink()

    docs, questions, ground_truths = load_test_data()
    print(f"Loaded {len(docs)} document, {len(questions)} questions")

    # Run all Single Model iterations in parallel
    print(f"\nLaunching {NUM_ITERATIONS} Single Model iterations in parallel...")
    single_tasks = []
    for i in range(NUM_ITERATIONS):
        config = create_single_model_config(docs, questions, ground_truths)
        run_id = f"single_{i:03d}"
        single_tasks.append(run_single_iteration(config, run_id, i, "Single Model", questions, ground_truths))

    single_results = await asyncio.gather(*single_tasks, return_exceptions=True)
    single_success = sum(1 for r in single_results if not isinstance(r, Exception))
    single_errors = sum(1 for r in single_results if isinstance(r, Exception))
    print(f"Single Model: {single_success} successful, {single_errors} errors")

    # Run all ASAP iterations in parallel
    print(f"\nLaunching {NUM_ITERATIONS} ASAP iterations in parallel...")
    asap_tasks = []
    for i in range(NUM_ITERATIONS):
        config = create_asap_config(docs, questions, ground_truths)
        run_id = f"asap_{i:03d}"
        asap_tasks.append(run_single_iteration(config, run_id, i, "ASAP", questions, ground_truths))

    asap_results = await asyncio.gather(*asap_tasks, return_exceptions=True)
    asap_success = sum(1 for r in asap_results if not isinstance(r, Exception))
    asap_errors = sum(1 for r in asap_results if isinstance(r, Exception))
    print(f"ASAP: {asap_success} successful, {asap_errors} errors")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print(f"Results saved to: {DETAILED_DATA_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
