#!/usr/bin/env python3
"""Finish remaining ASAP benchmark iterations with high concurrency."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from llm_agg.config import RunConfig, ModelRow, BenchmarkConfig, ScorerConfig
from llm_agg.runner import run_pipeline
from llm_agg.cli import _load_doc, _parse_questions_md

GEMINI_MODEL = "google/gemini-2.0-flash-001"
DETAILED_DATA_FILE = Path("benchmark_detailed_data.jsonl")

def load_test_data():
    test_data_path = Path("TestData")
    docs = []
    for doc_file in ["doc1.txt", "doc2.pdf", "doc3.jpg"]:
        doc_path = test_data_path / doc_file
        if doc_path.exists():
            doc = _load_doc(str(doc_path), doc_path.stem)
            docs.append(doc)
    questions_path = test_data_path / "Questions.md"
    questions, ground_truths = _parse_questions_md(str(questions_path))
    return docs, questions, ground_truths

def create_asap_config(docs, questions, ground_truths):
    return RunConfig(
        questions=questions,
        docs=docs,
        doers=[ModelRow(model_id=GEMINI_MODEL, timeout_s=60, n_calls=10, temperature=0.7)],
        judges=[ModelRow(model_id=GEMINI_MODEL, timeout_s=60, n_calls=5, temperature=0.3)],
        final_judges=[ModelRow(model_id=GEMINI_MODEL, timeout_s=60, n_calls=1, temperature=0.0)],
        send_doc_to_judges=True,
        send_doc_to_final_judges=True,
        send_doer_responses_to_judges=True,
        send_doer_outputs_to_final_judges=True,
        send_judge_outputs_to_final_judges=True,
        cap_total_calls=5000,
        max_output_tokens=300,
        retries=1,
        max_concurrency=2000,  # HIGH CONCURRENCY
        benchmark=BenchmarkConfig(
            enabled=True,
            mode="llm",
            strip_whitespace=True,
            scorer=ScorerConfig(model_id=GEMINI_MODEL, timeout_s=30, temperature=0.0),
            ground_truths=ground_truths,
        ),
    )

def save_detailed_record(benchmark_type, iteration, results, scores, questions, ground_truths):
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

async def run_iteration(config, run_id, iteration, questions, ground_truths):
    results, attempts, scores = await run_pipeline(config, run_id)
    save_detailed_record("ASAP", iteration, results, scores, questions, ground_truths)
    return scores

async def main():
    # Count existing ASAP iterations
    existing_asap = 0
    with open(DETAILED_DATA_FILE) as f:
        for line in f:
            record = json.loads(line)
            if record["benchmark_type"] == "ASAP":
                existing_asap += 1

    remaining = 100 - existing_asap
    print(f"Existing ASAP iterations: {existing_asap}", flush=True)
    print(f"Remaining iterations: {remaining}", flush=True)

    if remaining <= 0:
        print("All ASAP iterations complete!", flush=True)
        return

    docs, questions, ground_truths = load_test_data()

    # Run all remaining iterations IN PARALLEL
    print(f"Starting {remaining} iterations in parallel with concurrency=2000...", flush=True)

    tasks = []
    for i in range(remaining):
        iteration_num = existing_asap + i
        config = create_asap_config(docs, questions, ground_truths)
        run_id = f"asap_{iteration_num:03d}"
        tasks.append(run_iteration(config, run_id, iteration_num, questions, ground_truths))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    success = sum(1 for r in results if not isinstance(r, Exception))
    errors = sum(1 for r in results if isinstance(r, Exception))

    print(f"\nCompleted: {success} successful, {errors} errors", flush=True)
    print("Done!", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
