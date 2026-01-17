#!/usr/bin/env python3
"""
Benchmark: ASAP (multi-model aggregation) vs Single Model Run

Compares:
- Single Model: 1 Gemini Flash model, 1 call, no judges
- ASAP: 10 Gemini doers, 5 Gemini judges, 1 Gemini final judge

Runs each configuration 100 times and computes average success rates.
Saves ALL detailed responses for later analysis.
"""

import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from llm_agg.config import RunConfig, DocInfo, ModelRow, BenchmarkConfig, ScorerConfig
from llm_agg.runner import run_pipeline
from llm_agg.cli import _load_doc, _parse_questions_md

GEMINI_MODEL = "google/gemini-2.0-flash-001"
NUM_ITERATIONS = 100

# File to store ALL detailed responses
DETAILED_DATA_FILE = Path("benchmark_detailed_data.jsonl")


def load_test_data():
    """Load documents and questions from TestData folder."""
    test_data_path = Path("TestData")

    # Load documents
    docs = []
    for doc_file in ["doc1.txt", "doc2.pdf", "doc3.jpg"]:
        doc_path = test_data_path / doc_file
        if doc_path.exists():
            doc = _load_doc(str(doc_path), doc_path.stem)
            docs.append(doc)

    # Load questions and ground truths
    questions_path = test_data_path / "Questions.md"
    questions, ground_truths = _parse_questions_md(str(questions_path))

    return docs, questions, ground_truths


def create_single_model_config(docs, questions, ground_truths):
    """Create config for single model run: 1 Gemini, 1 call, no judges."""
    return RunConfig(
        questions=questions,
        docs=docs,
        doers=[
            ModelRow(model_id=GEMINI_MODEL, timeout_s=60, n_calls=1, temperature=0.3)
        ],
        judges=[],  # No judges
        final_judges=[],  # No final judges
        cap_total_calls=500,  # Enough for all questions * docs + scoring
        max_output_tokens=300,
        retries=1,
        max_concurrency=10,
        benchmark=BenchmarkConfig(
            enabled=True,
            mode="llm",  # Use LLM (Gemini) for scoring, not string match
            strip_whitespace=True,
            scorer=ScorerConfig(
                model_id=GEMINI_MODEL,
                timeout_s=30,
                temperature=0.0,
            ),
            ground_truths=ground_truths,
        ),
    )


def create_asap_config(docs, questions, ground_truths):
    """Create config for ASAP run: 10 doers, 5 judges, 1 final judge."""
    return RunConfig(
        questions=questions,
        docs=docs,
        doers=[
            ModelRow(model_id=GEMINI_MODEL, timeout_s=60, n_calls=10, temperature=0.7)
        ],
        judges=[
            ModelRow(model_id=GEMINI_MODEL, timeout_s=60, n_calls=5, temperature=0.3)
        ],
        final_judges=[
            ModelRow(model_id=GEMINI_MODEL, timeout_s=60, n_calls=1, temperature=0.0)
        ],
        send_doc_to_judges=True,
        send_doc_to_final_judges=True,
        send_doer_responses_to_judges=True,
        send_doer_outputs_to_final_judges=True,
        send_judge_outputs_to_final_judges=True,
        cap_total_calls=2000,  # Enough for 10+5+1 calls per question*doc + scoring
        max_output_tokens=300,
        retries=1,
        max_concurrency=10,
        benchmark=BenchmarkConfig(
            enabled=True,
            mode="llm",  # Use LLM (Gemini) for scoring, not string match
            strip_whitespace=True,
            scorer=ScorerConfig(
                model_id=GEMINI_MODEL,
                timeout_s=30,
                temperature=0.0,
            ),
            ground_truths=ground_truths,
        ),
    )


def extract_primary_scores(scores, config):
    """Extract scores from the primary output stage (final if exists, else doer)."""
    primary_stage = "final" if config.final_judges else "doer"
    primary_scores = [s for s in scores if s["stage"] == primary_stage]
    return primary_scores


def save_detailed_record(benchmark_type, iteration, results, scores, questions, ground_truths):
    """Save detailed record of this iteration to JSONL file."""
    # Build detailed record with all responses
    record = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_type": benchmark_type,
        "iteration": iteration,
        "results": [],
        "scores": scores,
    }

    # Extract all responses with their details
    for item in results:
        result_entry = {
            "doc_id": item["doc_id"],
            "q_index": item["q_index"],
            "question": item["question"],
            "ground_truth": ground_truths[item["q_index"]] if item["q_index"] < len(ground_truths) else None,
            "doer_responses": [
                {
                    "model_id": d["model_id"],
                    "call_index": d["call_index"],
                    "text": d["text"],
                    "status": d["status"],
                }
                for d in item["doers"]
            ],
            "judge_responses": [
                {
                    "model_id": j["model_id"],
                    "call_index": j["call_index"],
                    "text": j["text"],
                    "status": j["status"],
                }
                for j in item["judges"]
            ],
            "final_responses": [
                {
                    "model_id": f["model_id"],
                    "call_index": f["call_index"],
                    "text": f["text"],
                    "status": f["status"],
                }
                for f in item["finals"]
            ],
            "primary_output": item["primary_outputs"][0]["text"] if item["primary_outputs"] else None,
        }
        record["results"].append(result_entry)

    # Append to JSONL file
    with open(DETAILED_DATA_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


async def run_single_iteration(config, run_id):
    """Run a single pipeline iteration and return all data."""
    results, attempts, scores = await run_pipeline(config, run_id)
    return results, attempts, scores


def compute_score_matrix(all_scores, questions, docs):
    """
    Compute score matrix: scores[doc_id][q_index] = list of scores across runs.
    Returns a dict where each key is (doc_id, q_index) -> list of 1s and 0s.
    """
    score_matrix = defaultdict(list)

    for run_scores in all_scores:
        for score_entry in run_scores:
            key = (score_entry["doc_id"], score_entry["q_index"])
            if score_entry["score"] is not None:
                score_matrix[key].append(score_entry["score"])

    return score_matrix


def compute_averages(score_matrix, questions, docs):
    """Compute various averages from score matrix."""
    results = {
        "per_doc_question": {},  # Average for each (doc, question) pair
        "per_question": {},      # Average for each question across all docs
        "per_doc": {},           # Average for each doc across all questions
        "overall": 0.0,          # Overall average
    }

    # Per doc/question averages
    all_scores = []
    for (doc_id, q_idx), scores in score_matrix.items():
        if scores:
            avg = sum(scores) / len(scores)
            results["per_doc_question"][(doc_id, q_idx)] = {
                "average": avg,
                "count": len(scores),
            }
            all_scores.extend(scores)

    # Per question averages (across all docs)
    for q_idx in range(len(questions)):
        q_scores = []
        for doc in docs:
            key = (doc.doc_id, q_idx)
            if key in score_matrix:
                q_scores.extend(score_matrix[key])
        if q_scores:
            results["per_question"][q_idx] = {
                "average": sum(q_scores) / len(q_scores),
                "count": len(q_scores),
            }

    # Per doc averages (across all questions)
    for doc in docs:
        d_scores = []
        for q_idx in range(len(questions)):
            key = (doc.doc_id, q_idx)
            if key in score_matrix:
                d_scores.extend(score_matrix[key])
        if d_scores:
            results["per_doc"][doc.doc_id] = {
                "average": sum(d_scores) / len(d_scores),
                "count": len(d_scores),
            }

    # Overall average
    if all_scores:
        results["overall"] = sum(all_scores) / len(all_scores)

    return results


async def run_benchmark(name, config_creator, docs, questions, ground_truths, num_iterations):
    """Run benchmark for given configuration."""
    print(f"\n{'='*60}", flush=True)
    print(f"Running {name} Benchmark ({num_iterations} iterations)", flush=True)
    print(f"{'='*60}", flush=True)

    all_scores = []

    for i in range(num_iterations):
        config = config_creator(docs, questions, ground_truths)
        run_id = f"{name.lower().replace(' ', '_')}_{i:03d}"

        try:
            results, attempts, scores = await run_single_iteration(config, run_id)
            primary_scores = extract_primary_scores(scores, config)
            all_scores.append(primary_scores)

            # Save detailed data for this iteration
            save_detailed_record(name, i, results, scores, questions, ground_truths)

            # Progress update every iteration
            print(f"  [{i + 1}/{num_iterations}] completed", flush=True)

        except Exception as e:
            print(f"  [{i + 1}/{num_iterations}] ERROR: {e}", flush=True)
            continue

    # Compute averages
    score_matrix = compute_score_matrix(all_scores, questions, docs)
    averages = compute_averages(score_matrix, questions, docs)

    return averages, score_matrix


def generate_markdown_report(single_results, asap_results, questions, docs):
    """Generate markdown report with comparison tables."""

    lines = [
        "# Benchmark Results: ASAP vs Single Model",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Iterations per configuration:** {NUM_ITERATIONS}",
        "",
        f"**Scoring method:** LLM-based (Gemini judges correctness)",
        "",
        "## Configuration",
        "",
        "### Single Model",
        f"- Model: `{GEMINI_MODEL}`",
        "- Doers: 1 call (temperature=0.3)",
        "- Judges: None",
        "- Final Judge: None",
        "",
        "### ASAP (Aggregation System)",
        f"- Model: `{GEMINI_MODEL}`",
        "- Doers: 10 calls (temperature=0.7)",
        "- Judges: 5 calls (temperature=0.3)",
        "- Final Judge: 1 call (temperature=0.0)",
        "",
        "---",
        "",
        "## Overall Results",
        "",
        "| Configuration | Average Success Rate |",
        "|--------------|---------------------|",
        f"| Single Model | {single_results['overall']*100:.1f}% |",
        f"| ASAP         | {asap_results['overall']*100:.1f}% |",
        f"| **Improvement** | **{(asap_results['overall'] - single_results['overall'])*100:+.1f}%** |",
        "",
        "---",
        "",
        "## Results by Document",
        "",
        "| Document | Single Model | ASAP | Improvement |",
        "|----------|-------------|------|-------------|",
    ]

    for doc in docs:
        single_doc = single_results["per_doc"].get(doc.doc_id, {}).get("average", 0)
        asap_doc = asap_results["per_doc"].get(doc.doc_id, {}).get("average", 0)
        improvement = asap_doc - single_doc
        lines.append(f"| {doc.doc_id} ({doc.filename}) | {single_doc*100:.1f}% | {asap_doc*100:.1f}% | {improvement*100:+.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## Results by Question",
        "",
        "| Q# | Question (truncated) | Single Model | ASAP | Improvement |",
        "|----|---------------------|-------------|------|-------------|",
    ])

    for q_idx, question in enumerate(questions):
        q_short = question[:50] + "..." if len(question) > 50 else question
        single_q = single_results["per_question"].get(q_idx, {}).get("average", 0)
        asap_q = asap_results["per_question"].get(q_idx, {}).get("average", 0)
        improvement = asap_q - single_q
        lines.append(f"| Q{q_idx+1} | {q_short} | {single_q*100:.1f}% | {asap_q*100:.1f}% | {improvement*100:+.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## Detailed Results by Document and Question",
        "",
        "### Single Model Results",
        "",
        "| Doc \\ Question | " + " | ".join([f"Q{i+1}" for i in range(len(questions))]) + " | Avg |",
        "|" + "---|" * (len(questions) + 2),
    ])

    for doc in docs:
        row = [doc.doc_id]
        doc_scores = []
        for q_idx in range(len(questions)):
            score = single_results["per_doc_question"].get((doc.doc_id, q_idx), {}).get("average", 0)
            row.append(f"{score*100:.0f}%")
            doc_scores.append(score)
        avg = sum(doc_scores) / len(doc_scores) if doc_scores else 0
        row.append(f"**{avg*100:.0f}%**")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "### ASAP Results",
        "",
        "| Doc \\ Question | " + " | ".join([f"Q{i+1}" for i in range(len(questions))]) + " | Avg |",
        "|" + "---|" * (len(questions) + 2),
    ])

    for doc in docs:
        row = [doc.doc_id]
        doc_scores = []
        for q_idx in range(len(questions)):
            score = asap_results["per_doc_question"].get((doc.doc_id, q_idx), {}).get("average", 0)
            row.append(f"{score*100:.0f}%")
            doc_scores.append(score)
        avg = sum(doc_scores) / len(doc_scores) if doc_scores else 0
        row.append(f"**{avg*100:.0f}%**")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "### Improvement (ASAP - Single Model)",
        "",
        "| Doc \\ Question | " + " | ".join([f"Q{i+1}" for i in range(len(questions))]) + " | Avg |",
        "|" + "---|" * (len(questions) + 2),
    ])

    for doc in docs:
        row = [doc.doc_id]
        doc_improvements = []
        for q_idx in range(len(questions)):
            single_score = single_results["per_doc_question"].get((doc.doc_id, q_idx), {}).get("average", 0)
            asap_score = asap_results["per_doc_question"].get((doc.doc_id, q_idx), {}).get("average", 0)
            improvement = asap_score - single_score
            doc_improvements.append(improvement)
            row.append(f"{improvement*100:+.0f}%")
        avg = sum(doc_improvements) / len(doc_improvements) if doc_improvements else 0
        row.append(f"**{avg*100:+.0f}%**")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "---",
        "",
        "## Summary",
        "",
    ])

    overall_improvement = asap_results['overall'] - single_results['overall']
    if overall_improvement > 0:
        lines.append(f"ASAP shows a **{overall_improvement*100:.1f}%** improvement over single model runs.")
    else:
        lines.append(f"Single model shows a **{abs(overall_improvement)*100:.1f}%** advantage over ASAP.")

    lines.extend([
        "",
        "### Key Observations",
        "",
    ])

    # Find best/worst improvements
    improvements = []
    for doc in docs:
        for q_idx in range(len(questions)):
            single_score = single_results["per_doc_question"].get((doc.doc_id, q_idx), {}).get("average", 0)
            asap_score = asap_results["per_doc_question"].get((doc.doc_id, q_idx), {}).get("average", 0)
            improvements.append((doc.doc_id, q_idx, asap_score - single_score))

    if improvements:
        improvements.sort(key=lambda x: x[2], reverse=True)
        best = improvements[0]
        worst = improvements[-1]

        lines.append(f"- **Largest improvement:** {best[0]}, Q{best[1]+1} with {best[2]*100:+.1f}%")
        lines.append(f"- **Largest regression:** {worst[0]}, Q{worst[1]+1} with {worst[2]*100:+.1f}%")

    lines.extend([
        "",
        "---",
        "",
        "## Data Files",
        "",
        f"- **Detailed responses:** `{DETAILED_DATA_FILE}` (JSONL format with all model responses)",
        "- **Summary stats:** `benchmark_raw_results.json`",
        "",
    ])

    return "\n".join(lines)


async def main():
    # Clear previous detailed data file
    if DETAILED_DATA_FILE.exists():
        DETAILED_DATA_FILE.unlink()

    print("Loading test data...", flush=True)
    docs, questions, ground_truths = load_test_data()
    print(f"Loaded {len(docs)} documents, {len(questions)} questions", flush=True)
    print(f"Detailed responses will be saved to: {DETAILED_DATA_FILE}", flush=True)

    # Run single model benchmark
    single_results, single_matrix = await run_benchmark(
        "Single Model",
        create_single_model_config,
        docs, questions, ground_truths,
        NUM_ITERATIONS
    )

    # Run ASAP benchmark
    asap_results, asap_matrix = await run_benchmark(
        "ASAP",
        create_asap_config,
        docs, questions, ground_truths,
        NUM_ITERATIONS
    )

    # Generate report
    print("\nGenerating markdown report...", flush=True)
    report = generate_markdown_report(single_results, asap_results, questions, docs)

    # Write to file
    output_path = Path("BENCHMARK_RESULTS.md")
    output_path.write_text(report)
    print(f"Report saved to: {output_path}", flush=True)

    # Also save raw aggregated data
    raw_data = {
        "config": {
            "model": GEMINI_MODEL,
            "num_iterations": NUM_ITERATIONS,
            "scoring_mode": "llm",
        },
        "single_model": {
            "overall": single_results["overall"],
            "per_doc": {k: v for k, v in single_results["per_doc"].items()},
            "per_question": {str(k): v for k, v in single_results["per_question"].items()},
            "per_doc_question": {f"{k[0]}_{k[1]}": v for k, v in single_results["per_doc_question"].items()},
        },
        "asap": {
            "overall": asap_results["overall"],
            "per_doc": {k: v for k, v in asap_results["per_doc"].items()},
            "per_question": {str(k): v for k, v in asap_results["per_question"].items()},
            "per_doc_question": {f"{k[0]}_{k[1]}": v for k, v in asap_results["per_doc_question"].items()},
        },
    }

    raw_path = Path("benchmark_raw_results.json")
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Raw aggregated data saved to: {raw_path}", flush=True)

    # Print summary
    print("\n" + "="*60, flush=True)
    print("BENCHMARK COMPLETE", flush=True)
    print("="*60, flush=True)
    print(f"Single Model Overall: {single_results['overall']*100:.1f}%", flush=True)
    print(f"ASAP Overall:         {asap_results['overall']*100:.1f}%", flush=True)
    print(f"Improvement:          {(asap_results['overall']-single_results['overall'])*100:+.1f}%", flush=True)
    print(f"\nDetailed data file: {DETAILED_DATA_FILE}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
