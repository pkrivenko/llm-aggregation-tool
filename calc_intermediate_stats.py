#!/usr/bin/env python3
"""Calculate intermediate benchmark stats from current data."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DETAILED_DATA_FILE = Path("benchmark_detailed_data.jsonl")

def load_data():
    """Load all records from JSONL file."""
    records = []
    with open(DETAILED_DATA_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def compute_stats(records):
    """Compute statistics from records."""
    # Separate by benchmark type
    single_records = [r for r in records if r["benchmark_type"] == "Single Model"]
    asap_records = [r for r in records if r["benchmark_type"] == "ASAP"]

    # Score matrices: (doc_id, q_index) -> list of scores
    single_scores = defaultdict(list)
    asap_scores = defaultdict(list)

    # Extract scores
    for record in single_records:
        for score in record["scores"]:
            if score["score"] is not None and score["stage"] == "doer":
                key = (score["doc_id"], score["q_index"])
                single_scores[key].append(score["score"])

    for record in asap_records:
        for score in record["scores"]:
            if score["score"] is not None and score["stage"] == "final":
                key = (score["doc_id"], score["q_index"])
                asap_scores[key].append(score["score"])

    return single_scores, asap_scores, len(single_records), len(asap_records)

def compute_averages(score_dict):
    """Compute averages from score dictionary."""
    # Per doc/question
    per_dq = {}
    for key, scores in score_dict.items():
        if scores:
            per_dq[key] = sum(scores) / len(scores)

    # Per question (across docs)
    per_q = defaultdict(list)
    for (doc_id, q_idx), scores in score_dict.items():
        per_q[q_idx].extend(scores)
    per_q_avg = {q: sum(s)/len(s) for q, s in per_q.items() if s}

    # Per doc (across questions)
    per_d = defaultdict(list)
    for (doc_id, q_idx), scores in score_dict.items():
        per_d[doc_id].extend(scores)
    per_d_avg = {d: sum(s)/len(s) for d, s in per_d.items() if s}

    # Overall
    all_scores = []
    for scores in score_dict.values():
        all_scores.extend(scores)
    overall = sum(all_scores) / len(all_scores) if all_scores else 0

    return per_dq, per_q_avg, per_d_avg, overall

def generate_report(single_scores, asap_scores, n_single, n_asap):
    """Generate markdown report."""
    single_dq, single_q, single_d, single_overall = compute_averages(single_scores)
    asap_dq, asap_q, asap_d, asap_overall = compute_averages(asap_scores)

    # Get all docs and questions
    all_docs = sorted(set(k[0] for k in single_scores.keys()) | set(k[0] for k in asap_scores.keys()))
    all_qs = sorted(set(k[1] for k in single_scores.keys()) | set(k[1] for k in asap_scores.keys()))

    lines = [
        "# Intermediate Benchmark Results: ASAP vs Single Model",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Iterations completed:**",
        f"- Single Model: {n_single}/100",
        f"- ASAP: {n_asap}/100",
        "",
        "---",
        "",
        "## Configuration Parameters",
        "",
        "### Model Settings",
        "",
        "| Parameter | Single Model | ASAP |",
        "|-----------|-------------|------|",
        "| **Model** | `google/gemini-2.0-flash-001` | `google/gemini-2.0-flash-001` |",
        "",
        "### Doer Stage",
        "",
        "| Parameter | Single Model | ASAP |",
        "|-----------|-------------|------|",
        "| Number of calls | 1 | 10 |",
        "| Temperature | 0.3 | 0.7 |",
        "| Timeout (sec) | 60 | 60 |",
        "",
        "### Judge Stage",
        "",
        "| Parameter | Single Model | ASAP |",
        "|-----------|-------------|------|",
        "| Enabled | No | Yes |",
        "| Number of calls | - | 5 |",
        "| Temperature | - | 0.3 |",
        "| Timeout (sec) | - | 60 |",
        "| Send doc to judges | - | Yes |",
        "| Send doer responses | - | Yes |",
        "",
        "### Final Judge Stage",
        "",
        "| Parameter | Single Model | ASAP |",
        "|-----------|-------------|------|",
        "| Enabled | No | Yes |",
        "| Number of calls | - | 1 |",
        "| Temperature | - | 0.0 |",
        "| Timeout (sec) | - | 60 |",
        "| Send doc to final | - | Yes |",
        "| Send doer outputs | - | Yes |",
        "| Send judge outputs | - | Yes |",
        "",
        "### Global Settings (Same for Both)",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Max output tokens | 300 |",
        "| Retries on failure | 1 |",
        "| Max concurrency | 10 |",
        "| Cap total calls | 500 (Single) / 2000 (ASAP) |",
        "",
        "### Scoring Settings",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Scoring mode | LLM-based (not exact match) |",
        "| Scorer model | `google/gemini-2.0-flash-001` |",
        "| Scorer temperature | 0.0 |",
        "| Scorer timeout | 30 sec |",
        "| Strip whitespace | Yes |",
        "",
        "### Test Data",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Documents | doc1.txt, doc2.pdf, doc3.jpg |",
        "| Questions | 11 (from Questions.md) |",
        "| Ground truths | Provided for all questions |",
        "",
        "---",
        "",
        "## Overall Results",
        "",
        "| Configuration | Iterations | Average Success Rate |",
        "|--------------|------------|---------------------|",
        f"| Single Model | {n_single} | **{single_overall*100:.1f}%** |",
        f"| ASAP         | {n_asap} | **{asap_overall*100:.1f}%** |",
        f"| **Difference** | - | **{(asap_overall - single_overall)*100:+.1f}%** |",
        "",
        "---",
        "",
        "## Results by Document",
        "",
        "| Document | Single Model | ASAP | Difference |",
        "|----------|-------------|------|------------|",
    ]

    for doc in all_docs:
        s = single_d.get(doc, 0)
        a = asap_d.get(doc, 0)
        lines.append(f"| {doc} | {s*100:.1f}% | {a*100:.1f}% | {(a-s)*100:+.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## Results by Question",
        "",
        "| Q# | Single Model | ASAP | Difference |",
        "|----|-------------|------|------------|",
    ])

    for q in all_qs:
        s = single_q.get(q, 0)
        a = asap_q.get(q, 0)
        lines.append(f"| Q{q+1} | {s*100:.1f}% | {a*100:.1f}% | {(a-s)*100:+.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## Detailed Results Matrix",
        "",
        "### Single Model Success Rates",
        "",
        "| Doc \\ Q | " + " | ".join([f"Q{q+1}" for q in all_qs]) + " | Avg |",
        "|" + "---|" * (len(all_qs) + 2),
    ])

    for doc in all_docs:
        row = [doc]
        doc_scores = []
        for q in all_qs:
            s = single_dq.get((doc, q), 0)
            row.append(f"{s*100:.0f}%")
            doc_scores.append(s)
        avg = sum(doc_scores)/len(doc_scores) if doc_scores else 0
        row.append(f"**{avg*100:.0f}%**")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "### ASAP Success Rates",
        "",
        "| Doc \\ Q | " + " | ".join([f"Q{q+1}" for q in all_qs]) + " | Avg |",
        "|" + "---|" * (len(all_qs) + 2),
    ])

    for doc in all_docs:
        row = [doc]
        doc_scores = []
        for q in all_qs:
            s = asap_dq.get((doc, q), 0)
            row.append(f"{s*100:.0f}%")
            doc_scores.append(s)
        avg = sum(doc_scores)/len(doc_scores) if doc_scores else 0
        row.append(f"**{avg*100:.0f}%**")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "### Improvement (ASAP - Single)",
        "",
        "| Doc \\ Q | " + " | ".join([f"Q{q+1}" for q in all_qs]) + " | Avg |",
        "|" + "---|" * (len(all_qs) + 2),
    ])

    for doc in all_docs:
        row = [doc]
        doc_diffs = []
        for q in all_qs:
            s = single_dq.get((doc, q), 0)
            a = asap_dq.get((doc, q), 0)
            diff = a - s
            doc_diffs.append(diff)
            row.append(f"{diff*100:+.0f}%")
        avg = sum(doc_diffs)/len(doc_diffs) if doc_diffs else 0
        row.append(f"**{avg*100:+.0f}%**")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "---",
        "",
        "## Summary",
        "",
    ])

    if asap_overall > single_overall:
        lines.append(f"Based on current data, ASAP shows **{(asap_overall-single_overall)*100:.1f}%** improvement over single model.")
    else:
        lines.append(f"Based on current data, single model shows **{(single_overall-asap_overall)*100:.1f}%** advantage over ASAP.")

    lines.append("")
    lines.append("*Note: These are intermediate results. Final results will be more accurate with all 100 iterations.*")

    return "\n".join(lines)

def main():
    records = load_data()
    single_scores, asap_scores, n_single, n_asap = compute_stats(records)
    report = generate_report(single_scores, asap_scores, n_single, n_asap)

    output_path = Path("intermediate_benchmark.md")
    output_path.write_text(report)
    print(f"Intermediate report saved to: {output_path}")
    print(f"Single Model: {n_single} iterations")
    print(f"ASAP: {n_asap} iterations")

if __name__ == "__main__":
    main()
