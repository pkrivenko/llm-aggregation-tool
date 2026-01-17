#!/usr/bin/env python3
"""Calculate stats for focused benchmark (doc2, Q10-Q11)."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DETAILED_DATA_FILE = Path("benchmark_focused_data.jsonl")

def load_data():
    records = []
    with open(DETAILED_DATA_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def compute_stats(records):
    single_records = [r for r in records if r["benchmark_type"] == "Single Model"]
    asap_records = [r for r in records if r["benchmark_type"] == "ASAP"]

    # Score matrices: q_index -> list of scores
    single_scores = defaultdict(list)
    asap_scores = defaultdict(list)

    for record in single_records:
        for score in record["scores"]:
            if score["score"] is not None and score["stage"] == "doer":
                single_scores[score["q_index"]].append(score["score"])

    for record in asap_records:
        for score in record["scores"]:
            if score["score"] is not None and score["stage"] == "final":
                asap_scores[score["q_index"]].append(score["score"])

    return single_scores, asap_scores, len(single_records), len(asap_records)

def generate_report(single_scores, asap_scores, n_single, n_asap):
    # Compute averages
    single_q_avg = {q: sum(s)/len(s) for q, s in single_scores.items() if s}
    asap_q_avg = {q: sum(s)/len(s) for q, s in asap_scores.items() if s}

    single_overall = sum(sum(s) for s in single_scores.values()) / sum(len(s) for s in single_scores.values()) if single_scores else 0
    asap_overall = sum(sum(s) for s in asap_scores.values()) / sum(len(s) for s in asap_scores.values()) if asap_scores else 0

    lines = [
        "# Focused Benchmark Results: ASAP vs Single Model",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Test Configuration",
        "",
        "| Parameter | Single Model | ASAP |",
        "|-----------|-------------|------|",
        "| **Model** | `google/gemini-2.0-flash-001` | `google/gemini-2.0-flash-001` |",
        "| **Document** | doc2.pdf only | doc2.pdf only |",
        "| **Questions** | Q10, Q11 only | Q10, Q11 only |",
        "| **Doer calls** | 1 | 20 |",
        "| **Doer temperature** | 0.0 | 0.7 |",
        "| **Judge calls** | - | 7 |",
        "| **Judge temperature** | - | 0.3 |",
        "| **Final judge calls** | - | 1 |",
        "| **Final temperature** | - | 0.0 |",
        "| **Max concurrency** | 1000 | 1000 |",
        "",
        f"**Iterations:** {n_single} Single Model, {n_asap} ASAP",
        "",
        "---",
        "",
        "## Overall Results",
        "",
        "| Configuration | Average Success Rate |",
        "|--------------|---------------------|",
        f"| Single Model (temp=0) | **{single_overall*100:.1f}%** |",
        f"| ASAP (20 doers, 7 judges) | **{asap_overall*100:.1f}%** |",
        f"| **Difference** | **{(asap_overall - single_overall)*100:+.1f}%** |",
        "",
        "---",
        "",
        "## Results by Question",
        "",
        "| Question | Single Model | ASAP | Difference |",
        "|----------|-------------|------|------------|",
    ]

    # Map q_index back to original question numbers (Q10=index 0, Q11=index 1 in this test)
    q_labels = {0: "Q10", 1: "Q11"}
    for q_idx in sorted(set(single_scores.keys()) | set(asap_scores.keys())):
        s = single_q_avg.get(q_idx, 0)
        a = asap_q_avg.get(q_idx, 0)
        label = q_labels.get(q_idx, f"Q{q_idx}")
        lines.append(f"| {label} | {s*100:.1f}% | {a*100:.1f}% | {(a-s)*100:+.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## Summary",
        "",
    ])

    if asap_overall > single_overall:
        lines.append(f"ASAP with 20 doers and 7 judges shows **{(asap_overall-single_overall)*100:.1f}%** improvement over single model (temp=0).")
    elif single_overall > asap_overall:
        lines.append(f"Single model (temp=0) shows **{(single_overall-asap_overall)*100:.1f}%** advantage over ASAP.")
    else:
        lines.append("Both configurations performed equally.")

    return "\n".join(lines)

def main():
    if not DETAILED_DATA_FILE.exists():
        print(f"No data file found: {DETAILED_DATA_FILE}")
        return

    records = load_data()
    single_scores, asap_scores, n_single, n_asap = compute_stats(records)
    report = generate_report(single_scores, asap_scores, n_single, n_asap)

    output_path = Path("BENCHMARK_FOCUSED_RESULTS.md")
    output_path.write_text(report)
    print(f"Report saved to: {output_path}")
    print(f"Single Model: {n_single} iterations")
    print(f"ASAP: {n_asap} iterations")

if __name__ == "__main__":
    main()
