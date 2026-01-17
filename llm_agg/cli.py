import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import statistics
import zlib
from pathlib import Path
from typing import Optional

from .config import RunConfig, DocInfo, BenchmarkConfig, MAX_FILE_SIZE
from .io import (
    generate_run_id, write_resolved_config, append_call_log,
    write_results, write_stats, write_accuracy,
)
from .runner import run_pipeline
from .stats import compute_stats


def _get_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".txt": "text/plain",
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    return mime_map.get(ext, mimetypes.guess_type(filename)[0] or "application/octet-stream")


def _extract_pdf_content(filepath: str) -> tuple[str, str]:
    """Extract text and first page image from PDF using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(filepath)

    # Extract text from all pages
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    full_text = "\n\n".join(text_parts)

    # Render first page as image
    first_page = doc[0]
    # Use 2x zoom for better quality
    mat = fitz.Matrix(2, 2)
    pix = first_page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    img_b64 = base64.b64encode(img_bytes).decode("ascii")

    doc.close()
    return full_text, img_b64


def _load_doc(filepath: str, doc_id: str) -> DocInfo:
    path = Path(filepath)
    data = path.read_bytes()
    if len(data) > MAX_FILE_SIZE:
        raise ValueError(f"File {filepath} exceeds {MAX_FILE_SIZE} bytes limit")

    mime = _get_mime(path.name)

    # Handle PDFs specially - extract text and render first page
    if mime == "application/pdf":
        text, img_b64 = _extract_pdf_content(filepath)
        return DocInfo(
            doc_id=doc_id,
            filename=path.name,
            mime=mime,
            encoding="pdf",
            content=text,
            pdf_image=img_b64,
        )

    try:
        text = data.decode("utf-8")
        return DocInfo(
            doc_id=doc_id,
            filename=path.name,
            mime=mime,
            encoding="utf-8",
            content=text,
        )
    except UnicodeDecodeError:
        if mime.startswith("image/"):
            return DocInfo(
                doc_id=doc_id,
                filename=path.name,
                mime=mime,
                encoding="image",
                content=base64.b64encode(data).decode("ascii"),
            )
        compressed = zlib.compress(data, level=9)
        return DocInfo(
            doc_id=doc_id,
            filename=path.name,
            mime=mime,
            encoding="base64+zlib",
            content=base64.b64encode(compressed).decode("ascii"),
        )


def _load_docs_from_folder(folder: str) -> list[DocInfo]:
    path = Path(folder)
    docs = []
    for f in sorted(path.iterdir()):
        if f.is_file() and not f.name.startswith(".") and f.name != "Questions.md":
            docs.append(_load_doc(str(f), f.stem))
    return docs


def _parse_questions_md(filepath: str) -> tuple[list[str], list[str]]:
    text = Path(filepath).read_text()
    questions = []
    answers = []

    q_pattern = re.compile(r"^Q(\d+):\s*(.+)$", re.MULTILINE)
    a_pattern = re.compile(r"^A(\d+):\s*(.+)$", re.MULTILINE)

    q_matches = {int(m.group(1)): m.group(2).strip() for m in q_pattern.finditer(text)}
    a_matches = {int(m.group(1)): m.group(2).strip() for m in a_pattern.finditer(text)}

    for i in sorted(q_matches.keys()):
        questions.append(q_matches[i])
        answers.append(a_matches.get(i, ""))

    return questions, answers


def _compute_accuracy(scores: list[dict]) -> dict:
    buckets = {}
    for s in scores:
        if s["score"] is None:
            continue
        key = f"{s['stage']}:{s['model_id']}"
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(s["score"])

    result = {}
    for key, vals in buckets.items():
        result[key] = {
            "n_scored": len(vals),
            "accuracy": sum(vals) / len(vals) if vals else 0.0,
        }
    return result


def _run_single(config: RunConfig, out_dir: str) -> tuple[dict, list[dict], list[dict]]:
    run_id = generate_run_id()
    write_resolved_config(out_dir, config)

    def on_attempt(record):
        append_call_log(out_dir, record)

    results, attempts, scores = asyncio.run(run_pipeline(config, run_id, on_attempt))

    write_results(out_dir, {"run_id": run_id, "items": results})

    stats = compute_stats(attempts)
    write_stats(out_dir, stats)

    if config.benchmark.enabled and scores:
        accuracy = _compute_accuracy(scores)
        write_accuracy(out_dir, accuracy)

    return stats, attempts, scores


def _aggregate_stats(all_stats: list[dict]) -> dict:
    def aggregate_bucket(key_path: list[str]) -> dict:
        buckets = []
        for s in all_stats:
            obj = s
            for k in key_path:
                obj = obj.get(k, {})
            if obj:
                buckets.append(obj)

        if not buckets:
            return {}

        numeric_keys = [k for k in buckets[0].keys() if isinstance(buckets[0][k], (int, float))]
        result = {}
        for k in numeric_keys:
            vals = [b[k] for b in buckets if k in b]
            if vals:
                result[f"{k}_mean"] = statistics.mean(vals)
                result[f"{k}_median"] = statistics.median(vals)
        return result

    agg = {"overall": aggregate_bucket(["overall"])}

    all_stages = set()
    for s in all_stats:
        all_stages.update(s.get("per_stage", {}).keys())

    per_stage = {}
    for stage in all_stages:
        per_stage[stage] = aggregate_bucket(["per_stage", stage])
    if per_stage:
        agg["per_stage"] = per_stage

    all_keys = set()
    for s in all_stats:
        all_keys.update(s.get("per_stage_model", {}).keys())

    per_stage_model = {}
    for key in all_keys:
        per_stage_model[key] = aggregate_bucket(["per_stage_model", key])
    if per_stage_model:
        agg["per_stage_model"] = per_stage_model

    return agg


def _aggregate_accuracy(all_accuracy: list[dict]) -> dict:
    all_keys = set()
    for a in all_accuracy:
        all_keys.update(a.keys())

    result = {}
    for key in all_keys:
        n_vals = [a[key]["n_scored"] for a in all_accuracy if key in a]
        acc_vals = [a[key]["accuracy"] for a in all_accuracy if key in a]
        if acc_vals:
            result[key] = {
                "n_scored_mean": statistics.mean(n_vals),
                "n_scored_median": statistics.median(n_vals),
                "accuracy_mean": statistics.mean(acc_vals),
                "accuracy_median": statistics.median(acc_vals),
            }
    return result


def cmd_run(args):
    with open(args.config) as f:
        config_data = json.load(f)

    config = RunConfig(**config_data)
    _run_single(config, args.out)
    print(f"Run complete. Output: {args.out}")


def cmd_bench(args):
    with open(args.config) as f:
        config_data = json.load(f)

    if args.dataset:
        dataset_path = Path(args.dataset)
        if dataset_path.is_dir():
            docs = _load_docs_from_folder(str(dataset_path))
            questions_file = dataset_path / "Questions.md"
            if questions_file.exists():
                questions, answers = _parse_questions_md(str(questions_file))
                config_data["questions"] = questions
                config_data["docs"] = [d.model_dump() for d in docs]
                if answers and any(answers):
                    if "benchmark" not in config_data:
                        config_data["benchmark"] = {}
                    config_data["benchmark"]["enabled"] = True
                    config_data["benchmark"]["ground_truths"] = answers
        else:
            with open(args.dataset) as f:
                dataset = json.load(f)
            if "questions" in dataset:
                config_data["questions"] = dataset["questions"]
            if "docs" in dataset:
                config_data["docs"] = dataset["docs"]
            if "ground_truths" in dataset:
                if "benchmark" not in config_data:
                    config_data["benchmark"] = {}
                config_data["benchmark"]["enabled"] = True
                config_data["benchmark"]["ground_truths"] = dataset["ground_truths"]

    all_stats = []
    all_accuracy = []

    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    for i in range(args.repeat):
        run_dir = out_base / f"run_{i:03d}"
        config = RunConfig(**config_data)
        stats, attempts, scores = _run_single(config, str(run_dir))
        all_stats.append(stats)
        if config.benchmark.enabled and scores:
            all_accuracy.append(_compute_accuracy(scores))
        print(f"Run {i+1}/{args.repeat} complete")

    agg_stats = _aggregate_stats(all_stats)
    with open(out_base / "aggregate_stats.json", "w") as f:
        json.dump(agg_stats, f, indent=2)

    _write_aggregate_csv(out_base / "aggregate_stats.csv", agg_stats)

    if all_accuracy:
        agg_acc = _aggregate_accuracy(all_accuracy)
        with open(out_base / "aggregate_accuracy.json", "w") as f:
            json.dump(agg_acc, f, indent=2)
        _write_aggregate_accuracy_csv(out_base / "aggregate_accuracy.csv", agg_acc)

    print(f"Benchmark complete. Output: {args.out}")


def _write_aggregate_csv(filepath: Path, agg: dict):
    import csv
    rows = []
    if "overall" in agg:
        rows.append({"bucket": "overall", **agg["overall"]})
    for stage, data in agg.get("per_stage", {}).items():
        rows.append({"bucket": f"stage:{stage}", **data})
    for key, data in agg.get("per_stage_model", {}).items():
        rows.append({"bucket": key, **data})

    if rows:
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def _write_aggregate_accuracy_csv(filepath: Path, agg: dict):
    import csv
    rows = [{"bucket": k, **v} for k, v in agg.items()]
    if rows:
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(prog="llm_agg")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Single run")
    run_parser.add_argument("--config", required=True, help="Config JSON file")
    run_parser.add_argument("--out", required=True, help="Output directory")
    run_parser.set_defaults(func=cmd_run)

    bench_parser = subparsers.add_parser("bench", help="Benchmark loop")
    bench_parser.add_argument("--config", required=True, help="Config JSON file")
    bench_parser.add_argument("--dataset", help="Dataset JSON file or folder path")
    bench_parser.add_argument("--repeat", type=int, default=20, help="Number of runs")
    bench_parser.add_argument("--out", required=True, help="Output directory")
    bench_parser.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
