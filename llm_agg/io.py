import csv
import json
import random
import string
from datetime import datetime
from pathlib import Path


def generate_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{ts}_{rand}"


def _ensure_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_resolved_config(out_dir: str, config) -> None:
    p = _ensure_dir(out_dir)
    data = config.model_dump() if hasattr(config, "model_dump") else dict(config)
    with open(p / "resolved_config.json", "w") as f:
        json.dump(data, f, indent=2)


def append_call_log(out_dir: str, attempt_record: dict) -> None:
    p = _ensure_dir(out_dir)
    with open(p / "call_logs.jsonl", "a") as f:
        f.write(json.dumps(attempt_record) + "\n")


def write_results(out_dir: str, results: dict) -> None:
    p = _ensure_dir(out_dir)
    with open(p / "results.json", "w") as f:
        json.dump(results, f, indent=2)


def write_stats(out_dir: str, stats_dict: dict) -> None:
    p = _ensure_dir(out_dir)
    with open(p / "stats.json", "w") as f:
        json.dump(stats_dict, f, indent=2)

    rows = []
    if "overall" in stats_dict:
        rows.append({"bucket": "overall", **stats_dict["overall"]})
    for stage, data in stats_dict.get("per_stage", {}).items():
        rows.append({"bucket": f"stage:{stage}", **data})
    for key, data in stats_dict.get("per_stage_model", {}).items():
        rows.append({"bucket": key, **data})

    if rows:
        with open(p / "stats.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def write_accuracy(out_dir: str, accuracy_dict: dict) -> None:
    p = _ensure_dir(out_dir)
    with open(p / "accuracy.json", "w") as f:
        json.dump(accuracy_dict, f, indent=2)

    rows = []
    for key, data in accuracy_dict.items():
        rows.append({"bucket": key, **data})

    if rows:
        with open(p / "accuracy.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
