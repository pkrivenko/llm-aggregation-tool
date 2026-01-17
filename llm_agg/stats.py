from typing import Any


def compute_stats(attempts: list) -> dict:
    def make_bucket(items: list) -> dict:
        ok = [a for a in items if a.get("status") == "ok"]
        timeout = [a for a in items if a.get("status") == "timeout"]
        error = [a for a in items if a.get("status") == "error"]
        skipped = [a for a in items if a.get("status") == "skipped_budget"]

        denom = len(ok) + len(timeout) + len(error)

        ok_latencies = [a["latency_ms"] for a in ok if a.get("latency_ms") is not None]

        costs = [(a.get("usage") or {}).get("cost_usd", 0) or 0 for a in items]
        prompt_tokens = sum((a.get("usage") or {}).get("prompt_tokens", 0) or 0 for a in items)
        completion_tokens = sum((a.get("usage") or {}).get("completion_tokens", 0) or 0 for a in items)
        total_tokens = sum((a.get("usage") or {}).get("total_tokens", 0) or 0 for a in items)

        return {
            "attempts_total": len(items),
            "calls_ok": len(ok),
            "calls_timeout": len(timeout),
            "calls_error": len(error),
            "calls_skipped_budget": len(skipped),
            "valid_rate": len(ok) / denom if denom > 0 else 0.0,
            "timeout_rate": len(timeout) / denom if denom > 0 else 0.0,
            "error_rate": len(error) / denom if denom > 0 else 0.0,
            "avg_latency_ms_ok": sum(ok_latencies) / len(ok_latencies) if ok_latencies else 0.0,
            "sum_cost_usd": sum(costs),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    overall = make_bucket(attempts)

    per_stage = {}
    stages = {"doer", "judge", "final", "scorer"}
    for stage in stages:
        stage_attempts = [a for a in attempts if a.get("stage") == stage]
        if stage_attempts:
            per_stage[stage] = make_bucket(stage_attempts)

    per_stage_model = {}
    for a in attempts:
        stage = a.get("stage")
        model_id = a.get("model_id")
        if stage and model_id:
            key = f"{stage}:{model_id}"
            if key not in per_stage_model:
                per_stage_model[key] = []
            per_stage_model[key].append(a)

    per_stage_model = {k: make_bucket(v) for k, v in per_stage_model.items()}

    return {
        "overall": overall,
        "per_stage": per_stage,
        "per_stage_model": per_stage_model,
    }
