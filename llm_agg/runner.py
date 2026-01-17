from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from typing import Callable, Optional, Union

from .config import RunConfig, DocInfo, ModelRow
from .openrouter import call_openrouter
from .prompts import (
    DOER_DEFAULT, JUDGE_DEFAULT, FINAL_DEFAULT, SCORER_DEFAULT,
    build_doc_block, build_doer_user_message, build_judge_user_message,
    build_final_user_message, build_scorer_user_message,
)


class BudgetCounter:
    def __init__(self, cap: int):
        self._cap = cap
        self._count = 0
        self._lock = asyncio.Lock()

    async def try_increment(self) -> bool:
        async with self._lock:
            if self._count >= self._cap:
                return False
            self._count += 1
            return True

    @property
    def used(self) -> int:
        return self._count


async def _make_call(
    messages: list[dict],
    model_row: ModelRow,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    budget: BudgetCounter,
    run_id: str,
    stage: str,
    doc_id: str,
    q_index: int,
    call_index: int,
    retries: int,
    on_attempt: Optional[Callable],
) -> dict:
    for attempt in range(retries + 1):
        if not await budget.try_increment():
            record = _build_attempt_record(
                run_id, stage, doc_id, q_index, model_row.model_id, call_index, attempt,
                datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat(),
                0, "skipped_budget", None, "Budget exhausted", None, None, None, None
            )
            if on_attempt:
                on_attempt(record)
            return {"status": "skipped_budget", "text": None, "record": record}

        if attempt > 0:
            await asyncio.sleep(min(2.0, 0.25 * attempt))

        started_at = datetime.now(timezone.utc).isoformat()
        async with semaphore:
            result = await call_openrouter(
                messages, model_row.model_id, model_row.timeout_s,
                max_tokens, model_row.temperature
            )
        ended_at = datetime.now(timezone.utc).isoformat()

        record = _build_attempt_record(
            run_id, stage, doc_id, q_index, model_row.model_id, call_index, attempt,
            started_at, ended_at, result["latency_ms"], result["status"],
            result["http_status"], result["error_message"], result["request"],
            result["response_text"], result["response_json"], result["usage"]
        )
        if on_attempt:
            on_attempt(record)

        if result["status"] == "ok":
            return {"status": "ok", "text": result["response_text"], "record": record}
        if result["status"] == "timeout":
            return {"status": "timeout", "text": None, "record": record}

    return {"status": "error", "text": None, "record": record}


def _build_attempt_record(
    run_id, stage, doc_id, q_index, model_id, call_index, attempt,
    started_at, ended_at, latency_ms, status, http_status, error_message,
    request, response_text, response_json, usage
) -> dict:
    return {
        "run_id": run_id,
        "stage": stage,
        "doc_id": doc_id,
        "q_index": q_index,
        "model_id": model_id,
        "call_index": call_index,
        "attempt": attempt,
        "started_at": started_at,
        "ended_at": ended_at,
        "latency_ms": latency_ms,
        "status": status,
        "http_status": http_status,
        "error_message": error_message,
        "request": request,
        "response_text": response_text,
        "response_json": response_json,
        "usage": usage,
    }


def _get_system_prompt(custom: str, default: str) -> str:
    return custom if custom.strip() else default


async def _run_stage(
    model_rows: list[ModelRow],
    system_prompt: str,
    user_message: Union[str, list],
    semaphore: asyncio.Semaphore,
    budget: BudgetCounter,
    run_id: str,
    stage: str,
    doc_id: str,
    q_index: int,
    default_max_tokens: int,
    retries: int,
    on_attempt: Optional[Callable],
) -> tuple[list[dict], list[dict]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    tasks = []
    for model_row in model_rows:
        # Use per-model max_tokens if set, otherwise use global default
        max_tokens = model_row.max_tokens or default_max_tokens
        for call_idx in range(model_row.n_calls):
            tasks.append(_make_call(
                messages, model_row, max_tokens, semaphore, budget,
                run_id, stage, doc_id, q_index, call_idx, retries, on_attempt
            ))

    results = await asyncio.gather(*tasks)
    outputs = []
    attempts = []
    idx = 0
    for model_row in model_rows:
        for call_idx in range(model_row.n_calls):
            r = results[idx]
            attempts.append(r["record"])
            if r["status"] == "ok" and r["text"]:
                outputs.append({
                    "model_id": model_row.model_id,
                    "call_index": call_idx,
                    "text": r["text"],
                    "status": "ok",
                })
            else:
                outputs.append({
                    "model_id": model_row.model_id,
                    "call_index": call_idx,
                    "text": r.get("text"),
                    "status": r["status"],
                })
            idx += 1
    return outputs, attempts


async def _run_for_doc_question(
    config: RunConfig,
    doc: Optional[DocInfo],
    question: str,
    q_index: int,
    semaphore: asyncio.Semaphore,
    budget: BudgetCounter,
    run_id: str,
    on_attempt: Optional[Callable],
) -> tuple[dict, list[dict]]:
    doc_id = doc.doc_id if doc else "__no_doc__"
    doc_block = build_doc_block(doc.model_dump()) if doc else None

    all_attempts = []

    doer_system = _get_system_prompt(config.doer_system_prompt, DOER_DEFAULT)
    doer_user = build_doer_user_message(question, doc_block)
    doer_outputs, doer_attempts = await _run_stage(
        config.doers, doer_system, doer_user, semaphore, budget,
        run_id, "doer", doc_id, q_index, config.max_output_tokens, config.retries, on_attempt
    )
    all_attempts.extend(doer_attempts)

    valid_doers = [o for o in doer_outputs if o["status"] == "ok" and o["text"]]
    judge_outputs = []
    if config.judges:
        judge_system = _get_system_prompt(config.judge_system_prompt, JUDGE_DEFAULT)
        judge_doc_block = doc_block if config.send_doc_to_judges else None
        judge_doers = valid_doers if config.send_doer_responses_to_judges else None
        judge_user = build_judge_user_message(question, judge_doc_block, judge_doers)
        judge_outputs, judge_attempts = await _run_stage(
            config.judges, judge_system, judge_user, semaphore, budget,
            run_id, "judge", doc_id, q_index, config.max_output_tokens, config.retries, on_attempt
        )
        all_attempts.extend(judge_attempts)

    valid_judges = [o for o in judge_outputs if o["status"] == "ok" and o["text"]]
    final_outputs = []
    if config.final_judges:
        final_system = _get_system_prompt(config.final_system_prompt, FINAL_DEFAULT)
        final_doc_block = doc_block if config.send_doc_to_final_judges else None
        final_doers = valid_doers if config.send_doer_outputs_to_final_judges else None
        final_judges = valid_judges if config.send_judge_outputs_to_final_judges else None
        final_user = build_final_user_message(question, final_doc_block, final_doers, final_judges)
        final_outputs, final_attempts = await _run_stage(
            config.final_judges, final_system, final_user, semaphore, budget,
            run_id, "final", doc_id, q_index, config.max_output_tokens, config.retries, on_attempt
        )
        all_attempts.extend(final_attempts)

    primary = final_outputs if config.final_judges else doer_outputs
    result_item = {
        "doc_id": doc_id,
        "q_index": q_index,
        "question": question,
        "doers": doer_outputs,
        "judges": judge_outputs,
        "finals": final_outputs,
        "primary_outputs": primary,
    }
    return result_item, all_attempts


async def _score_response(
    response_text: str,
    ground_truth: str,
    config: RunConfig,
    semaphore: asyncio.Semaphore,
    budget: BudgetCounter,
    run_id: str,
    doc_id: str,
    q_index: int,
    stage: str,
    model_id: str,
    call_index: int,
    on_attempt: Optional[Callable],
) -> tuple[Optional[int], Optional[dict]]:
    if config.benchmark.mode == "exact":
        if config.benchmark.strip_whitespace:
            match = ground_truth.strip() == response_text.strip()
        else:
            match = ground_truth == response_text
        return 1 if match else 0, None

    scorer = config.benchmark.scorer
    if not scorer:
        return None, None

    scorer_row = ModelRow(
        model_id=scorer.model_id,
        timeout_s=scorer.timeout_s,
        n_calls=1,
        temperature=scorer.temperature,
    )
    messages = [
        {"role": "system", "content": SCORER_DEFAULT},
        {"role": "user", "content": build_scorer_user_message(ground_truth, response_text)},
    ]
    result = await _make_call(
        messages, scorer_row, config.max_output_tokens, semaphore, budget,
        run_id, "scorer", doc_id, q_index, call_index, 0, on_attempt
    )
    if result["status"] == "ok" and result["text"]:
        stripped = result["text"].strip()
        if stripped == "1":
            return 1, result["record"]
        elif stripped == "0":
            return 0, result["record"]
    return None, result.get("record")


async def _benchmark_results(
    results: list[dict],
    config: RunConfig,
    semaphore: asyncio.Semaphore,
    budget: BudgetCounter,
    run_id: str,
    on_attempt: Optional[Callable],
) -> tuple[list[dict], list[dict]]:
    if not config.benchmark.enabled:
        return [], []

    scores = []
    attempts = []
    for item in results:
        q_idx = item["q_index"]
        if q_idx >= len(config.benchmark.ground_truths):
            continue
        truth = config.benchmark.ground_truths[q_idx]

        for stage, outputs in [("doer", item["doers"]), ("judge", item["judges"]), ("final", item["finals"])]:
            for out in outputs:
                if out["status"] != "ok" or not out.get("text"):
                    continue
                score, attempt = await _score_response(
                    out["text"], truth, config, semaphore, budget, run_id,
                    item["doc_id"], q_idx, stage, out["model_id"], out["call_index"], on_attempt
                )
                scores.append({
                    "doc_id": item["doc_id"],
                    "q_index": q_idx,
                    "stage": stage,
                    "model_id": out["model_id"],
                    "call_index": out["call_index"],
                    "score": score,
                })
                if attempt:
                    attempts.append(attempt)
    return scores, attempts


async def run_pipeline(
    config: RunConfig,
    run_id: str,
    on_attempt: Optional[Callable] = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    semaphore = asyncio.Semaphore(config.max_concurrency)
    budget = BudgetCounter(config.cap_total_calls)

    docs_iter = config.docs if config.docs else [None]
    tasks = []
    for doc in docs_iter:
        for q_idx, question in enumerate(config.questions):
            tasks.append(_run_for_doc_question(
                config, doc, question, q_idx, semaphore, budget, run_id, on_attempt
            ))

    task_results = await asyncio.gather(*tasks)
    results = []
    all_attempts = []
    for result_item, attempts in task_results:
        results.append(result_item)
        all_attempts.extend(attempts)

    scores, scorer_attempts = await _benchmark_results(
        results, config, semaphore, budget, run_id, on_attempt
    )
    all_attempts.extend(scorer_attempts)

    return results, all_attempts, scores
