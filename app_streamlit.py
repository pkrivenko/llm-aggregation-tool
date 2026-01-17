import asyncio
import base64
import json
import mimetypes
import zlib
from pathlib import Path

import streamlit as st

from llm_agg.config import RunConfig, ModelRow, DocInfo, BenchmarkConfig, ScorerConfig, MAX_FILE_SIZE
from llm_agg.runner import run_pipeline
from llm_agg.stats import compute_stats
from llm_agg.io import (
    generate_run_id, write_resolved_config, append_call_log,
    write_results, write_stats, write_accuracy
)

st.set_page_config(page_title="LLM Aggregation Tool", layout="wide")

SETTINGS_PATH = Path(__file__).parent / "settings.json"


def load_settings():
    """Load all settings from settings.json."""
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    # Return minimal defaults if settings.json doesn't exist
    return {
        "models": [],
        "system_prompts": {"doer": "", "judge": "", "final": "", "scorer": ""},
        "default_model": "",
        "model_defaults": {"timeout_s": 30.0, "n_calls": 1, "temperature": None},
        "global_controls": {"cap_total_calls": 100, "max_output_tokens": 1000, "retries": 0, "max_concurrency": 1000, "debug_mode": False},
        "pipeline_options": {"send_doc_to_judges": False, "send_doc_to_final_judges": False, "send_doer_responses_to_judges": True, "send_doer_outputs_to_final_judges": True, "send_judge_outputs_to_final_judges": True},
        "benchmark_defaults": {"enabled": False, "mode": "exact", "strip_whitespace": True, "scorer_model": None, "scorer_timeout_s": 30.0, "scorer_temperature": 0.0},
        "ui": {"max_files": 10, "max_file_size_kb": 200, "max_model_rows": 10}
    }


# Load settings once at startup
SETTINGS = load_settings()
MODELS_CATALOG = SETTINGS.get("models", [])
SYSTEM_PROMPTS = SETTINGS.get("system_prompts", {})
MODEL_DEFAULTS = SETTINGS.get("model_defaults", {})
GLOBAL_CONTROLS = SETTINGS.get("global_controls", {})
PIPELINE_OPTIONS = SETTINGS.get("pipeline_options", {})
BENCHMARK_DEFAULTS = SETTINGS.get("benchmark_defaults", {})
UI_SETTINGS = SETTINGS.get("ui", {})

MAX_FILES = UI_SETTINGS.get("max_files", 10)
MAX_MODEL_ROWS = UI_SETTINGS.get("max_model_rows", 10)


def init_session_state():
    default_model = SETTINGS.get("default_model", "")
    timeout = MODEL_DEFAULTS.get("timeout_s", 30.0)
    n_calls = MODEL_DEFAULTS.get("n_calls", 1)

    if "doer_rows" not in st.session_state:
        st.session_state.doer_rows = [{"model_id": default_model, "timeout_s": timeout, "n_calls": n_calls, "use_temp": False, "temperature": 1.0}]
    if "judge_rows" not in st.session_state:
        st.session_state.judge_rows = []
    if "final_rows" not in st.session_state:
        st.session_state.final_rows = []
    if "run_results" not in st.session_state:
        st.session_state.run_results = None
    if "run_stats" not in st.session_state:
        st.session_state.run_stats = None
    if "run_accuracy" not in st.session_state:
        st.session_state.run_accuracy = None
    if "run_config" not in st.session_state:
        st.session_state.run_config = None
    if "run_logs" not in st.session_state:
        st.session_state.run_logs = []
    if "run_id" not in st.session_state:
        st.session_state.run_id = None


def parse_questions(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def extract_pdf_content(content_bytes: bytes) -> tuple[str, str]:
    """Extract text and first page image from PDF using PyMuPDF."""
    import fitz  # PyMuPDF
    import io

    doc = fitz.open(stream=content_bytes, filetype="pdf")

    # Extract text from all pages
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    full_text = "\n\n".join(text_parts)

    # Render first page as image
    first_page = doc[0]
    mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
    pix = first_page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    img_b64 = base64.b64encode(img_bytes).decode("ascii")

    doc.close()
    return full_text, img_b64


def encode_file(uploaded_file) -> DocInfo:
    content_bytes = uploaded_file.read()
    filename = uploaded_file.name
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    # Handle PDFs specially - extract text and render first page
    if mime == "application/pdf":
        text, img_b64 = extract_pdf_content(content_bytes)
        return DocInfo(
            doc_id=filename,
            filename=filename,
            mime=mime,
            encoding="pdf",
            content=text,
            pdf_image=img_b64
        )

    try:
        text = content_bytes.decode("utf-8", errors="strict")
        return DocInfo(
            doc_id=filename,
            filename=filename,
            mime=mime,
            encoding="utf-8",
            content=text
        )
    except UnicodeDecodeError:
        if mime.startswith("image/"):
            return DocInfo(
                doc_id=filename,
                filename=filename,
                mime=mime,
                encoding="image",
                content=base64.b64encode(content_bytes).decode("ascii")
            )
        compressed = zlib.compress(content_bytes, level=9)
        return DocInfo(
            doc_id=filename,
            filename=filename,
            mime=mime,
            encoding="base64+zlib",
            content=base64.b64encode(compressed).decode("ascii")
        )


def combine_documents(docs: list[DocInfo]) -> DocInfo:
    """Combine multiple documents into a single DocInfo."""
    combined_parts = []

    for doc in docs:
        if doc.encoding == "utf-8":
            combined_parts.append(f"=== {doc.filename} ===\n{doc.content}")
        elif doc.encoding == "pdf":
            combined_parts.append(f"=== {doc.filename} (PDF) ===\n{doc.content}")
        elif doc.encoding == "image":
            combined_parts.append(f"=== {doc.filename} (Image) ===\n[Image content - see original file]")
        else:
            combined_parts.append(f"=== {doc.filename} ===\n[Binary content]")

    combined_content = "\n\n".join(combined_parts)
    filenames = [d.filename for d in docs]
    combined_name = f"Combined: {', '.join(filenames[:3])}{'...' if len(filenames) > 3 else ''}"

    return DocInfo(
        doc_id="__combined__",
        filename=combined_name,
        mime="text/plain",
        encoding="utf-8",
        content=combined_content,
        pdf_image=None
    )


def render_model_rows_editor(key_prefix: str, rows_key: str, catalog: list[dict]):
    rows = st.session_state[rows_key]
    to_remove = None

    # Header row for alignment
    if rows:
        header_cols = st.columns([3, 1, 1, 0.5, 1, 0.5])
        with header_cols[0]:
            st.caption("Model")
        with header_cols[1]:
            st.caption("Timeout(s)")
        with header_cols[2]:
            st.caption("Calls")
        with header_cols[3]:
            st.caption("Temp")
        with header_cols[4]:
            st.caption("T value")
        with header_cols[5]:
            st.caption("")

    for i, row in enumerate(rows):
        cols = st.columns([3, 1, 1, 0.5, 1, 0.5])
        with cols[0]:
            options = [m["id"] for m in catalog]
            current = row.get("model_id", "")
            if current and current not in options:
                options = [current] + options
            idx = options.index(current) if current in options else 0
            selected = st.selectbox(
                "Model", options, index=idx if options else 0,
                key=f"{key_prefix}_model_{i}", label_visibility="collapsed"
            )
            custom = st.text_input("Custom model ID", value="" if selected else current,
                key=f"{key_prefix}_custom_{i}", label_visibility="collapsed", placeholder="Or enter custom ID")
            row["model_id"] = custom if custom else selected
        with cols[1]:
            row["timeout_s"] = st.number_input("Timeout", value=row.get("timeout_s", 30.0), min_value=0.1,
                key=f"{key_prefix}_timeout_{i}", label_visibility="collapsed")
        with cols[2]:
            row["n_calls"] = st.number_input("Calls", value=row.get("n_calls", 1), min_value=1,
                key=f"{key_prefix}_ncalls_{i}", label_visibility="collapsed")
        with cols[3]:
            row["use_temp"] = st.checkbox("T", value=row.get("use_temp", False),
                key=f"{key_prefix}_usetemp_{i}", label_visibility="collapsed")
        with cols[4]:
            if row["use_temp"]:
                row["temperature"] = st.number_input("T", value=row.get("temperature", 1.0), min_value=0.0, max_value=2.0,
                    key=f"{key_prefix}_temp_{i}", label_visibility="collapsed")
        with cols[5]:
            st.button("✕", key=f"{key_prefix}_remove_{i}", on_click=lambda idx=i: rows.pop(idx) or st.rerun())

    if len(rows) < MAX_MODEL_ROWS:
        if st.button("+ Add model", key=f"{key_prefix}_add"):
            rows.append({"model_id": "", "timeout_s": 30.0, "n_calls": 1, "use_temp": False, "temperature": 1.0})
            st.rerun()


def build_model_rows(rows: list[dict], catalog: list[dict]) -> list[ModelRow]:
    # Build lookup for per-model max_tokens from catalog
    catalog_max_tokens = {m["id"]: m.get("max_tokens") for m in catalog}

    result = []
    for r in rows:
        if not r.get("model_id"):
            continue
        temp = r.get("temperature") if r.get("use_temp") else None
        max_tokens = catalog_max_tokens.get(r["model_id"])  # None if not in catalog or not set
        result.append(ModelRow(
            model_id=r["model_id"],
            timeout_s=r["timeout_s"],
            n_calls=r["n_calls"],
            temperature=temp,
            max_tokens=max_tokens
        ))
    return result


def compute_dry_run(questions: list[str], docs_count: int, doer_rows: list, judge_rows: list, final_rows: list, benchmark_enabled: bool, benchmark_mode: str, combine_docs: bool = False):
    D = 1 if combine_docs and docs_count >= 2 else max(1, docs_count)
    Q = len(questions)
    doer_n = sum(r.get("n_calls", 1) for r in doer_rows if r.get("model_id"))
    judge_n = sum(r.get("n_calls", 1) for r in judge_rows if r.get("model_id"))
    final_n = sum(r.get("n_calls", 1) for r in final_rows if r.get("model_id"))
    base_calls = D * Q * (doer_n + judge_n + final_n)
    score_calls = base_calls if benchmark_enabled and benchmark_mode == "llm" else 0
    return {"D": D, "Q": Q, "base": base_calls, "score": score_calls, "total": base_calls + score_calls}


def compute_accuracy_report(scores: list[dict]) -> dict:
    buckets = {}
    for s in scores:
        key = f"{s['stage']}:{s['model_id']}"
        if key not in buckets:
            buckets[key] = []
        if s["score"] is not None:
            buckets[key].append(s["score"])
    result = {}
    for key, vals in buckets.items():
        result[key] = {
            "n_scored": len(vals),
            "accuracy": sum(vals) / len(vals) if vals else 0.0
        }
    return result


def main():
    init_session_state()

    st.title("LLM Aggregation Tool")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Questions (required)")
        questions_text = st.text_area("One question per line", height=150, key="questions_input")
        questions = parse_questions(questions_text)
        st.caption(f"Q = {len(questions)}")

        st.subheader("Documents (optional)")
        uploaded_files = st.file_uploader(
            f"Upload up to {MAX_FILES} files (max {MAX_FILE_SIZE // 1024}KB each)",
            accept_multiple_files=True, key="docs_uploader"
        )
        docs_count = len(uploaded_files) if uploaded_files else 0
        oversized_files = []
        if uploaded_files:
            for f in uploaded_files:
                size = f.size
                st.caption(f"{f.name}: {size / 1024:.1f} KB")
                if size > MAX_FILE_SIZE:
                    oversized_files.append(f.name)
            if len(uploaded_files) > MAX_FILES:
                st.error(f"Too many files. Max {MAX_FILES}.")
            if oversized_files:
                st.error(f"Files exceed {MAX_FILE_SIZE // 1024}KB: {', '.join(oversized_files)}")
        st.caption(f"D = {max(1, docs_count)}")

        # Combine documents option (only show when 2+ files)
        combine_docs = False
        if docs_count >= 2:
            combine_docs = st.checkbox(
                "Combine all documents into single context",
                value=False,
                key="combine_docs",
                help="When enabled, all documents are merged into one combined context for each question"
            )

        st.subheader("Doers")
        with st.expander("System Prompt", expanded=False):
            doer_system = st.text_area("Doer system prompt", value=SYSTEM_PROMPTS.get("doer", ""), key="doer_system", label_visibility="collapsed")
        render_model_rows_editor("doer", "doer_rows", MODELS_CATALOG)

        st.subheader("Judges (optional)")
        with st.expander("System Prompt", expanded=False):
            judge_system = st.text_area("Judge system prompt", value=SYSTEM_PROMPTS.get("judge", ""), key="judge_system", label_visibility="collapsed")
        render_model_rows_editor("judge", "judge_rows", MODELS_CATALOG)

        st.subheader("Final Judges (optional)")
        with st.expander("System Prompt", expanded=False):
            final_system = st.text_area("Final judge system prompt", value=SYSTEM_PROMPTS.get("final", ""), key="final_system", label_visibility="collapsed")
        render_model_rows_editor("final", "final_rows", MODELS_CATALOG)

    with col_right:
        st.subheader("Options")
        send_doc_to_judges = st.checkbox("Send doc to judges", value=PIPELINE_OPTIONS.get("send_doc_to_judges", False), key="send_doc_judges")
        send_doc_to_final = st.checkbox("Send doc to final judges", value=PIPELINE_OPTIONS.get("send_doc_to_final_judges", False), key="send_doc_final")
        send_doer_to_judges = st.checkbox("Send doer responses to judges", value=PIPELINE_OPTIONS.get("send_doer_responses_to_judges", True), key="send_doer_judges")
        send_doer_to_final = st.checkbox("Send doer outputs to final judges", value=PIPELINE_OPTIONS.get("send_doer_outputs_to_final_judges", True), key="send_doer_final")
        send_judge_to_final = st.checkbox("Send judge outputs to final judges", value=PIPELINE_OPTIONS.get("send_judge_outputs_to_final_judges", True), key="send_judge_final")

        st.subheader("Global Controls")
        cap_total_calls = st.number_input("Cap total calls", value=GLOBAL_CONTROLS.get("cap_total_calls", 100), min_value=1, key="cap_calls")
        max_output_tokens = st.number_input("Max output tokens", value=GLOBAL_CONTROLS.get("max_output_tokens", 1000), min_value=1, key="max_tokens")
        retries = st.number_input("Retries (non-timeout)", value=GLOBAL_CONTROLS.get("retries", 0), min_value=0, key="retries")
        max_concurrency = st.number_input("Max concurrency", value=GLOBAL_CONTROLS.get("max_concurrency", 1000), min_value=1, key="max_conc")
        debug_mode = st.checkbox("Debug mode", value=GLOBAL_CONTROLS.get("debug_mode", False), key="debug_mode")

        st.subheader("Benchmark")
        benchmark_enabled = st.checkbox("Enable benchmarking", value=BENCHMARK_DEFAULTS.get("enabled", False), key="bench_enabled")
        ground_truth_text = ""
        benchmark_mode = BENCHMARK_DEFAULTS.get("mode", "exact")
        strip_whitespace = BENCHMARK_DEFAULTS.get("strip_whitespace", True)
        scorer_model = BENCHMARK_DEFAULTS.get("scorer_model") or ""
        scorer_timeout = BENCHMARK_DEFAULTS.get("scorer_timeout_s", 30.0)
        scorer_temp = BENCHMARK_DEFAULTS.get("scorer_temperature", 0.0)

        if benchmark_enabled:
            ground_truth_text = st.text_area("Ground truth (one per question)", key="ground_truth")
            benchmark_mode = st.radio("Mode", ["exact", "llm"], key="bench_mode")
            if benchmark_mode == "exact":
                strip_whitespace = st.checkbox("Strip whitespace", value=strip_whitespace, key="strip_ws")
            else:
                scorer_options = [m["id"] for m in MODELS_CATALOG]
                scorer_model = st.selectbox("Scorer model", scorer_options, key="scorer_model") if scorer_options else ""
                scorer_custom = st.text_input("Or custom scorer model", key="scorer_custom")
                if scorer_custom:
                    scorer_model = scorer_custom
                scorer_timeout = st.number_input("Scorer timeout(s)", value=scorer_timeout, min_value=0.1, key="scorer_timeout")
                scorer_temp = st.number_input("Scorer temperature", value=scorer_temp, min_value=0.0, max_value=2.0, key="scorer_temp")

        st.subheader("Dry Run Estimate")
        dry_run = compute_dry_run(
            questions, docs_count,
            st.session_state.doer_rows,
            st.session_state.judge_rows,
            st.session_state.final_rows,
            benchmark_enabled, benchmark_mode, combine_docs
        )
        st.text(f"D = {dry_run['D']}, Q = {dry_run['Q']}")
        st.text(f"BaseCalls = {dry_run['base']}")
        st.text(f"ScoreCalls = {dry_run['score']}")
        st.text(f"TotalEstimate = {dry_run['total']}")

        cap_exceeded = dry_run["total"] > cap_total_calls
        no_questions = len(questions) == 0
        no_doers = not any(r.get("model_id") for r in st.session_state.doer_rows)
        files_invalid = len(uploaded_files) > MAX_FILES if uploaded_files else False
        files_oversized = len(oversized_files) > 0

        ground_truths = parse_questions(ground_truth_text)
        truth_mismatch = benchmark_enabled and len(ground_truths) != len(questions) and len(ground_truths) > 0

        can_run = not (cap_exceeded or no_questions or no_doers or files_invalid or files_oversized or truth_mismatch)

        if cap_exceeded:
            st.error("Estimated calls exceed cap!")
        if no_questions:
            st.warning("Enter at least one question")
        if no_doers:
            st.warning("Add at least one doer model")
        if truth_mismatch:
            st.error(f"Ground truth lines ({len(ground_truths)}) must match questions ({len(questions)})")

        run_clicked = st.button("Run", disabled=not can_run, key="run_btn", type="primary", use_container_width=True)

    if run_clicked and can_run:
        docs = []
        if uploaded_files:
            for f in uploaded_files:
                f.seek(0)
                docs.append(encode_file(f))

        # Combine documents if option is enabled and there are 2+ docs
        if combine_docs and len(docs) >= 2:
            docs = [combine_documents(docs)]

        doers = build_model_rows(st.session_state.doer_rows, MODELS_CATALOG)
        judges = build_model_rows(st.session_state.judge_rows, MODELS_CATALOG)
        finals = build_model_rows(st.session_state.final_rows, MODELS_CATALOG)

        benchmark_cfg = BenchmarkConfig(
            enabled=benchmark_enabled,
            mode=benchmark_mode,
            strip_whitespace=strip_whitespace,
            ground_truths=ground_truths if benchmark_enabled else [],
            scorer=ScorerConfig(model_id=scorer_model, timeout_s=scorer_timeout, temperature=scorer_temp) if benchmark_mode == "llm" and scorer_model else None
        )

        config = RunConfig(
            questions=questions,
            docs=docs,
            doers=doers,
            judges=judges,
            final_judges=finals,
            doer_system_prompt=doer_system,
            judge_system_prompt=judge_system,
            final_system_prompt=final_system,
            send_doc_to_judges=send_doc_to_judges,
            send_doc_to_final_judges=send_doc_to_final,
            send_doer_responses_to_judges=send_doer_to_judges,
            send_doer_outputs_to_final_judges=send_doer_to_final,
            send_judge_outputs_to_final_judges=send_judge_to_final,
            cap_total_calls=cap_total_calls,
            max_output_tokens=max_output_tokens,
            retries=retries,
            debug_mode=debug_mode,
            max_concurrency=max_concurrency,
            benchmark=benchmark_cfg
        )

        run_id = generate_run_id()
        out_dir = Path("runs") / run_id

        write_resolved_config(str(out_dir), config)

        st.session_state.run_logs = []
        counters = {"ok": 0, "error": 0, "timeout": 0, "skipped": 0}
        total_estimate = dry_run["total"]

        progress_bar = st.progress(0)
        status_text = st.empty()
        debug_container = st.empty() if debug_mode else None

        def on_attempt(record):
            append_call_log(str(out_dir), record)
            st.session_state.run_logs.append(record)
            status = record.get("status", "")
            if status == "ok":
                counters["ok"] += 1
            elif status == "timeout":
                counters["timeout"] += 1
            elif status == "skipped_budget":
                counters["skipped"] += 1
            else:
                counters["error"] += 1
            done = counters["ok"] + counters["error"] + counters["timeout"] + counters["skipped"]
            pct = min(1.0, done / total_estimate) if total_estimate > 0 else 1.0
            progress_bar.progress(pct)
            status_text.text(f"OK: {counters['ok']} | Error: {counters['error']} | Timeout: {counters['timeout']} | Skipped: {counters['skipped']}")
            if debug_container:
                with debug_container.container():
                    st.caption(f"[{record.get('stage')}] {record.get('model_id')} - {record.get('status')} - {record.get('latency_ms', 0):.0f}ms")

        results, all_attempts, scores = asyncio.run(run_pipeline(config, run_id, on_attempt))

        results_data = {"run_id": run_id, "items": results}
        write_results(str(out_dir), results_data)

        stats = compute_stats(all_attempts)
        write_stats(str(out_dir), stats)

        accuracy = {}
        if benchmark_enabled and scores:
            accuracy = compute_accuracy_report(scores)
            write_accuracy(str(out_dir), accuracy)

        st.session_state.run_results = results_data
        st.session_state.run_stats = stats
        st.session_state.run_accuracy = accuracy
        st.session_state.run_config = config.model_dump()
        st.session_state.run_id = run_id

        st.success(f"Run complete! Output saved to runs/{run_id}/")

    if st.session_state.run_results:
        st.divider()
        st.header("Results")

        results_data = st.session_state.run_results
        for item in results_data["items"]:
            st.subheader(f"Doc: {item['doc_id']} | Q{item['q_index']}: {item['question'][:80]}...")

            st.write("**Primary Outputs:**")
            for out in item["primary_outputs"]:
                status_icon = "OK" if out["status"] == "ok" else out["status"].upper()
                st.write(f"[{status_icon}] {out['model_id']}#{out['call_index']}: {out.get('text', 'N/A')}")

            with st.expander("Doer Outputs"):
                for out in item["doers"]:
                    st.write(f"[{out['status']}] {out['model_id']}#{out['call_index']}: {out.get('text', 'N/A')}")

            if item["judges"]:
                with st.expander("Judge Outputs"):
                    for out in item["judges"]:
                        st.write(f"[{out['status']}] {out['model_id']}#{out['call_index']}: {out.get('text', 'N/A')}")

            if item["finals"]:
                with st.expander("Final Outputs"):
                    for out in item["finals"]:
                        st.write(f"[{out['status']}] {out['model_id']}#{out['call_index']}: {out.get('text', 'N/A')}")

            item_logs = [log for log in st.session_state.run_logs
                        if log.get("doc_id") == item["doc_id"] and log.get("q_index") == item["q_index"]]
            if item_logs:
                with st.expander("Attempt Metadata"):
                    for log in item_logs:
                        usage = log.get("usage") or {}
                        st.write(f"**{log.get('stage')}** {log.get('model_id')}#{log.get('call_index')} (attempt {log.get('attempt', 0)})")
                        st.write(f"  Status: {log.get('status')} | Latency: {log.get('latency_ms', 0):.0f}ms | Cost: ${usage.get('cost_usd', 0) or 0:.6f}")
                        st.write(f"  Tokens: {usage.get('prompt_tokens', 0) or 0} prompt + {usage.get('completion_tokens', 0) or 0} completion = {usage.get('total_tokens', 0) or 0} total")

        if st.session_state.run_config and st.session_state.run_config.get("debug_mode") and st.session_state.run_logs:
            with st.expander("Debug: All Attempt Logs"):
                for log in st.session_state.run_logs:
                    with st.expander(f"{log.get('stage')} - {log.get('model_id')} - {log.get('status')}"):
                        st.write(f"**Latency:** {log.get('latency_ms', 0):.0f}ms")
                        usage = log.get("usage") or {}
                        st.write(f"**Cost:** ${usage.get('cost_usd', 0) or 0:.6f}")
                        st.write(f"**Tokens:** {usage.get('prompt_tokens', 0) or 0} / {usage.get('completion_tokens', 0) or 0} / {usage.get('total_tokens', 0) or 0}")
                        if log.get("request"):
                            st.write("**Request Messages:**")
                            st.json(log["request"].get("messages", []))
                        if log.get("response_text"):
                            st.write("**Response:**")
                            st.text(log["response_text"])

        st.divider()
        st.header("Stats")
        if st.session_state.run_stats:
            st.json(st.session_state.run_stats)

        if st.session_state.run_accuracy:
            st.header("Accuracy")
            st.json(st.session_state.run_accuracy)

        st.divider()
        st.header("Downloads")

        run_id = st.session_state.run_id
        out_dir = Path("runs") / run_id

        config_path = out_dir / "resolved_config.json"
        if config_path.exists():
            st.download_button("Download resolved_config.json", config_path.read_text(), "resolved_config.json", "application/json")

        results_path = out_dir / "results.json"
        if results_path.exists():
            st.download_button("Download results.json", results_path.read_text(), "results.json", "application/json")

        logs_path = out_dir / "call_logs.jsonl"
        if logs_path.exists():
            st.download_button("Download call_logs.jsonl", logs_path.read_text(), "call_logs.jsonl", "text/plain")

        stats_json_path = out_dir / "stats.json"
        if stats_json_path.exists():
            st.download_button("Download stats.json", stats_json_path.read_text(), "stats.json", "application/json")

        stats_csv_path = out_dir / "stats.csv"
        if stats_csv_path.exists():
            st.download_button("Download stats.csv", stats_csv_path.read_text(), "stats.csv", "text/csv")

        accuracy_json_path = out_dir / "accuracy.json"
        if accuracy_json_path.exists():
            st.download_button("Download accuracy.json", accuracy_json_path.read_text(), "accuracy.json", "application/json")

        accuracy_csv_path = out_dir / "accuracy.csv"
        if accuracy_csv_path.exists():
            st.download_button("Download accuracy.csv", accuracy_csv_path.read_text(), "accuracy.csv", "text/csv")


if __name__ == "__main__":
    main()
