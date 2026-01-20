**Target stack (current):** Python 3.14.2, Streamlit 1.53.0, httpx 0.28.1, pydantic 2.12.5 (pandas optional) [Xenoss](https://xenoss.io/blog/openrouter-vs-litellm?utm_source=chatgpt.com)
## 0) Purpose

Build a simple LLM aggregation tool with:
- A **web UI** for interactive runs.
- A **CLI** for benchmarking (repeat runs + aggregate stats).
- Uses **OpenRouter** (OpenAI-compatible chat completions API) for all model calls.
    
The tool runs a multi-stage pipeline over **(documents × questions)**, fan-outs to multiple models/calls per stage, collects responses, and computes stats + optional benchmarking vs ground truth.
Non-goals: production hardening, accounts/auth, DB, job queue, streaming tokens, document parsing/chunking.

---
## 1) Terminology
- **Doc**: an uploaded file (0–10). If none uploaded, treat as a single “no-doc” iteration.
- **Question**: one line in the questions textarea (required).
- **Stage**:
    - **Doers** = stage 1
    - **Judges** = stage 2 (optional)
    - **Final judges** = stage 3 (optional)
- **Model row**: a configured model + params + `n_calls` for a stage.
- **Call**: a single HTTP request to OpenRouter.
- **Attempt**: a call attempt including retries. (One logical call may produce multiple attempts.)
- **Valid response**: a call that returns a non-empty assistant message.

---
## 2) High-level behavior (must match)

For each doc (or one iteration if no docs), for each question:

1. **Doers**: send `(doc, doer_system_prompt, question)` to each doer model row `n_calls` times; collect responses.
2. If judges list is non-empty:
    - **Judges**: send `(question, judge_system_prompt, doer responses)` and optionally doc to each judge model row `n_calls` times; collect responses.
3. If final judges list is non-empty:
    - **Final judges**: send `(question, final_system_prompt)` plus optional doc plus optional doer outputs plus optional judge outputs to each final model row `n_calls` times; collect responses.
4. Outputs returned to user:
    - If final judges list **non-empty**: return final judge responses.
    - Else: return doer responses.
5. Compute stats:
    - Overall + per stage + per model-id:
        -  calls, cost, avg latency to valid response, % valid, % errors, % timeouts.
6. Benchmarking (optional):
    - If ground truth provided:
        - Compare **each response** (doer + judge + final) to ground truth:
            - Either exact string match (no LLM)
            - Or LLM scorer model outputs 0/1
        - Report accuracy per **stage × model-id**.
Constraints:
- Parallelize as much as possible.
- Respect timeouts; if some calls timeout, continue with others (no retry on timeout).
- Respect retries on non-timeout errors.
- Respect a global cap on total LLM calls (default 100), including retries and LLM-based scoring.
---
## 3) Architecture

### 3.1 Components
1. **Core runner library** (importable module)
    - All pipeline logic
    - OpenRouter HTTP client
    - Stats + logging + output writing
2. **Web UI** (Streamlit)
    - Builds a config
    - Shows dry-run counter
    - Runs the core runner and displays results
3. **CLI**
    - Runs core runner with JSON config and/or dataset files
    - Supports benchmark loops (repeat N times) and aggregates results
### 3.2 File layout (recommended)

`llm_agg/   __init__.py   config.py          # pydantic models + defaults + validation   openrouter.py      # async client   prompts.py         # message builders + default system prompts   runner.py          # pipeline execution + concurrency + budget   stats.py           # aggregation utilities   io.py              # write outputs, jsonl logs   cli.py             # argparse entrypoints app_streamlit.py     # UI models_catalog.json  # provided model list`

---
## 4) External dependencies / versions (current)

- Python: **3.14.2**
- Streamlit: **1.53.0** [Xenoss](https://xenoss.io/blog/openrouter-vs-litellm?utm_source=chatgpt.com)
- httpx: **0.28.1**
- pydantic: **2.12.5**
- pandas (optional, for CSV exports): **2.3.3**
- python-dotenv (optional): **1.2.1**
---
## 5) OpenRouter integration

### 5.1 Endpoint
Use OpenAI-compatible chat completions:
- `POST https://openrouter.ai/api/v1/chat/completions`
### 5.2 Headers

Required:
- `Authorization: Bearer <OPENROUTER_API_KEY>`
- `Content-Type: application/json`

Optional (recommended):
- `HTTP-Referer: http://localhost` (or your UI URL)
- `X-Title: LLM Aggregation Tool`

### 5.3 Request body
Core fields:
- `model` (string model id, e.g. `google/gemini-2.0-flash`)
- `messages` (system + user)
- `max_tokens` (global per-run max output tokens, default 200)
- `usage: { include: true }` to receive cost + token usage (simplifies stats)

Temperature handling (IMPORTANT):
- Temperature is **optional and default OFF** per model row.
- If temperature is OFF (`null`), **omit** the `temperature` field entirely (OpenRouter default is 1.0).
- If temperature is ON, include `temperature: float`.
### 5.4 Response parsing
A call is “ok” if:
- HTTP 200
- JSON parses
- `choices[0].message.content` exists and is non-empty after `.strip()`
Usage and cost:
- If `usage.include=true`, record tokens + cost from the response.
---
## 6) Inputs and validation rules
### 6.1 Required input
- **Questions**: textarea, newline-separated
    - Parse into list: splitlines → strip → drop empty lines
    - `Q >= 1`
### 6.2 Optional inputs
1. **Documents (0–10)**
    - Any file type accepted
    - Hard file-size limit: **200 KB = 200 * 1024 = 204,800 bytes**
    - If any uploaded file exceeds limit: block the run (UI error)
2. **Stage model configurations**
    - Doers: up to 10 model rows
    - Judges: up to 10 model rows
    - Final judges: up to 10 model rows

Each model row fields:
- `model_id` (string, required)
- `timeout_s` (int/float, required, >0)
- `n_calls` (int, required, >=1)
- `temperature` (optional):
    - UI toggle “Use temperature” (default OFF)
    - If ON: float in [0.0, 2.0]
    - If OFF: `null` and omit from API request
4. **Global run parameters**
- `cap_total_calls` (int, default 100) — applies to all LLM calls including retries and LLM-scoring
- `max_output_tokens` (int, default 200) — sent as `max_tokens`
- `retries` (int, default 0) — retries on non-timeout errors only
- `debug_mode` (bool, default false)
- `max_concurrency` (int, default = OpenRouter free-tier RPM = 20) [OpenRouter](https://openrouter.ai/pricing?utm_source=chatgpt.com)
    - Note: RPM is not the same as concurrency; defaulting to 20 is a simple safe-ish default. Allow user override.

5. **Prompt inputs**
- Doer system prompt (optional; if empty use default)
- Judge system prompt (optional; if empty use default)
- Final judge system prompt (optional; if empty use default)

6. **Checkbox options**
- `send_doc_to_judges` (default false)
- `send_doc_to_final_judges` (default false)
- `send_doer_responses_to_judges` (default true)
- `send_doer_outputs_to_final_judges` (default **true**) ✅ (new)
- `send_judge_outputs_to_final_judges` (default **true**) ✅ (new)

7. **Benchmarking inputs**
- `benchmark_enabled` (bool)
- Ground truth input (UI):
    - textarea with one ground-truth answer per question line (same order)
    - if provided, must have same count as questions (or block run)
- Benchmark scoring mode:
    - `exact` (no LLM calls)
    - `llm` (uses scorer model that outputs 0/1)
- For exact mode:
    - `strip_whitespace` boolean (default true)
- For llm mode:
    - scorer model row: model_id, timeout_s, optional temperature (default ON with 0.0)
---
## 7) Document “send as-is” encoding
For each doc (<= 200KB):
1. Read bytes.
2. Try strict UTF-8 decode:
    - If success: store `encoding="utf-8"` and store `text`
    - Else: store `encoding="base64"` and store `bytes_b64`
When inserted into prompts, include a single doc block:
Text doc block:
`[DOCUMENT] doc_id: <doc_id> filename: <filename> mime: <mime> encoding: utf-8 content: <text> [/DOCUMENT]`
Binary doc block:
`[DOCUMENT] doc_id: <doc_id> filename: <filename> mime: <mime> encoding: base64 content: <base64 bytes> [/DOCUMENT]`
If no docs uploaded:
- Use `doc=None` and **omit** the document block entirely.
---
## 8) Prompting and message building

### 8.1 Default system prompts (constants)

Doer default:
- “You are a research assistant. Use the provided document if present. Answer the question directly. If uncertain, say so briefly.”
Judge default:
- “You are evaluating multiple candidate answers. Identify the best answer and why. Point out errors and missing details.”
Final judge default:
- “Write the best final answer using the candidate answers and the document if present. Output only the final answer.”
LLM scorer default:
- “Output only `0` or `1`. `1` means the candidate agrees with the ground truth; otherwise `0`.”
### 8.2 User message builders
**Doer user message**
`QUESTION: {question}  {doc_block_if_included}`
**Judge user message**  
Include doc only if `send_doc_to_judges`.  
Include doer outputs only if `send_doer_responses_to_judges`.
`QUESTION: {question}  {doc_block_if_included}  DOER RESPONSES: - [doer:{model_id}#{call_index}] {doer_text}  - [doer:{model_id}#{call_index}] {doer_text}  INSTRUCTIONS: Evaluate the responses. Prefer correctness and completeness.`
**Final judge user message**  
Include doc only if `send_doc_to_final_judges`.  
Include doer outputs only if `send_doer_outputs_to_final_judges`.  
Include judge outputs only if judges ran AND `send_judge_outputs_to_final_judges`.
`QUESTION: {question}  {doc_block_if_included}  DOER RESPONSES: ...  JUDGE RESPONSES: ...  INSTRUCTIONS: Write the best final answer. If the document conflicts with a candidate response, trust the document.`
**LLM scorer user message**
`GROUND TRUTH: {truth}  CANDIDATE: {answer}  Does the candidate agree with the ground truth? Output only 1 (agree) or 0 (disagree).`

---
## 9) Pipeline execution details (exact)

### 9.1 Iteration set
Let:
- `docs_iter = uploaded_docs if uploaded_docs else [None]`
- `questions = parsed_questions`

Process all pairs `(doc, question)`.
### 9.2 Stage 1: Doers
For each doer model row:
- Launch `n_calls` async calls in parallel.
- Each call input:
    - system = doer_system_prompt
    - user = doer_user_message(question, doc_block)

Collect doer results:
- Keep an array of attempt records (for logs/stats).
- Keep a list of **valid doer outputs** (for downstream prompts).

### 9.3 Stage 2: Judges (optional)
If judges model rows exist:
- Build judge user message from question + (optional doc) + (optional doer outputs).
- For each judge model row: launch `n_calls` calls in parallel.  
    Collect valid judge outputs.
### 9.4 Stage 3: Final judges (optional)
If final judge model rows exist:
- Build final user message from question + optional doc + optional doer outputs + optional judge outputs.
- For each final model row: launch `n_calls` calls in parallel.  
    Collect valid final outputs.
### 9.5 What the UI returns as “answers”
- If final stage configured (final rows exist): show final outputs as primary answers.
- Else: show doer outputs as primary answers.
- Always keep/show judge outputs in the run artifacts if judges ran.

---
## 10) Concurrency, timeout, retry, and budget
### 10.1 Concurrency
- Use a single global `asyncio.Semaphore(max_concurrency)` around every outbound HTTP request (including retries and LLM scoring).
- Default `max_concurrency = 20` (aligned to documented OpenRouter free-tier RPM). [OpenRouter](https://openrouter.ai/pricing?utm_source=chatgpt.com)
- Expose as user input.
### 10.2 Timeout semantics (per-call)
- Each call uses the model row’s `timeout_s`.
- If a call times out:
    - mark attempt status = `timeout`
    - **do not retry**
    - proceed with other calls and next stages using whatever valid responses exist
### 10.3 Retry semantics (non-timeout failures only)
- On errors like:
    - HTTP non-200
    - 429 rate limits
    - JSON parse failure
    - missing/empty content
- Retry up to `retries` times.
- Each retry is a new attempt record.
- Optional simple backoff (keep simple):
    - sleep `min(2.0, 0.25 * attempt_number)` seconds before retry
### 10.4 Global call cap (hard budget)
- Maintain an atomic counter `calls_used`.
- Before starting any attempt (including retries and scorer calls):
    - if `calls_used >= cap_total_calls`:
        - do not send request
        - record status = `skipped_budget`
    - else increment and proceed
Budget scope:
- Counts:
    - doer calls
    - judge calls
    - final calls
    - LLM scorer calls
    - retries
---
## 11) Dry-run counter (UI, dynamic)

Compute instantly as user edits parameters.
Define:
- `D = max(1, num_docs_uploaded)`
- `Q = number_of_questions`
- `Doer_N = sum(n_calls over all doer model rows)`
- `Judge_N = sum(n_calls over all judge model rows)`
- `Final_N = sum(n_calls over all final model rows)`

Base estimate (matches your formula):
- `BaseCalls = D * Q * (Doer_N + Judge_N + Final_N)`

Scoring estimate:
- If benchmark enabled AND mode == `llm`:
    - Score every stage’s configured responses:
    - `ScoreCalls = D * Q * (Doer_N + Judge_N + Final_N)`
- Else:
    - `ScoreCalls = 0`
Display:
- `TotalCallsEstimate = BaseCalls + ScoreCalls`
- Show component breakdown.
Validation:
- If `TotalCallsEstimate > cap_total_calls`:
    - Disable Run
    - Show “cap exceeded” error
Note:
- This estimate does **not** include retries. Runtime cap enforcement guarantees the tool will not exceed `cap_total_calls`.
---
## 12) Logging, outputs, and reproducibility
### 12.1 Run folder
Every run writes to a folder:
- UI: `runs/<run_id>/`
- CLI: `--out <dir>`
`run_id` format recommended:
- `YYYYMMDD_HHMMSS_<short_random>`
### 12.2 Save resolved config (mandatory)
Before any model call, write:
- `<out>/resolved_config.json`
Resolved config must include:
- Parsed questions array
- Encoded docs (with sizes + encoding)
- All default system prompts filled in if user left blank
- All checkbox values
- Model rows normalized:
    - `temperature: null` if disabled
- Global params (cap, retries, max_tokens, debug_mode, max_concurrency)
- Benchmark section including mode and truth lines
### 12.3 Call logs (JSONL)
Always write:
- `<out>/call_logs.jsonl`  
    One line per attempt record.
Attempt record fields:
`{   "run_id": "...",   "stage": "doer|judge|final|scorer",   "doc_id": "doc1|__no_doc__",   "q_index": 0,   "model_id": "provider/model",   "call_index": 0,   "attempt": 0,    "started_at": "ISO8601",   "ended_at": "ISO8601",   "latency_ms": 1234,    "status": "ok|timeout|error|skipped_budget",   "http_status": 200,   "error_message": "string|null",    "request": {     "messages": [...],     "max_tokens": 200,     "temperature": 1.0   },   "response_text": "string|null",   "response_json": { ... },   "usage": {     "prompt_tokens": 123,     "completion_tokens": 200,     "total_tokens": 323,     "cost_usd": 0.0042   } }`
### 12.4 Results file

Write:
- `<out>/results.json`
Suggested structure:
`{   "run_id": "...",   "items": [     {       "doc_id": "doc1|__no_doc__",       "q_index": 0,       "question": "...",        "doers": [ { "model_id": "...", "call_index": 0, "text": "...", "status": "ok" }, ... ],       "judges": [ ... ],       "finals": [ ... ],        "primary_outputs": [ ... ]  // finals if present else doers     }   ] }`
### 12.5 Stats files

Write:
- `<out>/stats.json` and/or `<out>/stats.csv`
- `<out>/accuracy.json` and/or `<out>/accuracy.csv` (if benchmark enabled)
---
## 13) Stats computation (definitions)
Compute buckets:
1. Overall
2. Per stage
3. Per stage × model_id
For each bucket:
- `attempts_total` (including retries, excluding skipped_budget or include separately)
- `calls_ok` (# attempts with status ok)
- `calls_timeout`
- `calls_error`
- `calls_skipped_budget`
- `valid_rate = calls_ok / (calls_ok + calls_timeout + calls_error)` (exclude skipped)
- `timeout_rate`, `error_rate`
- `avg_latency_ms_ok` over ok attempts
- `sum_cost_usd` over attempts that include usage cost
- Token sums (prompt/completion/total)
Source of cost/tokens: OpenRouter usage accounting.
---
## 14) Benchmarking vs ground truth
### 14.1 Ground truth mapping
UI format:
- One truth line per question (same order)
- truth for `(doc, question)` uses the question’s truth line (no doc-specific overrides in UI)
CLI dataset (recommended for richer tests) may support doc-specific overrides (see CLI section).
### 14.2 Exact string match mode
No LLM calls.
Normalization:
- If `strip_whitespace = true`: compare `truth.strip()` to `answer.strip()`
- Else: compare raw strings
Score:
- `1` if equal else `0`
### 14.3 LLM scorer mode
Uses a configured scorer model row (model_id, timeout, optional temperature default 0.0).
For each valid response (doer + judge + final):
- Create scorer call with system prompt + user message (truth + candidate)
- Parse output:
    - accept only `0` or `1` after strip
    - else scorer attempt = error/invalid
Budget:
- scorer calls count against `cap_total_calls`.
### 14.4 Accuracy reporting
Report accuracy per:
- stage ∈ {doer, judge, final}
- model_id
For each `(stage, model_id)`:
- `n_scored`
- `accuracy = mean(scores)` over scored responses
---
## 15) Web UI spec (Streamlit)
### 15.1 Controls (with defaults)
1. Questions (required)
- Textarea
- Show `Q` count

2. Documents uploader
    
- Multiple files, max 10
- Enforce 200KB per file; show per-file size
- Show `D = max(1, uploaded_docs_count)`

3. Stage panels: Doers / Judges / Final judges  
    Each panel:
- System prompt textarea + “Use default” button
- Model rows editor (max 10 rows):
    - model dropdown (from catalog) + fallback free-text
    - timeout seconds
    - n_calls
    - checkbox “Use temperature” (default OFF)
        - if ON: numeric input temperature
    - remove row
- “Add model” button

4. Options checkboxes
    
- send doc to judges (default off)
- send doc to final judges (default off)
- send doer responses to judges (default on)
- send doer outputs to final judges (default on)
- send judge outputs to final judges (default on)

5. Global controls

- cap_total_calls (default 100)
- max_output_tokens (default 200)
- retries (default 0)
- max_concurrency (default 20)
- debug_mode (default off)

6. Benchmark section
    
- Enable benchmarking checkbox
- Ground truth textarea (must match Q lines)
- Mode radio: Exact match / LLM scorer
    - If Exact: `strip_whitespace` checkbox (default on)
    - If LLM: scorer model config (like a model row, but single)

7. Dry run counter (always visible)
    
- Show BaseCalls, ScoreCalls, TotalCallsEstimate
- Disable Run if TotalCallsEstimate > cap
### 15.2 Run-time progress display
Always show:
- progress bar = (attempts completed) / (TotalCallsEstimate capped by budget)
- counters: ok / error / timeout / skipped_budget
Debug mode ON:
- Live log view (append events)
- Expanders per attempt showing:
    - request messages
    - response text
    - raw JSON (optional)
    - latency, tokens, cost
- Redact sensitive values (API key)
### 15.3 Results display
For each doc:
- For each question:
    - Primary outputs (finals if present else doers)
    - Expanders:
        - Doer outputs
        - Judge outputs
        - Final outputs
        - Attempt metadata table (status, latency, cost)
Download buttons:
- resolved_config.json
- results.json
- call_logs.jsonl
- stats.csv/json
- accuracy.csv/json (if enabled)
---
## 16) CLI spec
### 16.1 Commands
1. Single run
    
- `python -m llm_agg.cli run --config config.json --out out/run1`

2. Benchmark loop
    
- `python -m llm_agg.cli bench --config config.json --dataset dataset.json --repeat 20 --out out/bench1`
### 16.2 CLI config format
Same schema as UI resolved config, but without embedded doc contents if using dataset paths.
### 16.3 Dataset format (for your provided test set)
Folder TestData contains a set of documents (doc1, doc2, doc3) and a file Questions.md with question and answers. 
Resolution rules:
- Run over `docs × questions`.
- If an override exists for `(doc_id, q_id)`, use it; else use question-level truth.
Benchmark mode in CLI:
- `exact` uses string match
- `llm` uses scorer calls (counts toward cap)
Outputs:
- Per run: same output set as UI    
- For `bench --repeat N`:
    - `aggregate_stats.csv/json` (mean/median over runs)
    - `aggregate_accuracy.csv/json`
---
## 17) Error handling requirements
- Any failed call must still create an attempt record with:
    - status, error_message, latency
- Pipeline must continue even if:
    - some doers fail/time out
    - some judges fail/time out
    - some finals fail/time out
- Downstream stages include only **valid** upstream texts (non-empty).
- Timeout: never retried.
- Non-timeout error: retried up to `retries`.
- Budget exhausted: mark remaining attempts `skipped_budget` and continue (likely producing partial outputs).
---
## 18) Acceptance criteria (what “done” means)

### UI
- Can paste multiple questions, upload 0–10 docs (<=200KB each), configure model rows for each stage, run.
- Dry-run counter updates instantly and blocks run if cap exceeded.
- Progress updates during run; debug mode shows all request/response messages.
- Produces primary outputs:
    - finals if configured else doers
- Exports run artifacts including `resolved_config.json`.
### Pipeline correctness
- Nested doc×question behavior matches spec.
- Judges optional; finals optional; finals empty returns doers.
- Checkbox effects are correct:
    - doc to judges/finals
    - doers to judges
    - doers to finals (default on)
    - judges to finals (default on)
### Robustness
- Timeouts do not block pipeline; no retries on timeout.
- Retries work on non-timeout errors.
- Global call cap is never exceeded (including scoring).
### Benchmarking
- Exact match scoring produces per stage×model accuracy.
- LLM scoring produces 0/1 and respects call cap.
### Outputs
- `resolved_config.json` exists for every run and is sufficient to reproduce the same run setup.
---
## 19) Notes for the implementer (keep code small)
- Use one async `httpx.AsyncClient` per run.
- Centralize request building in `openrouter.py`.
- Centralize message building in `prompts.py`.
- One global semaphore + one global budget counter.
- Write call logs incrementally (jsonl) as attempts finish to avoid memory blow-ups.
---
## 20) OpenRouter rate limit default for max_concurrency (source)
Use `max_concurrency=100` as default. [OpenRouter](https://openrouter.ai/pricing?utm_source=chatgpt.com)
## 21) List of LLMs
google/gemini-3-flash-preview
openai/gpt-5-mini
If getting errors consistently, do a web search about the specific current syntax of how to call OpenRouter API
OpenRouterAPIKey.txt provided in the project folder
## 22) Code Quality Rules
### Keep It Minimal
- Implement exactly what's asked, nothing more
- Prefer editing existing code over creating new files
### No Unnecessary Cruft
- No comments unless logic is genuinely non-obvious
- No docstrings on simple/obvious functions
- No dead code or commented-out code
### Error Handling
- No try/except unless failure is expected and handled meaningfully
- Let errors bubble up - they're informative
- Only catch specific exceptions

### Functions
- < 30 lines per function preferred
- One function = one job
- Avoid deep nesting (> 3 levels)
## Don't
- Don't add features not in the plan
- Don't "improve" working code while fixing bugs
- Don't report done without testing