# Claude Code Session Learnings

This document captures challenges encountered during development and their resolutions to help new developers onboard faster.

---

## Project Context

The LLM Aggregation Tool is a multi-stage pipeline for querying LLMs via OpenRouter. Key features:
- Three-stage architecture: Doers → Judges → Final Judges
- Async execution with configurable concurrency
- Benchmarking with exact match or LLM-based scoring
- Support for text, PDF, and image documents

---

## Challenges and Resolutions

### 1. Image Token Consumption

**Problem:** Images encoded as base64 text consumed excessive tokens.
- gpt-4o-mini failed with 138K tokens for a 109KB image
- Exceeded 128K context limit

**Solution:** Use native multimodal format instead of base64 text in prompts.

```python
# WRONG: Base64 text in prompt
{"role": "user", "content": f"Here's the image: {base64_string}"}

# CORRECT: Native multimodal format
{"role": "user", "content": [
    {"type": "text", "text": "QUESTION: ..."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
]}
```

**Impact:** Token usage dropped from 138K to ~20K for images.

**Files affected:** `llm_agg/prompts.py`, `llm_agg/runner.py`

---

### 2. PDF Content Not Readable by LLMs

**Problem:** PDFs encoded as base64+zlib were unreadable.
- Model received compressed binary garbage
- 0% accuracy on PDF documents (hallucinated answers)

**Solution:** Dual encoding - extract text AND render first page as image.

```python
# Using PyMuPDF (fitz)
import fitz

doc = fitz.open(pdf_path)
text = ""
for page in doc:
    text += page.get_text()

# Render first page as image
page = doc[0]
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
image_bytes = pix.tobytes("png")
```

**Impact:** PDF accuracy improved from 0% to 82%.

**Files affected:** `llm_agg/cli.py` (`_load_doc` function)

---

### 3. NoneType Error in Stats Computation

**Problem:** `a.get("usage", {}).get("cost_usd")` crashed when `usage` was None.

```python
# This crashes if usage is None (not missing, but explicitly None)
cost = a.get("usage", {}).get("cost_usd")
```

**Solution:** Defensive pattern for optional nested fields.

```python
# Correct pattern
cost = (a.get("usage") or {}).get("cost_usd", 0) or 0
```

**Lesson:** Always use `(x or {})` pattern when accessing nested optional fields, since `get()` with default only handles missing keys, not `None` values.

**Files affected:** `llm_agg/stats.py`

---

### 4. Benchmark Running Too Slowly

**Problem:** Initial benchmark with max_concurrency=10 took too long for 200 iterations.

**Solution:** Increase concurrency dramatically for parallel execution.

```python
# Original (slow)
max_concurrency=10

# Fixed (fast)
max_concurrency=1000  # or even 2000
```

**Approach for long benchmarks:**
1. Start benchmark with moderate concurrency
2. Monitor progress
3. If too slow, create a "finish" script with high concurrency to complete remaining iterations

**Files created:** `finish_benchmark.py`

---

### 5. Scoring Stage Confusion

**Problem:** When extracting scores for benchmarking, confused about which stage's scores to use.

**Resolution:**
- **Single model runs:** Use `stage="doer"` scores (there are no judges)
- **ASAP runs:** Use `stage="final"` scores (the aggregated answer)

```python
# For single model
for score in record["scores"]:
    if score["score"] is not None and score["stage"] == "doer":
        scores[key].append(score["score"])

# For ASAP
for score in record["scores"]:
    if score["score"] is not None and score["stage"] == "final":
        scores[key].append(score["score"])
```

**Files affected:** `calc_intermediate_stats.py`, `calc_focused_stats.py`

---

### 6. Question Index Mapping in Focused Tests

**Problem:** When testing only specific questions (e.g., Q10, Q11), the q_index in results started at 0 but referred to original question numbers.

**Solution:** Track the mapping explicitly.

```python
# When loading only Q10 and Q11
questions = [all_questions[9], all_questions[10]]  # 0-indexed originals

# When displaying results, map back
q_labels = {0: "Q10", 1: "Q11"}  # New index -> original label
```

**Files affected:** `benchmark_focused.py`, `calc_focused_stats.py`

---

### 7. State-of-the-Art Aggregation Improvements

**Problem:** Initial implementation lacked explicit voting mechanisms and relied entirely on judge synthesis, missing potential +15-25% accuracy gains from established techniques.

**Research Findings:**
- Self-consistency (majority voting) provides +6-18% on reasoning tasks
- Chain-of-thought prompting provides +5-10%
- Confidence-weighted voting provides +2-5%
- Heterogeneous model ensembles reduce systematic bias (+3-5%)

**Solution:** Implemented Phase 1 improvements:

1. **Chain-of-thought doer prompts** - Step-by-step reasoning before answering
2. **Confidence scoring** - Doers report confidence (1-10) for weighted voting
3. **Explicit judge selection** - Judges must output `SELECTED: [doer:model#index]`
4. **Aggregation utilities** - New `aggregation.py` module with:
   - `extract_answer()` - Parse final answer from verbose response
   - `extract_confidence()` - Parse confidence score
   - `majority_vote()` - Self-consistency voting with optional confidence weighting
   - `aggregate_judge_selections()` - Count judge votes for doers
   - `compute_agreement_score()` - Measure consensus among responses

**Expected Impact:**
- +8-15% from prompt changes alone (CoT + explicit selection)
- +5-10% from voting mechanism when implemented in scoring
- Better interpretability through aggregation statistics

**Files affected:** `llm_agg/prompts.py`, `llm_agg/runner.py`, `llm_agg/aggregation.py`, `settings.json`

**See also:** `Research/04_Implementation_Comparison_and_Improvements.md` for full analysis

---

### 8. Benchmark Parallelization and Model Updates

**Problem:** Original benchmark was slow due to sequential iteration execution and low concurrency.

**Issues identified:**
- `max_concurrency=10` instead of 1000+
- Iterations ran in sequential for-loop instead of parallel `asyncio.gather()`
- Using deprecated model `google/gemini-2.0-flash-001`
- Temperature=0 for scorer caused issues with Gemini

**Solution:**
1. Updated model to `google/gemini-3-flash-preview`
2. Increased `max_concurrency` to 2000
3. Changed to fully parallel iteration execution with `asyncio.gather()`
4. Set temperature > 0 for all Gemini calls (0.5 for judges/scorers, 0.7 for doers)
5. Increased `max_output_tokens` to 500 for chain-of-thought responses

**Configuration changes:**
```python
# Before
GEMINI_MODEL = "google/gemini-2.0-flash-001"
max_concurrency = 10
# Sequential: for i in range(100): await run_iteration(...)

# After
GEMINI_MODEL = "google/gemini-3-flash-preview"
max_concurrency = 2000
# Parallel: await asyncio.gather(*[run_iteration(i) for i in range(100)])
```

**Files affected:** `benchmark_asap_vs_single.py`

---

## Async Patterns Used

### Concurrency Control

```python
# Global semaphore limits concurrent API calls
semaphore = asyncio.Semaphore(max_concurrency)

async def make_call():
    async with semaphore:
        # API call here
        pass
```

### Budget Tracking

```python
# Thread-safe budget counter
class BudgetCounter:
    def __init__(self, cap):
        self.cap = cap
        self.used = 0
        self.lock = asyncio.Lock()

    async def try_use(self):
        async with self.lock:
            if self.used >= self.cap:
                return False
            self.used += 1
            return True
```

### Parallel Execution

```python
# Run all iterations in parallel
tasks = [run_iteration(config, i) for i in range(100)]
results = await asyncio.gather(*tasks, return_exceptions=True)

# Count successes/failures
success = sum(1 for r in results if not isinstance(r, Exception))
```

---

## Testing Patterns

### Running Benchmark Iterations

1. **Sequential (slow but safe):**
```python
for i in range(100):
    await run_iteration(config, i)
```

2. **Parallel (fast):**
```python
tasks = [run_iteration(config, i) for i in range(100)]
await asyncio.gather(*tasks, return_exceptions=True)
```

3. **Hybrid (resume incomplete):**
```python
# Count existing
existing = count_existing_iterations()
remaining = 100 - existing

# Run only remaining
tasks = [run_iteration(config, existing + i) for i in range(remaining)]
await asyncio.gather(*tasks, return_exceptions=True)
```

### Saving Detailed Data

Use JSONL format for incremental writes:

```python
def save_record(record):
    with open("data.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
```

Benefits:
- Append-only (no data loss on crash)
- Easy to count records: `wc -l data.jsonl`
- Easy to process incrementally

---

## Performance Insights

### Document Type Performance

| Type | Accuracy | Notes |
|------|----------|-------|
| Text (.txt) | ~100% | Best for text-heavy content |
| PDF (.pdf) | ~82-95% | Good with dual encoding |
| Image (.jpg) | ~27% | Limited by visual info only |

### ASAP vs Single Model

- Overall improvement: +1-2% on easy questions
- Significant improvement on difficult questions: +20-48%
- Most benefit when single model consistently fails
- More doers + judges = better aggregation

### Concurrency Sweet Spots

| Use Case | Recommended Concurrency |
|----------|------------------------|
| Interactive (UI) | 10-20 |
| Batch processing | 50-100 |
| Benchmark runs | 500-2000 |

---

## Common Debugging Steps

### 1. API Errors

Check `call_logs.jsonl` for:
- `status`: "ok", "timeout", "error", "skipped_budget"
- `http_status`: Should be 200
- `error_message`: Detailed error info

### 2. Wrong Answers

1. Check if document was loaded correctly:
   - Text docs: `encoding="utf-8"`
   - PDFs: Should have both `content` (text) and `pdf_image`
   - Images: `encoding="image"`

2. Check prompts in call_logs.jsonl:
   - Is document included in prompt?
   - Are doer responses included for judges?

### 3. Budget Issues

```python
# Quick budget check
with open("call_logs.jsonl") as f:
    statuses = [json.loads(line)["status"] for line in f]
    print(f"OK: {statuses.count('ok')}")
    print(f"Skipped: {statuses.count('skipped_budget')}")
```

---

## File Reference

| File | Purpose |
|------|---------|
| `llm_agg/runner.py` | Main pipeline execution |
| `llm_agg/prompts.py` | Message building for each stage |
| `llm_agg/openrouter.py` | API client |
| `llm_agg/config.py` | Pydantic models |
| `llm_agg/cli.py` | CLI + document loading |
| `benchmark_asap_vs_single.py` | Main benchmark |
| `benchmark_focused.py` | Focused benchmark |
| `calc_intermediate_stats.py` | Stats from benchmark data |

---

## Tips for New Developers

1. **Start with the Web UI** (`streamlit run app_streamlit.py`) to understand the flow
2. **Read `runner.py`** - it's the core of the pipeline
3. **Use debug mode** to see all API calls
4. **Check `call_logs.jsonl`** when things go wrong
5. **Run small tests first** - use 1-2 iterations before running 100
6. **Monitor API costs** - LLM scoring doubles the API calls

---

## Critical: Claude Code Instructions

**IMPORTANT FOR ALL CLAUDE CODE SESSIONS:**

### Always Verify Before Confirming

When the user says they've added, created, or configured something:
1. **VERIFY IT EXISTS** - Check the file/resource actually exists
2. **VERIFY IT WORKS** - Test that it functions correctly
3. **REPORT ISSUES IMMEDIATELY** - If something is missing or broken, tell the user right away

### When Tasks Fail

If you cannot perform a task properly for ANY reason:
1. **Try harder first** - Attempt alternative approaches
2. **If still failing, NOTIFY THE USER IMMEDIATELY** with:
   - What you tried
   - What went wrong
   - What's needed to fix it

**Never assume success without verification. Never silently skip missing dependencies.**

### Examples of Required Verification

- User says "I added the API key" → **Check the file exists and loads correctly**
- User says "the file is at X" → **Verify the file exists at that path**
- User says "run the tests" → **Verify tests pass, report failures immediately**
- Setting up configuration → **Test the configuration actually works**

### What NOT to Do

- ❌ Assume a file exists without checking
- ❌ Report success without testing
- ❌ Silently fall back to alternatives without informing user
- ❌ Proceed with partial setup without flagging missing pieces

### After Adding New Features

**ALWAYS update documentation after implementing new features:**

1. Update `README.md` with:
   - Feature description and how it works
   - Usage examples
   - Any new configuration options

2. If the feature involved solving a notable challenge, add it to the "Challenges and Resolutions" section in this file (CLAUDE.md)

3. Commit documentation updates separately or together with the feature

### After Code Changes

**ALWAYS restart the application after making any code changes:**

```bash
# For Streamlit app
pkill -f streamlit; python3 -m streamlit run app_streamlit.py --server.port 8501

# For CLI/scripts
# Just re-run the command - Python will reload modules
```

**Why:** Python caches imported modules. Without restart, the app uses old code even after you edit files. This leads to confusion where "fixes don't work" when they actually do - just not loaded yet.

---

## API Key Security

### Secure Storage

API keys are loaded in this priority order:
1. Environment variables (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`)
2. Key files in project root (`OpenAIAPIKey.txt`, `GoogleAPIKey.txt`, `AnthropicAPIKey.txt`, `OpenRouterAPIKey.txt`)

### Git Protection

The `.gitignore` excludes:
- `*APIKey*.txt` - all API key files
- `*.key` - any key files
- `.env` and `.env.*` - environment files

### API Routing

The system automatically routes requests:
- `openai/*` or `gpt-*` models → OpenAI API (direct)
- `google/*` or `gemini*` models → Google Generative AI API (direct)
- `anthropic/*` or `claude-*` models → Anthropic API (direct)
- All other models → OpenRouter API

This allows using native APIs directly (often faster/cheaper) while still supporting other providers through OpenRouter.

### Adding a New API Key

1. **For OpenAI API:**
   ```bash
   export OPENAI_API_KEY="your-key"
   # OR
   echo "your-key" > OpenAIAPIKey.txt
   ```

2. **For Google API:**
   ```bash
   export GOOGLE_API_KEY="your-key"
   # OR
   echo "your-key" > GoogleAPIKey.txt
   ```

3. **For Anthropic API:**
   ```bash
   export ANTHROPIC_API_KEY="your-key"
   # OR
   echo "your-key" > AnthropicAPIKey.txt
   ```

4. **For OpenRouter:**
   ```bash
   export OPENROUTER_API_KEY="your-key"
   # OR
   echo "your-key" > OpenRouterAPIKey.txt
   ```

Never commit key files - they're git-ignored for safety.

---

## Settings File

All app configuration is centralized in `settings.json`:

```json
{
  "models": [...],           // Model catalog with per-model max_tokens
  "system_prompts": {...},   // Default prompts for doer/judge/final/scorer
  "default_model": "...",    // Default model for new doer rows
  "model_defaults": {...},   // Default timeout, n_calls, temperature
  "global_controls": {...},  // cap_total_calls, max_output_tokens, etc.
  "pipeline_options": {...}, // What data flows between stages
  "benchmark_defaults": {...}, // Benchmark configuration
  "ui": {...}                // UI limits (max_files, max_model_rows)
}
```

Edit this file to customize the app without changing code.
