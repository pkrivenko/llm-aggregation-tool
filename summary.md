# LLM Aggregation Tool - Development Summary

## Overview

The LLM Aggregation Tool is a multi-stage pipeline for querying multiple Large Language Models (LLMs) and aggregating their responses. It supports a three-stage architecture: Doers → Judges → Final Judges, where each stage can use different models to generate, evaluate, and synthesize answers.

## Project Structure

```
CAI/
├── app_streamlit.py          # Web UI (Streamlit)
├── llm_agg/                   # Core library
│   ├── __init__.py
│   ├── config.py              # Pydantic models for configuration
│   ├── cli.py                 # Command-line interface
│   ├── runner.py              # Async pipeline execution
│   ├── openrouter.py          # OpenRouter API client
│   ├── prompts.py             # Prompt builders for each stage
│   ├── stats.py               # Statistics computation
│   └── io.py                  # File I/O utilities
├── models_catalog.json        # Available models list
├── TestData/                  # Test documents and questions
│   ├── doc1.txt               # Text document
│   ├── doc2.pdf               # PDF document
│   ├── doc3.jpg               # Image document (resume screenshot)
│   └── Questions.md           # Questions with ground truth answers
├── runs/                      # Output directory for runs
└── req.md                     # Original requirements specification
```

## Implementation Timeline

### Phase 1: Core Implementation

Based on the 22-section specification in `req.md`, the following components were implemented:

1. **config.py** - Pydantic models for type-safe configuration:
   - `ModelRow`: Model configuration (model_id, timeout, n_calls, temperature)
   - `DocInfo`: Document metadata and content
   - `RunConfig`: Complete run configuration
   - `BenchmarkConfig`: Benchmarking settings
   - `ScorerConfig`: LLM scorer configuration

2. **openrouter.py** - Async HTTP client for OpenRouter API:
   - Uses `httpx` for async HTTP requests
   - Handles timeouts and error responses
   - Returns structured results with latency, usage, and response data

3. **runner.py** - Pipeline execution engine:
   - `BudgetCounter`: Thread-safe call budget management with asyncio.Lock
   - `_run_stage()`: Executes a stage (doer/judge/final) for all models
   - `_run_for_doc_question()`: Runs full pipeline for one doc+question pair
   - `run_pipeline()`: Main entry point, handles concurrency with semaphore
   - `_score_response()`: Benchmarking with exact match or LLM scoring

4. **prompts.py** - Message builders:
   - `build_doc_block()`: Formats document for inclusion in prompts
   - `build_doer_user_message()`: Doer stage prompt
   - `build_judge_user_message()`: Judge stage prompt with doer outputs
   - `build_final_user_message()`: Final judge prompt with all outputs
   - `build_scorer_user_message()`: Benchmarking scorer prompt

5. **stats.py** - Statistics computation:
   - Computes per-stage and per-model statistics
   - Tracks latency, token usage, costs, success rates

6. **io.py** - File I/O utilities:
   - `generate_run_id()`: Timestamp-based unique run IDs
   - `write_resolved_config()`: Saves configuration JSON
   - `append_call_log()`: Incremental JSONL logging
   - `write_results()`, `write_stats()`, `write_accuracy()`

7. **cli.py** - Command-line interface:
   - `run` command: Single pipeline run
   - `bench` command: Repeated benchmark runs with aggregation
   - Document loading with encoding detection
   - Questions.md parsing for benchmarks

8. **app_streamlit.py** - Web UI:
   - Question input (multi-line text area)
   - Document upload (up to 10 files, 200KB each)
   - Model configuration for each stage (dropdown + parameters)
   - Options checkboxes for data flow control
   - Global controls (budget, tokens, retries, concurrency)
   - Benchmark configuration
   - Real-time progress display with debug logs
   - Results display with expandable sections
   - Attempt metadata (latency, cost, tokens)

### Phase 2: Bug Fixes and Refinements

1. **stats.py NoneType Error**:
   - **Problem**: `a.get("usage", {}).get("cost_usd")` crashed when `usage` was None
   - **Solution**: Changed to `(a.get("usage") or {}).get("cost_usd", 0) or 0`

2. **Missing MAX_FILE_SIZE constant**:
   - **Problem**: Constant was used but not defined in config.py
   - **Solution**: Added `MAX_FILE_SIZE = 204800` (200KB)

3. **Unused import in io.py**:
   - **Problem**: `import os` was never used
   - **Solution**: Removed the unused import

### Phase 3: Multimodal Image Support

1. **Problem**: Images encoded as base64 text consumed excessive tokens
   - gpt-4o-mini failed with 138K tokens for a 109KB image
   - Token limit exceeded (128K context)

2. **Solution**: Native multimodal format
   - Added `"image"` encoding type to DocInfo
   - Updated prompts.py to return multimodal content arrays:
     ```python
     [
         {"type": "text", "text": "QUESTION: ..."},
         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
     ]
     ```
   - Updated runner.py to handle `Union[str, list]` for user_message
   - Token usage dropped from 138K to ~20K for images

### Phase 4: PDF Dual Encoding

1. **Problem**: PDFs encoded as base64+zlib were unreadable by models
   - Model received compressed binary garbage
   - 0% accuracy on PDF documents (hallucinated answers)

2. **Solution**: Extract text AND render first page as image
   - Uses PyMuPDF (fitz) library
   - Extracts text from all pages
   - Renders first page at 2x zoom as PNG
   - Sends both to model with explanatory prompt:
     ```
     [PDF DOCUMENT: filename.pdf]
     The following is extracted text from the PDF. An image of the first page
     is also provided for layout reference.

     --- EXTRACTED TEXT ---
     {extracted_text}
     --- END EXTRACTED TEXT ---
     ```
   - PDF accuracy improved from 0% to 82%

## Architecture Details

### Pipeline Flow

```
Questions × Documents → Doers → Judges (optional) → Final Judges (optional)
                         ↓           ↓                    ↓
                    [responses]  [evaluations]      [final answer]
```

### Concurrency Model

- `asyncio.Semaphore` limits concurrent API calls (default: 20)
- `BudgetCounter` with `asyncio.Lock` ensures atomic budget tracking
- All API calls within a stage run in parallel via `asyncio.gather()`

### Document Encodings

| Type | Encoding | Content Field | Additional Fields |
|------|----------|---------------|-------------------|
| Text | `utf-8` | Plain text | - |
| Image | `image` | Base64 image data | - |
| PDF | `pdf` | Extracted text | `pdf_image`: Base64 PNG |
| Binary | `base64+zlib` | Compressed base64 | - |

### Message Format

For text documents:
```python
{"role": "user", "content": "QUESTION: ...\n\n[DOCUMENT]\n..."}
```

For images/PDFs (multimodal):
```python
{"role": "user", "content": [
    {"type": "text", "text": "QUESTION: ..."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]}
```

## Key Learnings

### 1. Token Efficiency Matters

- Base64 encoding is extremely token-inefficient (~4x bloat)
- Native multimodal format is much more efficient
- Always use `image_url` content type for images, not base64 text in prompts

### 2. PDFs Need Special Handling

- LLMs cannot parse binary PDF data
- Text extraction alone misses layout/formatting information
- Dual approach (text + image) provides best results:
  - Text: Searchable, precise content
  - Image: Visual layout understanding

### 3. Error Handling for Optional Fields

- Always use defensive patterns: `(a.get("field") or {}).get("subfield", default)`
- API responses may have None for optional fields
- Budget-skipped calls have no usage data

### 4. Async Concurrency Patterns

- Use `asyncio.Semaphore` for rate limiting
- Use `asyncio.Lock` for shared mutable state
- Use `asyncio.gather()` for parallel execution

### 5. Streamlit State Management

- Use `st.session_state` for persistent data across reruns
- Initialize all state in a dedicated function
- Use unique keys for dynamic widget lists

## Common Challenges and Solutions

### Challenge 1: Model Context Limits

**Symptoms**: API returns 400 error with "context length exceeded"

**Solutions**:
- Use multimodal format for images
- Truncate document content if too long
- Use models with larger context windows
- For PDFs: Extract only relevant pages

### Challenge 2: PDF Content Not Readable

**Symptoms**: Model hallucinates answers, mentions wrong names/data

**Solutions**:
- Use PyMuPDF to extract text
- Render pages as images for visual context
- Send both text and image to model

### Challenge 3: Benchmark Scoring Failures

**Symptoms**: Correct answers marked wrong by LLM scorer

**Solutions**:
- Use clear, unambiguous ground truth answers
- LLM mode is more forgiving than exact match
- Check scorer prompt for clarity

### Challenge 4: Rate Limiting / Budget Exhaustion

**Symptoms**: Many "skipped_budget" status in logs

**Solutions**:
- Increase `cap_total_calls` in config
- Reduce number of models or n_calls
- Check for failed retries consuming budget

### Challenge 5: Streamlit Reruns

**Symptoms**: State lost, widgets reset unexpectedly

**Solutions**:
- Store all mutable state in `st.session_state`
- Use unique keys for all widgets
- Avoid modifying state during render

## Testing the Application

### CLI Testing

```bash
# Single run
python3 -m llm_agg.cli run --config config.json --out ./output

# Benchmark with dataset folder
python3 -m llm_agg.cli bench --config config.json --dataset ./TestData --repeat 5 --out ./benchmark_output
```

### Web UI Testing

```bash
# Start Streamlit
python3 -m streamlit run app_streamlit.py --server.port 8501

# Access at http://localhost:8501
```

### Programmatic Testing

```python
import asyncio
from llm_agg.config import RunConfig, DocInfo, ModelRow
from llm_agg.runner import run_pipeline

config = RunConfig(
    questions=["What is in the document?"],
    docs=[DocInfo(...)],
    doers=[ModelRow(model_id="google/gemini-2.0-flash-001", timeout_s=30, n_calls=1)],
    cap_total_calls=10,
    max_output_tokens=200,
)

results, attempts, scores = asyncio.run(run_pipeline(config, "test_run"))
```

## Output Files

Each run produces:
- `resolved_config.json`: The exact configuration used
- `call_logs.jsonl`: Detailed log of every API call
- `results.json`: Structured results with all outputs
- `stats.json` / `stats.csv`: Aggregated statistics
- `accuracy.json`: Benchmark accuracy (if enabled)

## Dependencies

```
httpx          # Async HTTP client
pydantic       # Data validation
streamlit      # Web UI
PyMuPDF (fitz) # PDF text extraction and rendering
```

## Environment Variables

```
OPENROUTER_API_KEY  # Required for API access
```

## Future Improvements

1. **Multi-page PDF images**: Currently only first page is rendered
2. **PDF OCR**: For scanned PDFs without extractable text
3. **Streaming responses**: Show partial results as they arrive
4. **Model cost estimation**: Pre-run cost prediction
5. **Result caching**: Avoid re-running identical queries
6. **Custom scorer prompts**: User-defined scoring criteria

## Benchmark Results Summary

| Document Type | Encoding | Accuracy | Notes |
|---------------|----------|----------|-------|
| Text (.txt) | utf-8 | 91% | Best for text-heavy content |
| PDF (.pdf) | pdf (text+image) | 82% | Good with dual encoding |
| Image (.jpg) | image (multimodal) | 27% | Limited by visual info only |

**Key insight**: Always prefer text extraction when possible. Use images for layout understanding, not as primary content source.

## Additional Documentation

- **README.md** - Complete usage guide with Quick Start, configuration reference, and benchmark testing guide
- **CLAUDE.md** - Session learnings document with challenges, resolutions, and tips for new developers
- **BENCHMARK_RESULTS.md** - Full benchmark results (100 iterations: Single Model vs ASAP)
- **BENCHMARK_FOCUSED_RESULTS.md** - Focused benchmark results (doc2, Q10-Q11 only)

## Benchmark Scripts

| Script | Purpose |
|--------|---------|
| `benchmark_asap_vs_single.py` | Main benchmark comparing single model vs ASAP pipeline |
| `benchmark_focused.py` | Focused benchmark for specific doc/question combinations |
| `calc_intermediate_stats.py` | Generate reports from benchmark data |
| `calc_focused_stats.py` | Generate reports from focused benchmark data |
| `finish_benchmark.py` | Complete remaining iterations with high concurrency |
