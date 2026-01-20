# LLM Aggregation Tool

A multi-stage pipeline for querying multiple Large Language Models (LLMs) and aggregating their responses. Supports a three-stage architecture: **Doers → Judges → Final Judges**, where each stage can use different models to generate, evaluate, and synthesize answers.

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install streamlit httpx pydantic python-dotenv PyMuPDF Pillow
```

> **Note:** `PyMuPDF` (fitz) is required for PDF text extraction and page rendering. `Pillow` is required for image resizing/quality options.

### API Key Setup (Secure)

The tool supports four API providers with automatic routing:
- **OpenAI** (direct) - for `openai/*` and `gpt-*` models
- **Google Generative AI** (direct) - for `google/*` and `gemini*` models
- **Anthropic** (direct) - for `anthropic/*` and `claude-*` models
- **OpenRouter** - for all other models (fallback)

**Option 1: Environment Variables (Recommended for production)**

```bash
# For OpenAI models (direct API)
export OPENAI_API_KEY="your-openai-key"

# For Google models (direct API)
export GOOGLE_API_KEY="your-google-api-key"

# For Anthropic/Claude models (direct API)
export ANTHROPIC_API_KEY="your-anthropic-key"

# For OpenRouter (other models)
export OPENROUTER_API_KEY="your-openrouter-key"
```

**Option 2: Key Files (for local development)**

Create these files in the project root (they are git-ignored):

```bash
# For OpenAI API
echo "your-openai-key" > OpenAIAPIKey.txt

# For Google API
echo "your-google-api-key" > GoogleAPIKey.txt

# For Anthropic API
echo "your-anthropic-key" > AnthropicAPIKey.txt

# For OpenRouter
echo "your-openrouter-key" > OpenRouterAPIKey.txt
```

**Important:** API key files are in `.gitignore` and will NOT be pushed to GitHub. Never commit API keys to version control.

### Running the Web UI

```bash
streamlit run app_streamlit.py --server.port 8501
# Access at http://localhost:8501
```

### Running via CLI

```bash
# Single run with config file
python3 -m llm_agg.cli run --config config.json --out ./output

# Benchmark with test dataset
python3 -m llm_agg.cli bench --config config.json --dataset ./TestData --repeat 5 --out ./benchmark_output
```

### Running Programmatically

```python
import asyncio
from llm_agg.config import RunConfig, DocInfo, ModelRow, BenchmarkConfig, ScorerConfig
from llm_agg.runner import run_pipeline
from llm_agg.cli import _load_doc, _parse_questions_md

# Load documents and questions
doc = _load_doc("TestData/doc1.txt", "doc1")
questions, ground_truths = _parse_questions_md("TestData/Questions.md")

# Configure the pipeline
config = RunConfig(
    questions=questions,
    docs=[doc],
    doers=[ModelRow(model_id="google/gemini-2.0-flash-001", timeout_s=60, n_calls=1, temperature=0.3)],
    judges=[],  # Optional: add judge models
    final_judges=[],  # Optional: add final judge models
    cap_total_calls=100,
    max_output_tokens=300,
    retries=1,
    max_concurrency=20,
    benchmark=BenchmarkConfig(
        enabled=True,
        mode="llm",
        strip_whitespace=True,
        scorer=ScorerConfig(model_id="google/gemini-2.0-flash-001", timeout_s=30, temperature=0.0),
        ground_truths=ground_truths,
    ),
)

# Run the pipeline
results, attempts, scores = asyncio.run(run_pipeline(config, "my_run"))
```

---

## Architecture

### Pipeline Flow

```
Questions × Documents → Doers → Judges (optional) → Final Judges (optional)
                         ↓           ↓                    ↓
                    [responses]  [evaluations]      [final answer]
```

### Three-Stage Design

1. **Doers** (Stage 1): Generate initial answers to questions
   - Multiple models can be used in parallel
   - Each model can make multiple calls (n_calls)
   - Higher temperature = more diverse responses

2. **Judges** (Stage 2, Optional): Evaluate doer responses
   - Receive all doer outputs for comparison
   - Optionally receive the original document
   - Identify best answers and point out errors

3. **Final Judges** (Stage 3, Optional): Synthesize final answer
   - Receive doer outputs, judge evaluations, and optionally the document
   - Produce a single, best final answer

### Data Flow Options

Control what each stage receives via configuration:

| Option | Default | Description |
|--------|---------|-------------|
| `send_doc_to_judges` | false | Include document in judge prompts |
| `send_doc_to_final_judges` | false | Include document in final judge prompts |
| `send_doer_responses_to_judges` | true | Include doer outputs in judge prompts |
| `send_doer_outputs_to_final_judges` | true | Include doer outputs in final judge prompts |
| `send_judge_outputs_to_final_judges` | true | Include judge outputs in final judge prompts |

---

## Configuration Reference

### ModelRow Configuration

```python
ModelRow(
    model_id="google/gemini-2.0-flash-001",  # OpenRouter model ID
    timeout_s=60,                             # Request timeout in seconds
    n_calls=5,                                # Number of parallel calls
    temperature=0.7,                          # Optional: 0.0-2.0, omit for model default
)
```

### RunConfig Parameters

| Parameter | Code Default | UI Default | Description |
|-----------|--------------|------------|-------------|
| `questions` | required | - | List of questions to answer |
| `docs` | `[]` | - | List of DocInfo objects (max 20) |
| `doers` | `[]` | 1 row | List of ModelRow for doer stage (max 10) |
| `judges` | `[]` | empty | List of ModelRow for judge stage (max 10) |
| `final_judges` | `[]` | empty | List of ModelRow for final stage (max 10) |
| `cap_total_calls` | 100 | 100 | Maximum total API calls (hard budget) |
| `max_output_tokens` | 200 | 1000 | Max tokens per response |
| `retries` | 0 | 0 | Retries on non-timeout errors |
| `max_concurrency` | 1000 | 1000 | Max parallel API requests |
| `debug_mode` | false | false | Enable detailed logging |

> **Note:** UI defaults are configured in `settings.json` and may differ from code defaults in `config.py`.

### BenchmarkConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | false | Enable benchmarking |
| `mode` | "exact" | Scoring mode: "exact" or "llm" |
| `strip_whitespace` | true | Strip whitespace before comparison |
| `ground_truths` | `[]` | List of expected answers |
| `scorer` | - | ScorerConfig for LLM-based scoring |

---

## Benchmarking

### Overview

The tool includes comprehensive benchmarking capabilities to compare different configurations and measure accuracy against ground truth answers.

### Scoring Modes

1. **Exact Match** (`mode="exact"`): Simple string comparison
   - Fast, no additional API calls
   - Use `strip_whitespace=True` for flexible matching

2. **LLM Scorer** (`mode="llm"`): AI-based semantic comparison
   - More forgiving of phrasing differences
   - Requires additional API calls (counts toward budget)
   - Recommended for natural language answers

### Running Benchmarks

See [Benchmark Testing Guide](#benchmark-testing-guide) below for detailed instructions.

---

## Benchmark Testing Guide

This section documents the benchmark scripts created for comparing Single Model vs ASAP (multi-stage aggregation) approaches.

### Test Data Structure

```
TestData/
├── doc1.txt      # Text document (plain text)
├── doc2.pdf      # PDF document (text + visual)
├── doc3.jpg      # Image document (visual only)
└── Questions.md  # 11 questions with ground truth answers
```

### Benchmark Scripts

#### 1. Main Benchmark: `benchmark_asap_vs_single.py`

Compares single model performance vs full ASAP pipeline.

**Configuration:**

| Parameter | Single Model | ASAP |
|-----------|-------------|------|
| Doer calls | 1 | 10 |
| Doer temperature | 0.3 | 0.7 |
| Judge calls | - | 5 |
| Judge temperature | - | 0.3 |
| Final judge calls | - | 1 |
| Final temperature | - | 0.0 |

**Usage:**

```bash
# Run full benchmark (100 iterations each)
python3 benchmark_asap_vs_single.py

# Output files:
# - benchmark_detailed_data.jsonl  (all responses)
# - BENCHMARK_RESULTS.md           (summary report)
```

#### 2. Stats Calculator: `calc_intermediate_stats.py`

Generate reports from benchmark data (can run while benchmark is in progress).

```bash
python3 calc_intermediate_stats.py
# Outputs: intermediate_benchmark.md
```

#### 3. High-Concurrency Completion: `finish_benchmark.py`

Complete remaining iterations with maximum parallelization.

```bash
python3 finish_benchmark.py
# Uses max_concurrency=2000 for fast completion
```

#### 4. Focused Benchmark: `benchmark_focused.py`

Test specific document/question combinations with custom parameters.

**Configuration:**

| Parameter | Single Model | ASAP |
|-----------|-------------|------|
| Document | doc2.pdf only | doc2.pdf only |
| Questions | Q10, Q11 only | Q10, Q11 only |
| Doer calls | 1 | 20 |
| Doer temperature | 0.0 | 0.7 |
| Judge calls | - | 7 |
| Max concurrency | 1000 | 1000 |

**Usage:**

```bash
python3 benchmark_focused.py

# Output files:
# - benchmark_focused_data.jsonl
# - BENCHMARK_FOCUSED_RESULTS.md
```

#### 5. Focused Stats Calculator: `calc_focused_stats.py`

```bash
python3 calc_focused_stats.py
# Outputs: BENCHMARK_FOCUSED_RESULTS.md
```

### Creating Custom Benchmarks

To create a new benchmark test:

```python
#!/usr/bin/env python3
import asyncio
import json
from datetime import datetime
from pathlib import Path

from llm_agg.config import RunConfig, ModelRow, BenchmarkConfig, ScorerConfig
from llm_agg.runner import run_pipeline
from llm_agg.cli import _load_doc, _parse_questions_md

GEMINI_MODEL = "google/gemini-2.0-flash-001"
OUTPUT_FILE = Path("my_benchmark_data.jsonl")

def load_test_data():
    """Load your test documents and questions."""
    test_data_path = Path("TestData")

    # Load documents
    docs = []
    for doc_file in ["doc1.txt", "doc2.pdf"]:
        doc_path = test_data_path / doc_file
        if doc_path.exists():
            doc = _load_doc(str(doc_path), doc_path.stem)
            docs.append(doc)

    # Load questions
    questions_path = test_data_path / "Questions.md"
    questions, ground_truths = _parse_questions_md(str(questions_path))

    return docs, questions, ground_truths

def create_config(docs, questions, ground_truths):
    """Create your benchmark configuration."""
    return RunConfig(
        questions=questions,
        docs=docs,
        doers=[ModelRow(
            model_id=GEMINI_MODEL,
            timeout_s=60,
            n_calls=5,        # Customize
            temperature=0.5   # Customize
        )],
        judges=[ModelRow(
            model_id=GEMINI_MODEL,
            timeout_s=60,
            n_calls=3,        # Customize
            temperature=0.3
        )],
        final_judges=[ModelRow(
            model_id=GEMINI_MODEL,
            timeout_s=60,
            n_calls=1,
            temperature=0.0
        )],
        send_doc_to_judges=True,
        send_doc_to_final_judges=True,
        send_doer_responses_to_judges=True,
        send_doer_outputs_to_final_judges=True,
        send_judge_outputs_to_final_judges=True,
        cap_total_calls=2000,
        max_output_tokens=300,
        retries=1,
        max_concurrency=100,  # Adjust based on your needs
        benchmark=BenchmarkConfig(
            enabled=True,
            mode="llm",
            strip_whitespace=True,
            scorer=ScorerConfig(
                model_id=GEMINI_MODEL,
                timeout_s=30,
                temperature=0.0
            ),
            ground_truths=ground_truths,
        ),
    )

def save_record(benchmark_type, iteration, results, scores, questions, ground_truths):
    """Save detailed record to JSONL."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_type": benchmark_type,
        "iteration": iteration,
        "scores": scores,
        "results": [...]  # Add result details
    }
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

async def run_iteration(config, run_id, iteration, questions, ground_truths):
    """Run single benchmark iteration."""
    results, attempts, scores = await run_pipeline(config, run_id)
    save_record("MyBenchmark", iteration, results, scores, questions, ground_truths)
    return scores

async def main():
    docs, questions, ground_truths = load_test_data()

    # Run iterations in parallel
    tasks = []
    for i in range(100):  # Number of iterations
        config = create_config(docs, questions, ground_truths)
        run_id = f"bench_{i:03d}"
        tasks.append(run_iteration(config, run_id, i, questions, ground_truths))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    success = sum(1 for r in results if not isinstance(r, Exception))
    print(f"Completed: {success} successful")

if __name__ == "__main__":
    asyncio.run(main())
```

### Benchmark Results Summary

From our testing (100 iterations each):

**Full Benchmark (all docs, all questions):**

| Configuration | Success Rate |
|--------------|-------------|
| Single Model (temp=0.3) | 72.7% |
| ASAP (10 doers, 5 judges) | 74.0% |
| **Improvement** | **+1.3%** |

**Focused Benchmark (doc2, Q10-Q11 only):**

| Configuration | Success Rate |
|--------------|-------------|
| Single Model (temp=0) | 50.0% |
| ASAP (20 doers, 7 judges) | 73.0% |
| **Improvement** | **+23.0%** |

Key finding: ASAP shows significant improvement on difficult questions where single model struggles.

---

## Output Files

Each run produces:

| File | Description |
|------|-------------|
| `resolved_config.json` | Exact configuration used (reproducible) |
| `call_logs.jsonl` | Detailed log of every API call |
| `results.json` | Structured results with all outputs |
| `stats.json` / `stats.csv` | Aggregated statistics |
| `accuracy.json` | Benchmark accuracy (if enabled) |

---

## Document Support

### Supported Formats

| Type | Encoding | How Processed |
|------|----------|---------------|
| Text (.txt) | UTF-8 | Direct text inclusion |
| PDF (.pdf) | Configurable | Multiple modes available (see below) |
| Image (.jpg, .png) | Multimodal | Native image format with quality options |
| Binary | base64+zlib | Compressed base64 (limited support) |

### PDF Handling Modes

When uploading PDF files, you can choose how to send them to the LLM. Select one or more modes:

| Mode | Description | Best For |
|------|-------------|----------|
| **Send PDF as-is** (default) | Native PDF upload via base64 | Models with PDF support (Claude, Gemini) |
| **Extract text only** | Extract text using PyMuPDF | Text-heavy PDFs, fastest processing |
| **Send as images** | Render all pages as PNG images | Visual content, charts, diagrams |

**Combining modes:** You can select multiple modes to send different representations of the same PDF. This can improve accuracy when a document has both text and visual elements.

**Provider support for native PDF:**

| Provider | Native PDF Support | Notes |
|----------|-------------------|-------|
| Anthropic (Claude) | ✅ Full support | Best accuracy with native PDF |
| Google (Gemini) | ✅ Full support | Requires temperature > 0 |
| OpenAI (GPT) | ✅ Full support | Uses Files API format internally |

### Image Quality Options

For both regular images and PDF-to-image conversion, you can select the output quality:

| Quality | Max Dimension | Use Case |
|---------|---------------|----------|
| **Original** | No resize | Maximum fidelity, highest token cost |
| **Maximum (2048px)** | 2048px | High detail needs |
| **High (1536px)** | 1536px | Good balance of detail and cost |
| **Standard (1024px)** | 1024px | General use |
| **Draft (512px)** | 512px | Fast processing, lower cost |

Images are resized proportionally (preserving aspect ratio) using high-quality LANCZOS resampling.

**Benchmark findings:**
- For text-heavy PDFs, lower resolutions (512-1024px) often perform as well as higher ones
- Visual content (charts, diagrams) benefits from higher resolutions
- Draft quality can be 2x faster with similar accuracy on text documents

### Document Size Limit

Maximum file size: **20 MB** per file, up to **20 files** total (configurable in settings.json)

### Combine Documents Feature

When uploading multiple files (2+), you can enable **"Combine all documents into single context"** to merge all documents into one combined context for each question.

**How it works:**
- All uploaded files are merged with clear separators (`=== filename ===`)
- The pipeline runs once with the combined document instead of separately per file
- Useful for questions that require information from multiple sources

**Example combined output:**
```
=== sales_q1.txt ===
Q1 Sales Report:
- January: $10,000
- February: $12,000
...

=== sales_q2.txt ===
Q2 Sales Report:
- April: $14,000
...
```

**Use cases:**
- Cross-document analysis ("What is the total across all reports?")
- Comparing information from multiple sources
- Synthesizing data from related documents

**Dry run estimate:** When combining is enabled, D (document count) becomes 1 regardless of how many files are uploaded, reducing total API calls.

---

## Troubleshooting

### Common Issues

1. **"Context length exceeded" error**
   - Reduce document size or use models with larger context
   - For images, ensure using multimodal format (automatic)

2. **PDF answers are wrong/hallucinated**
   - Try different PDF modes: "Send as-is" works best for most models
   - For Gemini models, ensure temperature > 0 (0.7 recommended)
   - Check if PDF has extractable text (not scanned image)

3. **Budget exhausted (skipped_budget status)**
   - Increase `cap_total_calls`
   - Reduce number of models or n_calls

4. **Rate limiting / 429 errors**
   - Reduce `max_concurrency`
   - Add retries with backoff

5. **"Unsupported value: temperature" error (GPT-5-nano)**
   - GPT-5-nano only supports temperature=1.0
   - Leave temperature field empty for this model

### Model-Specific Notes

| Model | Temperature | PDF Support | Notes |
|-------|-------------|-------------|-------|
| Claude Haiku 4.5 | Any | Native | Best PDF accuracy (95%+) |
| Claude Sonnet/Opus | Any | Native | Excellent accuracy |
| Gemini 3 Flash/Pro | **Must be > 0** | Native | Use 0.7; temp=0 causes issues with PDFs |
| GPT-5-nano | **Only 1.0** | Native | Don't set temperature; limited accuracy |
| GPT-5-mini/5.2 | Any | Native | Good accuracy |

**Benchmark results (11-question PDF test):**

| Model | Text Mode | PDF Native | Images (1024px) |
|-------|-----------|------------|-----------------|
| Claude Haiku 4.5 | 96% | 95% | 75% |
| Gemini 3 Flash | 91% | 91% | 90% |
| GPT-5-nano | N/A | 64% | 22-58% |

### Debug Mode

Enable debug mode for detailed logging:

```python
config = RunConfig(
    ...
    debug_mode=True,
)
```

In the Streamlit UI, check "Debug mode" to see all request/response details.

---

## Project Structure

```
CAI/
├── app_streamlit.py              # Web UI (Streamlit)
├── settings.json                 # All app settings (models, prompts, defaults)
├── llm_agg/                      # Core library
│   ├── __init__.py
│   ├── config.py                 # Pydantic models
│   ├── cli.py                    # Command-line interface
│   ├── runner.py                 # Pipeline execution
│   ├── openrouter.py             # API client (multi-provider)
│   ├── prompts.py                # Message builders
│   ├── stats.py                  # Statistics
│   └── io.py                     # File I/O
├── TestData/                     # Test documents
│   ├── doc1.txt
│   ├── doc2.pdf
│   ├── doc3.jpg
│   └── Questions.md
├── benchmark_asap_vs_single.py   # Main benchmark script
├── benchmark_focused.py          # Focused benchmark script
├── calc_intermediate_stats.py    # Stats calculator
├── calc_focused_stats.py         # Focused stats calculator
├── finish_benchmark.py           # High-concurrency completion
├── BENCHMARK_RESULTS.md          # Main benchmark results
├── BENCHMARK_FOCUSED_RESULTS.md  # Focused benchmark results
└── runs/                         # Output directory
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For OpenAI models | Your OpenAI API key |
| `GOOGLE_API_KEY` | For Google models | Your Google Generative AI API key |
| `ANTHROPIC_API_KEY` | For Claude models | Your Anthropic API key |
| `OPENROUTER_API_KEY` | For other models | Your OpenRouter API key |

**API Routing:**
- Models starting with `openai/` or `gpt-` → OpenAI API (direct)
- Models starting with `google/` or `gemini` → Google API (direct)
- Models starting with `anthropic/` or `claude-` → Anthropic API (direct)
- All other models → OpenRouter

**Alternative:** Place keys in `OpenAIAPIKey.txt`, `GoogleAPIKey.txt`, `AnthropicAPIKey.txt`, or `OpenRouterAPIKey.txt` in project root (git-ignored).

---

## Settings File

All app configuration is stored in `settings.json`:

```json
{
  "models": [                    // Available models with max_tokens
    {"id": "openai/gpt-5-nano", "name": "GPT-5 Nano", "max_tokens": 100000},
    {"id": "anthropic/claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "max_tokens": 64000},
    ...
  ],
  "system_prompts": {            // Default prompts (editable in UI)
    "doer": "...",
    "judge": "...",
    "final": "...",
    "scorer": "..."
  },
  "default_model": "openai/gpt-5-nano",
  "model_defaults": {            // Defaults for new model rows
    "timeout_s": 30.0,
    "n_calls": 1,
    "temperature": null
  },
  "global_controls": {           // App-wide settings
    "cap_total_calls": 100,
    "max_output_tokens": 1000,
    "retries": 0,
    "max_concurrency": 1000,
    "debug_mode": false
  },
  "pipeline_options": {...},     // What data to pass between stages
  "benchmark_defaults": {...},   // Benchmark configuration
  "ui": {...}                    // UI limits
}
```

Edit this file directly to customize defaults, add models, or modify system prompts.

---

## Available Models

The following models are pre-configured in `settings.json`:

| Model ID | Display Name | Max Tokens | Default Temp |
|----------|--------------|------------|--------------|
| `openai/gpt-5-nano` | GPT-5 Nano | 100,000 | - |
| `openai/gpt-5-mini` | GPT-5 Mini | 100,000 | - |
| `openai/gpt-5.2` | GPT-5.2 | 100,000 | - |
| `google/gemini-3-flash-preview` | Gemini 3 Flash Preview | 64,000 | 0.7 |
| `google/gemini-3-pro-preview` | Gemini 3 Pro Preview | 64,000 | 0.7 |
| `anthropic/claude-haiku-4-5-20251001` | Claude Haiku 4.5 | 64,000 | - |
| `anthropic/claude-sonnet-4-5-20250929` | Claude Sonnet 4.5 | 64,000 | - |
| `anthropic/claude-opus-4-5-20251101` | Claude Opus 4.5 | 64,000 | - |

You can also enter custom model IDs directly in the UI.

---

## Documentation

| Document | Description |
|----------|-------------|
| `README.md` | This file - usage guide and reference |
| `CLAUDE.md` | Development session learnings and challenges |
| `REQUIREMENTS_STATUS.md` | Detailed requirements vs implementation status |
| `summary.md` | Development history and architecture overview |

---

## License

Internal project - not for public distribution.
