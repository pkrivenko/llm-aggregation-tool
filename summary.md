# LLM Aggregation Tool - Development Summary

## Overview

The LLM Aggregation Tool is a multi-stage pipeline for querying multiple Large Language Models (LLMs) and aggregating their responses. It supports a three-stage architecture: Doers → Judges → Final Judges, where each stage can use different models to generate, evaluate, and synthesize answers.

## Key Features

- **Multi-Stage Pipeline**: Doers → Judges → Final Judges architecture
- **State-of-the-Art Aggregation**: Chain-of-thought, confidence weighting, majority voting, explicit judge selection
- **Multi-Provider Support**: Direct APIs for OpenAI, Google, Anthropic + OpenRouter fallback
- **Multimodal Support**: Text, PDF (multiple modes), and image documents
- **Flexible Configuration**: All settings in `settings.json`
- **Comprehensive Benchmarking**: LLM-based and exact match scoring

## Project Structure

```
Relai/
├── app_streamlit.py              # Web UI (Streamlit)
├── settings.json                 # Centralized configuration (models, prompts, defaults)
├── llm_agg/                      # Core library
│   ├── __init__.py               # Package exports
│   ├── config.py                 # Pydantic models for configuration
│   ├── cli.py                    # Command-line interface
│   ├── runner.py                 # Async pipeline execution
│   ├── openrouter.py             # Multi-provider API client
│   ├── prompts.py                # Prompt builders for each stage
│   ├── aggregation.py            # Voting, confidence, answer extraction
│   ├── stats.py                  # Statistics computation
│   └── io.py                     # File I/O utilities
├── Research/                     # Research reports on aggregation techniques
│   ├── 01_LLM_Ensemble_Methods.md
│   ├── 02_LLM_as_Judge_Evaluation.md
│   ├── 03_Answer_Verification_Reliability.md
│   └── 04_Implementation_Comparison_and_Improvements.md
├── TestData/                     # Test documents and questions
│   ├── doc1.txt                  # Text document
│   ├── doc2.pdf                  # PDF document
│   ├── doc3.jpg                  # Image document
│   └── Questions.md              # Questions with ground truth answers
├── benchmark_asap_vs_single.py   # Main benchmark script
├── benchmark_focused.py          # Focused benchmark script
├── calc_intermediate_stats.py    # Stats calculator
├── calc_focused_stats.py         # Focused stats calculator
├── finish_benchmark.py           # High-concurrency completion
├── runs/                         # Output directory for runs
├── BENCHMARK_RESULTS.md          # Main benchmark results
├── BENCHMARK_FOCUSED_RESULTS.md  # Focused benchmark results
├── REQUIREMENTS_STATUS.md        # Requirements vs implementation status
├── req.md                        # Original requirements specification
└── CLAUDE.md                     # Development learnings and challenges
```

## Implementation Summary

### Core Modules

1. **config.py** - Pydantic models for type-safe configuration:
   - `ModelRow`: Model configuration (model_id, timeout, n_calls, temperature)
   - `DocInfo`: Document metadata and content
   - `RunConfig`: Complete run configuration
   - `BenchmarkConfig`: Benchmarking settings
   - `ScorerConfig`: LLM scorer configuration

2. **openrouter.py** - Multi-provider API client:
   - Automatic routing: OpenAI/Google/Anthropic direct, OpenRouter fallback
   - Format conversion between providers
   - Handles timeouts, errors, and multimodal content

3. **runner.py** - Pipeline execution engine:
   - `BudgetCounter`: Thread-safe call budget management
   - `run_pipeline()`: Main entry point with concurrency control
   - Stage-by-stage execution with configurable data flow

4. **prompts.py** - Message builders:
   - Chain-of-thought prompts for doers
   - Explicit selection prompts for judges
   - Multimodal support for images and PDFs

5. **aggregation.py** - State-of-the-art aggregation:
   - `extract_answer()`: Parse final answer from verbose response
   - `extract_confidence()`: Parse confidence score (1-10)
   - `majority_vote()`: Self-consistency voting with optional confidence weighting
   - `aggregate_judge_selections()`: Count judge votes for doers
   - `compute_agreement_score()`: Measure consensus (0-1)

6. **stats.py** - Statistics computation:
   - Per-stage and per-model statistics
   - Latency, token usage, costs, success rates

7. **cli.py** - Command-line interface:
   - `run` command: Single pipeline run
   - `bench` command: Repeated benchmark runs with aggregation
   - Document loading with format detection
   - PDF handling with multiple modes

8. **app_streamlit.py** - Web UI:
   - Interactive model configuration
   - Document upload (up to 20 files, 20MB each)
   - PDF handling modes and image quality options
   - Real-time progress display
   - Results visualization with aggregation stats

## Architecture Details

### Pipeline Flow

```
Questions × Documents → Doers → Judges (optional) → Final Judges (optional)
                         ↓           ↓                    ↓
                    [responses]  [evaluations]      [final answer]
```

### Multi-Provider API Routing

| Model Pattern | Provider | Endpoint |
|--------------|----------|----------|
| `openai/*`, `gpt-*` | OpenAI (direct) | api.openai.com |
| `google/*`, `gemini*` | Google Generative AI | generativelanguage.googleapis.com |
| `anthropic/*`, `claude-*` | Anthropic (direct) | api.anthropic.com |
| All others | OpenRouter | openrouter.ai |

### Concurrency Model

- `asyncio.Semaphore` limits concurrent API calls (default: 1000)
- `BudgetCounter` with `asyncio.Lock` ensures atomic budget tracking
- All API calls within a stage run in parallel via `asyncio.gather()`

### Document Encodings

| Type | Encoding | Processing |
|------|----------|------------|
| Text (.txt) | UTF-8 | Direct text inclusion |
| PDF (.pdf) | Configurable | Native PDF, text extraction, page images (or combinations) |
| Image (.jpg, .png) | Multimodal | Native image_url format with quality options |
| Binary | base64+zlib | Compressed base64 (limited support) |

### Aggregation Techniques

| Technique | Description | Expected Gain |
|-----------|-------------|---------------|
| Chain-of-Thought | Doers reason step-by-step before answering | +5-10% |
| Confidence Weighting | Responses weighted by self-reported confidence | +2-5% |
| Majority Voting | Most common answer wins (self-consistency) | +5-15% |
| Judge Selection | Explicit selection rather than prose synthesis | +3-8% |
| Heterogeneous Models | Mix of different model families | +3-5% |

## Key Learnings

### 1. Token Efficiency Matters
- Base64 encoding is extremely token-inefficient (~4x bloat)
- Native multimodal format is much more efficient
- Always use `image_url` content type for images, not base64 text in prompts

### 2. PDFs Need Special Handling
- LLMs cannot parse binary PDF data directly
- Multiple modes available: native PDF, text extraction, page images
- Provider-specific: Gemini requires temperature > 0 for PDFs

### 3. Model-Specific Behaviors
| Model | Temperature | PDF Support | Notes |
|-------|-------------|-------------|-------|
| Claude Haiku/Sonnet/Opus | Any | Native | Best PDF accuracy (95%+) |
| Gemini 3 Flash/Pro | **Must be > 0** | Native | Use 0.5-0.7; temp=0 causes issues |
| GPT-5-nano | **Only 1.0** | Native | Don't set temperature |

### 4. Async Concurrency Patterns
- Use `asyncio.Semaphore` for rate limiting
- Use `asyncio.Lock` for shared mutable state
- Use `asyncio.gather()` for parallel execution

## Benchmark Results Summary

| Document Type | Accuracy | Notes |
|---------------|----------|-------|
| Text (.txt) | ~95% | Best for text-heavy content |
| PDF (.pdf) | ~82-95% | Good with native PDF or dual encoding |
| Image (.jpg) | ~27% | Limited by visual info only |

**ASAP vs Single Model:**
- Overall improvement: +1-5% on easy questions
- Difficult questions: +20-40% improvement
- Best results with heterogeneous model ensembles

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
streamlit run app_streamlit.py --server.port 8501

# Access at http://localhost:8501
```

### Programmatic Testing

```python
import asyncio
from llm_agg.config import RunConfig, DocInfo, ModelRow
from llm_agg.runner import run_pipeline
from llm_agg import majority_vote, compute_agreement_score

config = RunConfig(
    questions=["What is in the document?"],
    docs=[doc],
    doers=[ModelRow(model_id="google/gemini-3-flash-preview", timeout_s=30, n_calls=5, temperature=0.7)],
    cap_total_calls=100,
    max_output_tokens=500,
)

results, attempts, scores = asyncio.run(run_pipeline(config, "test_run"))

# Access aggregation statistics
for result in results:
    print(f"Vote winner: {result['aggregation']['doer_vote']['winner']}")
    print(f"Agreement: {result['aggregation']['doer_agreement']:.0%}")
```

## Output Files

Each run produces:
- `resolved_config.json`: The exact configuration used (reproducible)
- `call_logs.jsonl`: Detailed log of every API call
- `results.json`: Structured results with all outputs and aggregation stats
- `stats.json` / `stats.csv`: Aggregated statistics
- `accuracy.json`: Benchmark accuracy (if enabled)

## Dependencies

```
httpx          # Async HTTP client
pydantic       # Data validation
streamlit      # Web UI
PyMuPDF (fitz) # PDF text extraction and rendering
Pillow         # Image resizing/quality
google-genai   # Google Generative AI SDK
openai         # OpenAI SDK
anthropic      # Anthropic SDK
python-dotenv  # Environment loading
```

## Environment Variables

```
OPENAI_API_KEY     # For OpenAI models (direct API)
GOOGLE_API_KEY     # For Google models (direct API)
ANTHROPIC_API_KEY  # For Claude models (direct API)
OPENROUTER_API_KEY # For other models (OpenRouter fallback)
```

Keys can also be stored in files: `OpenAIAPIKey.txt`, `GoogleAPIKey.txt`, `AnthropicAPIKey.txt`, `OpenRouterAPIKey.txt`

## Additional Documentation

- **README.md** - Complete usage guide with Quick Start, configuration reference, and benchmark testing guide
- **CLAUDE.md** - Development learnings with challenges, resolutions, and tips for new developers
- **REQUIREMENTS_STATUS.md** - Detailed requirements vs implementation status
- **BENCHMARK_RESULTS.md** - Full benchmark results (100 iterations: Single Model vs ASAP)
- **BENCHMARK_FOCUSED_RESULTS.md** - Focused benchmark results (doc2, Q10-Q11 only)
- **Research/** - Comprehensive research reports on LLM aggregation techniques
