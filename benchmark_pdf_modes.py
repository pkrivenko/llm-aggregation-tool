"""Benchmark PDF modes with GPT-5-nano - 10 iterations each, parallel execution"""
import asyncio
import base64
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from llm_agg.config import RunConfig, ModelRow, DocInfo
from llm_agg.runner import run_pipeline
from llm_agg.io import generate_run_id

PDF_PATH = Path("TestData/doc2.pdf")
QUESTION = "Respond with full name of the candidate. It consists of two words."
EXPECTED = "Pavel Krivenko"
ITERATIONS = 10

# Pre-load PDF content
PDF_BYTES = PDF_PATH.read_bytes()

def load_pdf_with_modes(pdf_modes: list) -> DocInfo:
    """Load PDF with specified modes."""
    pdf_raw = None
    pdf_text = None
    pdf_pages = None
    
    if "raw" in pdf_modes:
        pdf_raw = base64.b64encode(PDF_BYTES).decode("ascii")
    
    if "text" in pdf_modes:
        import fitz
        doc = fitz.open(stream=PDF_BYTES, filetype="pdf")
        text_parts = [page.get_text() for page in doc]
        pdf_text = "\n\n".join(text_parts)
        doc.close()
    
    if "images" in pdf_modes:
        import fitz
        doc = fitz.open(stream=PDF_BYTES, filetype="pdf")
        pdf_pages = []
        mat = fitz.Matrix(2, 2)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            pdf_pages.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
        doc.close()
    
    return DocInfo(
        doc_id="doc2.pdf",
        filename="doc2.pdf",
        mime="application/pdf",
        encoding="pdf",
        content="",
        pdf_raw=pdf_raw,
        pdf_text=pdf_text,
        pdf_pages=pdf_pages,
    )

@dataclass
class RunResult:
    mode_name: str
    iteration: int
    success: bool
    correct: bool
    latency_ms: float
    answer: Optional[str]
    error: Optional[str]

async def run_single(mode_name: str, pdf_modes: list, iteration: int) -> RunResult:
    """Run a single iteration."""
    doc = load_pdf_with_modes(pdf_modes)
    
    config = RunConfig(
        questions=[QUESTION],
        docs=[doc],
        doers=[ModelRow(model_id="openai/gpt-5-nano", timeout_s=60.0, n_calls=1)],
        judges=[],
        final_judges=[],
        doer_system_prompt="You are a helpful assistant. Answer based on the document provided.",
        cap_total_calls=10,
        max_output_tokens=500,
        max_concurrency=10,
    )
    
    run_id = generate_run_id()
    latency_ms = 0
    error_msg = None
    
    def on_attempt(record):
        nonlocal latency_ms, error_msg
        latency_ms = record.get("latency_ms", 0)
        if record.get("error_message"):
            error_msg = record["error_message"][:100]
    
    try:
        results, _, _ = await run_pipeline(config, run_id, on_attempt)
        
        if results and results[0]["primary_outputs"]:
            answer = results[0]["primary_outputs"][0].get("text")
            if answer:
                correct = EXPECTED.lower() in answer.lower()
                return RunResult(mode_name, iteration, True, correct, latency_ms, answer, None)
        
        return RunResult(mode_name, iteration, False, False, latency_ms, None, error_msg or "No answer")
    except Exception as e:
        return RunResult(mode_name, iteration, False, False, latency_ms, None, str(e)[:100])

async def main():
    print("=" * 70)
    print("PDF Modes Benchmark - GPT-5-nano")
    print(f"Iterations per mode: {ITERATIONS}")
    print("=" * 70)
    
    # All 7 non-empty combinations
    test_cases = [
        (["raw"], "Raw only"),
        (["text"], "Text only"),
        (["images"], "Images only"),
        (["raw", "text"], "Raw+Text"),
        (["raw", "images"], "Raw+Images"),
        (["text", "images"], "Text+Images"),
        (["raw", "text", "images"], "All three"),
    ]
    
    # Create all tasks
    all_tasks = []
    for modes, name in test_cases:
        for i in range(ITERATIONS):
            all_tasks.append(run_single(name, modes, i))
    
    print(f"\nRunning {len(all_tasks)} total calls in parallel...")
    start_time = time.time()
    
    # Run all in parallel
    results = await asyncio.gather(*all_tasks, return_exceptions=True)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")
    
    # Process results by mode
    mode_stats = {}
    for r in results:
        if isinstance(r, Exception):
            continue
        if r.mode_name not in mode_stats:
            mode_stats[r.mode_name] = {"success": 0, "correct": 0, "total": 0, "latencies": [], "errors": []}
        
        stats = mode_stats[r.mode_name]
        stats["total"] += 1
        if r.success:
            stats["success"] += 1
            stats["latencies"].append(r.latency_ms)
        if r.correct:
            stats["correct"] += 1
        if r.error:
            stats["errors"].append(r.error)
    
    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Mode':<15} {'Success':>8} {'Correct':>8} {'Accuracy':>10} {'Avg Latency':>12}")
    print("-" * 70)
    
    for modes, name in test_cases:
        stats = mode_stats.get(name, {"success": 0, "correct": 0, "total": 0, "latencies": []})
        success_rate = stats["success"] / stats["total"] * 100 if stats["total"] else 0
        accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] else 0
        avg_latency = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0
        
        print(f"{name:<15} {stats['success']:>5}/{stats['total']:<2} {stats['correct']:>5}/{stats['total']:<2} {accuracy:>9.0f}% {avg_latency:>10.0f}ms")
    
    print("-" * 70)
    
    # Note about Raw only
    raw_stats = mode_stats.get("Raw only", {})
    if raw_stats.get("errors"):
        print(f"\nNote: 'Raw only' fails because OpenAI doesn't support PDF document type.")
        print("      Use 'Text only' or 'Images only' for OpenAI, or use Anthropic for native PDF.")

if __name__ == "__main__":
    asyncio.run(main())
