"""Comprehensive benchmark: 11 questions × 2 models × 3 quality levels × 10 iterations"""
import asyncio
import base64
import time
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from llm_agg.config import RunConfig, ModelRow, DocInfo
from llm_agg.runner import run_pipeline
from llm_agg.io import generate_run_id

PDF_PATH = Path("TestData/doc2.pdf")
PDF_BYTES = PDF_PATH.read_bytes()
ITERATIONS = 10

# Parse questions and answers
QUESTIONS = []
ANSWERS = []
questions_text = Path("TestData/Questions.md").read_text()
for match in re.finditer(r"Q(\d+):\s*(.+?)(?=\n)", questions_text):
    QUESTIONS.append(match.group(2).strip())
for match in re.finditer(r"A(\d+):\s*(.+?)(?=\n)", questions_text):
    ANSWERS.append(match.group(2).strip())

MODELS = [
    "openai/gpt-5-nano",
    "google/gemini-3-flash-preview",
]

QUALITY_LEVELS = [
    ("Draft (512px)", 512),
    ("Standard (1024px)", 1024),
    ("High (1536px)", 1536),
]

def resize_image(img_bytes: bytes, max_dim: int) -> bytes:
    """Resize image to max dimension."""
    from PIL import Image
    import io
    
    img = Image.open(io.BytesIO(img_bytes))
    w, h = img.size
    if max(w, h) > max_dim:
        if w > h:
            new_w, new_h = max_dim, int(h * max_dim / w)
        else:
            new_h, new_w = max_dim, int(w * max_dim / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def load_pdf_as_images(max_dim: int) -> DocInfo:
    """Load PDF as images with specified max dimension."""
    import fitz
    
    doc = fitz.open(stream=PDF_BYTES, filetype="pdf")
    pdf_pages = []
    mat = fitz.Matrix(3, 3)  # High res base
    
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        if max_dim > 0:
            img_bytes = resize_image(img_bytes, max_dim)
        pdf_pages.append(base64.b64encode(img_bytes).decode("ascii"))
    doc.close()
    
    return DocInfo(
        doc_id="doc2.pdf",
        filename="doc2.pdf",
        mime="application/pdf",
        encoding="pdf",
        content="",
        pdf_pages=pdf_pages,
    )

@dataclass
class RunResult:
    model: str
    quality: str
    question_idx: int
    iteration: int
    success: bool
    correct: bool
    latency_ms: float
    answer: Optional[str]
    error: Optional[str]

def check_answer(answer: str, expected: str, q_idx: int) -> bool:
    """Check if answer matches expected."""
    if not answer:
        return False
    answer_lower = answer.lower().strip()
    expected_lower = expected.lower().strip()
    
    if expected_lower in answer_lower:
        return True
    
    if q_idx in [1, 9]:
        answer_nums = re.findall(r'\d+', answer)
        if answer_nums and expected_lower in answer_nums:
            return True
    
    return False

async def run_single(model: str, quality: str, doc: DocInfo, q_idx: int, iteration: int) -> RunResult:
    """Run a single test."""
    question = QUESTIONS[q_idx]
    expected = ANSWERS[q_idx]
    
    config = RunConfig(
        questions=[question],
        docs=[doc],
        doers=[ModelRow(model_id=model, timeout_s=90.0, n_calls=1)],
        judges=[],
        final_judges=[],
        doer_system_prompt="Answer based on the document. Be concise.",
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
                correct = check_answer(answer, expected, q_idx)
                return RunResult(model, quality, q_idx, iteration, True, correct, latency_ms, answer, None)
        
        return RunResult(model, quality, q_idx, iteration, False, False, latency_ms, None, error_msg or "No answer")
    except Exception as e:
        return RunResult(model, quality, q_idx, iteration, False, False, latency_ms, None, str(e)[:100])

async def main():
    total_calls = len(QUESTIONS) * len(MODELS) * len(QUALITY_LEVELS) * ITERATIONS
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK")
    print(f"Questions: {len(QUESTIONS)} | Models: {len(MODELS)} | Quality levels: {len(QUALITY_LEVELS)} | Iterations: {ITERATIONS}")
    print(f"Total API calls: {total_calls}")
    print("=" * 80)
    
    # PRE-COMPUTE images for each quality level (only 3 times, not 660!)
    print("\nPre-rendering PDF at each quality level...")
    docs_by_quality = {}
    for quality, max_dim in QUALITY_LEVELS:
        start = time.time()
        docs_by_quality[quality] = load_pdf_as_images(max_dim)
        print(f"  {quality}: {time.time() - start:.1f}s")
    
    # Create all tasks using pre-computed docs
    all_tasks = []
    for model in MODELS:
        for quality, max_dim in QUALITY_LEVELS:
            doc = docs_by_quality[quality]  # Reuse pre-computed doc
            for q_idx in range(len(QUESTIONS)):
                for i in range(ITERATIONS):
                    all_tasks.append(run_single(model, quality, doc, q_idx, i))
    
    print(f"\nRunning {len(all_tasks)} API calls in parallel...")
    start_time = time.time()
    
    results = await asyncio.gather(*all_tasks, return_exceptions=True)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s ({len(all_tasks)/elapsed:.1f} calls/sec)")
    
    # Aggregate results
    stats = {}
    for r in results:
        if isinstance(r, Exception):
            continue
        key = (r.model.split("/")[-1], r.quality)
        if key not in stats:
            stats[key] = {"correct": 0, "total": 0, "success": 0, "latencies": []}
        stats[key]["total"] += 1
        if r.success:
            stats[key]["success"] += 1
            stats[key]["latencies"].append(r.latency_ms)
        if r.correct:
            stats[key]["correct"] += 1
    
    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS BY MODEL AND QUALITY")
    print("=" * 80)
    print(f"{'Model':<25} {'Quality':<18} {'Success':>10} {'Accuracy':>10} {'Avg Latency':>12}")
    print("-" * 80)
    
    for model in MODELS:
        model_name = model.split("/")[-1]
        for quality, _ in QUALITY_LEVELS:
            key = (model_name, quality)
            s = stats.get(key, {"correct": 0, "total": 0, "success": 0, "latencies": []})
            accuracy = s["correct"] / s["total"] * 100 if s["total"] else 0
            avg_lat = sum(s["latencies"]) / len(s["latencies"]) if s["latencies"] else 0
            print(f"{model_name:<25} {quality:<18} {s['success']:>4}/{s['total']:<4} {accuracy:>9.1f}% {avg_lat:>10.0f}ms")
        print("-" * 80)
    
    # Per-question breakdown
    print("\n" + "=" * 80)
    print("ACCURACY BY QUESTION")
    print("=" * 80)
    
    q_stats = {}
    for r in results:
        if isinstance(r, Exception):
            continue
        if r.question_idx not in q_stats:
            q_stats[r.question_idx] = {"correct": 0, "total": 0}
        q_stats[r.question_idx]["total"] += 1
        if r.correct:
            q_stats[r.question_idx]["correct"] += 1
    
    for q_idx in range(len(QUESTIONS)):
        s = q_stats.get(q_idx, {"correct": 0, "total": 0})
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        q_short = QUESTIONS[q_idx][:55] + "..." if len(QUESTIONS[q_idx]) > 55 else QUESTIONS[q_idx]
        print(f"Q{q_idx+1:2}: {acc:5.1f}% - {q_short}")

if __name__ == "__main__":
    asyncio.run(main())
