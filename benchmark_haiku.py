"""Benchmark Claude Haiku 4.5: Text, Images (3 resolutions), PDF × 11 questions × 10 iterations"""
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
MODEL = "anthropic/claude-haiku-4-5-20251001"

# Parse questions and answers
questions_text = Path("TestData/Questions.md").read_text()
QUESTIONS = {}
ANSWERS = {}
for match in re.finditer(r"Q(\d+):\s*(.+)", questions_text):
    QUESTIONS[int(match.group(1))] = match.group(2).strip()
for match in re.finditer(r"A(\d+):\s*(.+)", questions_text):
    ANSWERS[int(match.group(1))] = match.group(2).strip()
Q_INDICES = sorted(QUESTIONS.keys())

def resize_image(img_bytes: bytes, max_dim: int) -> bytes:
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

def create_text_doc() -> DocInfo:
    import fitz
    doc = fitz.open(stream=PDF_BYTES, filetype="pdf")
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    return DocInfo(doc_id="doc2.pdf", filename="doc2.pdf", mime="application/pdf",
                   encoding="pdf", content="", pdf_text=text)

def create_image_doc(max_dim: int) -> DocInfo:
    import fitz
    doc = fitz.open(stream=PDF_BYTES, filetype="pdf")
    pdf_pages = []
    mat = fitz.Matrix(3, 3)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img_bytes = resize_image(pix.tobytes("png"), max_dim)
        pdf_pages.append(base64.b64encode(img_bytes).decode("ascii"))
    doc.close()
    return DocInfo(doc_id="doc2.pdf", filename="doc2.pdf", mime="application/pdf",
                   encoding="pdf", content="", pdf_pages=pdf_pages)

def create_pdf_doc() -> DocInfo:
    return DocInfo(doc_id="doc2.pdf", filename="doc2.pdf", mime="application/pdf",
                   encoding="pdf", content="", pdf_raw=base64.b64encode(PDF_BYTES).decode("ascii"))

@dataclass
class RunResult:
    mode: str
    question_idx: int
    iteration: int
    success: bool
    correct: bool
    latency_ms: float

def check_answer(answer: str, expected: str) -> bool:
    if not answer or not expected:
        return False
    return expected.lower() in answer.lower()

async def run_single(mode: str, doc: DocInfo, q_num: int, iteration: int) -> RunResult:
    question = QUESTIONS[q_num]
    expected = ANSWERS.get(q_num, "")
    
    config = RunConfig(
        questions=[question],
        docs=[doc],
        doers=[ModelRow(model_id=MODEL, timeout_s=90.0, n_calls=1)],
        judges=[], final_judges=[],
        doer_system_prompt="Answer based on the document. Be concise.",
        cap_total_calls=10, max_output_tokens=500, max_concurrency=10,
    )
    
    run_id = generate_run_id()
    latency_ms = 0
    
    def on_attempt(record):
        nonlocal latency_ms
        latency_ms = record.get("latency_ms", 0)
    
    try:
        results, _, _ = await run_pipeline(config, run_id, on_attempt)
        if results and results[0]["primary_outputs"]:
            answer = results[0]["primary_outputs"][0].get("text")
            if answer:
                return RunResult(mode, q_num, iteration, True, check_answer(answer, expected), latency_ms)
        return RunResult(mode, q_num, iteration, False, False, latency_ms)
    except:
        return RunResult(mode, q_num, iteration, False, False, latency_ms)

async def main():
    modes = [
        ("Text", create_text_doc()),
        ("Image 512px", create_image_doc(512)),
        ("Image 1024px", create_image_doc(1024)),
        ("Image 1536px", create_image_doc(1536)),
        ("PDF", create_pdf_doc()),
    ]
    
    total = len(modes) * len(Q_INDICES) * ITERATIONS
    print("=" * 70)
    print(f"CLAUDE HAIKU 4.5 BENCHMARK")
    print(f"Modes: {len(modes)} | Questions: {len(Q_INDICES)} | Iterations: {ITERATIONS}")
    print(f"Total API calls: {total}")
    print("=" * 70)
    
    print("\nPre-computing documents...")
    for name, _ in modes:
        print(f"  {name}: ready")
    
    all_tasks = []
    for mode_name, doc in modes:
        for q_num in Q_INDICES:
            for i in range(ITERATIONS):
                all_tasks.append(run_single(mode_name, doc, q_num, i))
    
    print(f"\nRunning {len(all_tasks)} API calls in parallel...")
    start = time.time()
    results = await asyncio.gather(*all_tasks, return_exceptions=True)
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s ({len(all_tasks)/elapsed:.1f} calls/sec)")
    
    # Aggregate
    stats = {}
    for r in results:
        if isinstance(r, Exception):
            continue
        if r.mode not in stats:
            stats[r.mode] = {"success": 0, "correct": 0, "total": 0, "latencies": []}
        stats[r.mode]["total"] += 1
        if r.success:
            stats[r.mode]["success"] += 1
            stats[r.mode]["latencies"].append(r.latency_ms)
        if r.correct:
            stats[r.mode]["correct"] += 1
    
    print("\n" + "=" * 70)
    print("RESULTS BY MODE")
    print("=" * 70)
    print(f"{'Mode':<18} {'Success':>10} {'Accuracy':>10} {'Avg Latency':>12}")
    print("-" * 70)
    for mode_name, _ in modes:
        s = stats.get(mode_name, {"success": 0, "correct": 0, "total": 0, "latencies": []})
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        lat = sum(s["latencies"]) / len(s["latencies"]) if s["latencies"] else 0
        print(f"{mode_name:<18} {s['success']:>4}/{s['total']:<4} {acc:>9.1f}% {lat:>10.0f}ms")
    print("-" * 70)

if __name__ == "__main__":
    asyncio.run(main())
