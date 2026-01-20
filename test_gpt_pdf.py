"""Test GPT-5-nano with raw PDF upload (no temperature)"""
import asyncio
import base64
import re
from pathlib import Path

from llm_agg.config import RunConfig, ModelRow, DocInfo
from llm_agg.runner import run_pipeline
from llm_agg.io import generate_run_id

PDF_PATH = Path("TestData/doc2.pdf")
PDF_BYTES = PDF_PATH.read_bytes()

questions_text = Path("TestData/Questions.md").read_text()
QUESTIONS = {}
ANSWERS = {}
for match in re.finditer(r"Q(\d+):\s*(.+)", questions_text):
    QUESTIONS[int(match.group(1))] = match.group(2).strip()
for match in re.finditer(r"A(\d+):\s*(.+)", questions_text):
    ANSWERS[int(match.group(1))] = match.group(2).strip()

Q_INDICES = sorted(QUESTIONS.keys())

def load_pdf_raw() -> DocInfo:
    return DocInfo(
        doc_id="doc2.pdf",
        filename="doc2.pdf",
        mime="application/pdf",
        encoding="pdf",
        content="",
        pdf_raw=base64.b64encode(PDF_BYTES).decode("ascii"),
    )

async def test_question(q_num: int, doc: DocInfo):
    question = QUESTIONS[q_num]
    expected = ANSWERS.get(q_num, "")
    
    config = RunConfig(
        questions=[question],
        docs=[doc],
        doers=[ModelRow(
            model_id="openai/gpt-5-nano",
            timeout_s=90.0,
            n_calls=1,
            # No temperature - GPT-5-nano only supports default (1.0)
        )],
        judges=[],
        final_judges=[],
        doer_system_prompt="Answer based on the document. Be concise.",
        cap_total_calls=10,
        max_output_tokens=500,
        max_concurrency=10,
    )
    
    run_id = generate_run_id()
    result = {"status": None, "answer": None, "error": None, "latency": 0}
    
    def on_attempt(record):
        result["status"] = record.get("status")
        result["latency"] = record.get("latency_ms", 0)
        if record.get("error_message"):
            result["error"] = record["error_message"][:150]
    
    try:
        results, _, _ = await run_pipeline(config, run_id, on_attempt)
        if results and results[0]["primary_outputs"]:
            result["answer"] = results[0]["primary_outputs"][0].get("text")
    except Exception as e:
        result["error"] = str(e)[:150]
    
    correct = False
    if result["answer"] and expected:
        if expected.lower() in result["answer"].lower():
            correct = True
    
    return q_num, result, expected, correct

async def main():
    print("=" * 80)
    print("GPT-5-NANO RAW PDF TEST (no temperature)")
    print(f"Questions: {len(QUESTIONS)} | Model: gpt-5-nano")
    print("=" * 80)
    
    doc = load_pdf_raw()
    print(f"PDF size: {len(PDF_BYTES) / 1024:.0f} KB")
    
    tasks = [test_question(q_num, doc) for q_num in Q_INDICES]
    results = await asyncio.gather(*tasks)
    
    print("\n" + "-" * 80)
    print(f"{'Q#':<4} {'Status':<8} {'Correct':<8} {'Latency':<10} {'Answer'}")
    print("-" * 80)
    
    correct_count = 0
    for q_num, result, expected, correct in results:
        status = result["status"] or "error"
        latency = f"{result['latency']:.0f}ms"
        
        if correct:
            correct_count += 1
            mark = "YES"
        else:
            mark = "NO"
        
        if result["answer"]:
            display = result["answer"][:50].replace('\n', ' ')
        elif result["error"]:
            display = f"ERR: {result['error'][:45]}"
        else:
            display = "No response"
        
        print(f"Q{q_num:<3} {status:<8} {mark:<8} {latency:<10} {display}")
    
    print("-" * 80)
    print(f"Total: {correct_count}/{len(QUESTIONS)} correct ({correct_count/len(QUESTIONS)*100:.0f}%)")

if __name__ == "__main__":
    asyncio.run(main())
