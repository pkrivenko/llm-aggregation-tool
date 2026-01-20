"""Test PDF modes with GPT-5-nano"""
import asyncio
import base64
from pathlib import Path

from llm_agg.config import RunConfig, ModelRow, DocInfo, BenchmarkConfig
from llm_agg.runner import run_pipeline
from llm_agg.io import generate_run_id

PDF_PATH = Path("TestData/doc2.pdf")
QUESTION = "Respond with full name of the candidate. It consists of two words."
EXPECTED = "Pavel Krivenko"

def load_pdf_with_modes(pdf_modes: list[str]) -> DocInfo:
    """Load PDF with specified modes."""
    content_bytes = PDF_PATH.read_bytes()
    
    pdf_raw = None
    pdf_text = None
    pdf_pages = None
    
    if "raw" in pdf_modes:
        pdf_raw = base64.b64encode(content_bytes).decode("ascii")
    
    if "text" in pdf_modes:
        import fitz
        doc = fitz.open(stream=content_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        pdf_text = "\n\n".join(text_parts)
        doc.close()
    
    if "images" in pdf_modes:
        import fitz
        doc = fitz.open(stream=content_bytes, filetype="pdf")
        pdf_pages = []
        mat = fitz.Matrix(2, 2)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            pdf_pages.append(base64.b64encode(img_bytes).decode("ascii"))
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

async def test_mode(pdf_modes: list[str], mode_name: str):
    """Test a specific PDF mode configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {mode_name}")
    print(f"PDF modes: {pdf_modes}")
    print(f"{'='*60}")
    
    doc = load_pdf_with_modes(pdf_modes)
    print(f"  pdf_raw: {'Yes' if doc.pdf_raw else 'No'}")
    print(f"  pdf_text: {'Yes (' + str(len(doc.pdf_text)) + ' chars)' if doc.pdf_text else 'No'}")
    print(f"  pdf_pages: {'Yes (' + str(len(doc.pdf_pages)) + ' pages)' if doc.pdf_pages else 'No'}")
    
    config = RunConfig(
        questions=[QUESTION],
        docs=[doc],
        doers=[ModelRow(model_id="openai/gpt-5-nano", timeout_s=60.0, n_calls=1)],
        judges=[],
        final_judges=[],
        doer_system_prompt="You are a helpful assistant. Answer based on the document provided.",
        cap_total_calls=10,
        max_output_tokens=500,  # Increased from 100
        max_concurrency=10,
    )
    
    run_id = generate_run_id()
    
    def on_attempt(record):
        status = record.get("status", "unknown")
        model = record.get("model_id", "")
        latency = record.get("latency_ms", 0)
        print(f"  [{status}] {model} - {latency:.0f}ms")
        if record.get("error_message"):
            print(f"    Error: {record['error_message'][:200]}")
    
    try:
        results, attempts, scores = await run_pipeline(config, run_id, on_attempt)
        
        if results:
            answer = results[0]["primary_outputs"][0].get("text")
            if answer:
                print(f"\n  Answer: {answer}")
                print(f"  Expected: {EXPECTED}")
                match = EXPECTED.lower() in answer.lower()
                print(f"  Match: {'YES' if match else 'NO'}")
                return match, answer
            else:
                print("  Answer is None/empty")
                return False, None
        else:
            print("  No results!")
            return False, None
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, None

async def main():
    print("Testing PDF modes with GPT-5-nano")
    print(f"PDF file: {PDF_PATH}")
    print(f"Question: {QUESTION}")
    
    # All 7 non-empty combinations (2^3 - 1)
    test_cases = [
        (["raw"], "PDF as-is"),
        (["text"], "Text only"),
        (["images"], "Images only"),
        (["raw", "text"], "PDF + Text"),
        (["raw", "images"], "PDF + Images"),
        (["text", "images"], "Text + Images"),
        (["raw", "text", "images"], "All three"),
    ]
    
    results = {}
    for modes, name in test_cases:
        success, answer = await test_mode(modes, name)
        results[name] = {"success": success, "answer": answer}
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, data in results.items():
        status = "PASS" if data["success"] else "FAIL"
        ans = (data["answer"] or "")[:50]
        print(f"  {name}: {status} - {ans}")
    
    passed = sum(1 for d in results.values() if d["success"])
    print(f"\nPassed: {passed}/{len(results)}")
    return passed == len(results)

if __name__ == "__main__":
    asyncio.run(main())
