from typing import Optional, Union

DOER_DEFAULT = """You are a research assistant. Use the provided document if present.

Think through this step by step:
1. Identify the key information needed to answer the question
2. Find relevant facts in the document (if provided)
3. Reason through what the answer should be
4. State your answer clearly

After your answer, rate your confidence from 1-10:
- 10 = Absolutely certain, directly stated in document
- 7-9 = Very confident, strongly supported by evidence
- 4-6 = Moderately confident, some inference required
- 1-3 = Uncertain, limited evidence

Format your response as:
[Your reasoning and answer]

CONFIDENCE: [1-10]"""
JUDGE_DEFAULT = """You are evaluating multiple candidate answers to select the most accurate one.

EVALUATION PROCESS:
1. Review each candidate answer carefully
2. Check each for factual accuracy against the document (if provided)
3. Check each for logical consistency and completeness
4. Identify which answers agree with each other (consensus signals correctness)
5. Note any errors or unsupported claims

SELECTION CRITERIA:
- Factual accuracy is most important
- Prefer answers supported by document evidence
- Consider confidence scores if provided
- Agreement among multiple candidates is a positive signal

Your response MUST end with exactly one selection in this format:
SELECTED: [doer:model_id#call_index]

Be decisive - pick the single best answer, even if none are perfect."""
FINAL_DEFAULT = """Synthesize the best final answer from the candidate answers and judge evaluations.

APPROACH:
1. Review the judge selections - they indicate which candidates are most accurate
2. Look for consensus among judges on which answer is best
3. If judges agree, use that answer as the basis
4. If judges disagree, examine the document directly to resolve
5. Trust the document over any candidate answer if there's a conflict

Output only the final answer - be concise and direct. Do not include reasoning or confidence scores in your final output."""
SCORER_DEFAULT = "Output only `0` or `1`. `1` means the candidate agrees with the ground truth; otherwise `0`."


def build_doc_block(doc_info: dict) -> Union[str, dict]:
    doc_id = doc_info["doc_id"]
    filename = doc_info["filename"]
    mime = doc_info["mime"]
    encoding = doc_info["encoding"]
    content = doc_info["content"]

    if encoding == "image":
        return {
            "type": "image",
            "doc_id": doc_id,
            "filename": filename,
            "mime": mime,
            "content": content,
        }

    if encoding == "pdf":
        return {
            "type": "pdf",
            "doc_id": doc_id,
            "filename": filename,
            "mime": mime,
            "pdf_raw": doc_info.get("pdf_raw"),      # base64 raw PDF
            "pdf_text": doc_info.get("pdf_text"),    # extracted text
            "pdf_pages": doc_info.get("pdf_pages"),  # list of base64 page images
        }

    return f"[DOCUMENT]\ndoc_id: {doc_id}\nfilename: {filename}\nmime: {mime}\nencoding: {encoding}\ncontent: {content}\n[/DOCUMENT]"


def build_doer_user_message(question: str, doc_block: Optional[Union[str, dict]]) -> Union[str, list]:
    if doc_block and isinstance(doc_block, dict) and doc_block.get("type") == "image":
        return [
            {"type": "text", "text": f"QUESTION: {question}\n\n[DOCUMENT: {doc_block['filename']}]"},
            {"type": "image_url", "image_url": {"url": f"data:{doc_block['mime']};base64,{doc_block['content']}"}}
        ]

    if doc_block and isinstance(doc_block, dict) and doc_block.get("type") == "pdf":
        # Build message with all available PDF representations
        parts = []
        text_intro = f"QUESTION: {question}\n\n[PDF DOCUMENT: {doc_block['filename']}]"

        # Add extracted text if available
        if doc_block.get("pdf_text"):
            text_intro += f"\n\n--- EXTRACTED TEXT ---\n{doc_block['pdf_text']}\n--- END EXTRACTED TEXT ---"

        parts.append({"type": "text", "text": text_intro})

        # Add raw PDF as document if available
        if doc_block.get("pdf_raw"):
            parts.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": doc_block["pdf_raw"]
                }
            })

        # Add page images if available
        if doc_block.get("pdf_pages"):
            for i, page_b64 in enumerate(doc_block["pdf_pages"]):
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{page_b64}"}
                })

        return parts if len(parts) > 1 else parts[0]["text"]

    msg = f"QUESTION: {question}"
    if doc_block:
        msg += f"\n\n{doc_block}"
    return msg


def _add_pdf_parts(doc_block: dict, parts: list) -> None:
    """Add PDF representations (raw, pages) to message parts list."""
    if doc_block.get("pdf_raw"):
        parts.append({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": doc_block["pdf_raw"]
            }
        })
    if doc_block.get("pdf_pages"):
        for page_b64 in doc_block["pdf_pages"]:
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{page_b64}"}
            })


def build_judge_user_message(
    question: str,
    doc_block: Optional[Union[str, dict]],
    doer_outputs: Optional[list[dict]],
) -> Union[str, list]:
    text_parts = [f"QUESTION: {question}"]

    is_image = doc_block and isinstance(doc_block, dict) and doc_block.get("type") == "image"
    is_pdf = doc_block and isinstance(doc_block, dict) and doc_block.get("type") == "pdf"

    if doc_block and not is_image and not is_pdf:
        text_parts.append(doc_block)
    elif is_image:
        text_parts.append(f"[DOCUMENT: {doc_block['filename']}]")
    elif is_pdf:
        text_parts.append(f"[PDF DOCUMENT: {doc_block['filename']}]")
        if doc_block.get("pdf_text"):
            text_parts.append(f"--- EXTRACTED TEXT ---\n{doc_block['pdf_text']}\n--- END EXTRACTED TEXT ---")

    if doer_outputs:
        text_parts.append("DOER RESPONSES:")
        for out in doer_outputs:
            text_parts.append(f"- [doer:{out['model_id']}#{out['call_index']}] {out['text']}")

    text_parts.append("INSTRUCTIONS: Evaluate the responses. Prefer correctness and completeness.")

    if is_image:
        return [
            {"type": "text", "text": "\n\n".join(text_parts)},
            {"type": "image_url", "image_url": {"url": f"data:{doc_block['mime']};base64,{doc_block['content']}"}}
        ]
    if is_pdf:
        parts = [{"type": "text", "text": "\n\n".join(text_parts)}]
        _add_pdf_parts(doc_block, parts)
        return parts if len(parts) > 1 else parts[0]["text"]
    return "\n\n".join(text_parts)


def build_final_user_message(
    question: str,
    doc_block: Optional[Union[str, dict]],
    doer_outputs: Optional[list[dict]],
    judge_outputs: Optional[list[dict]],
) -> Union[str, list]:
    text_parts = [f"QUESTION: {question}"]

    is_image = doc_block and isinstance(doc_block, dict) and doc_block.get("type") == "image"
    is_pdf = doc_block and isinstance(doc_block, dict) and doc_block.get("type") == "pdf"

    if doc_block and not is_image and not is_pdf:
        text_parts.append(doc_block)
    elif is_image:
        text_parts.append(f"[DOCUMENT: {doc_block['filename']}]")
    elif is_pdf:
        text_parts.append(f"[PDF DOCUMENT: {doc_block['filename']}]")
        if doc_block.get("pdf_text"):
            text_parts.append(f"--- EXTRACTED TEXT ---\n{doc_block['pdf_text']}\n--- END EXTRACTED TEXT ---")

    if doer_outputs:
        text_parts.append("DOER RESPONSES:")
        for out in doer_outputs:
            text_parts.append(f"- [doer:{out['model_id']}#{out['call_index']}] {out['text']}")

    if judge_outputs:
        text_parts.append("JUDGE RESPONSES:")
        for out in judge_outputs:
            text_parts.append(f"- [judge:{out['model_id']}#{out['call_index']}] {out['text']}")

    text_parts.append("INSTRUCTIONS: Write the best final answer. If the document conflicts with a candidate response, trust the document.")

    if is_image:
        return [
            {"type": "text", "text": "\n\n".join(text_parts)},
            {"type": "image_url", "image_url": {"url": f"data:{doc_block['mime']};base64,{doc_block['content']}"}}
        ]
    if is_pdf:
        parts = [{"type": "text", "text": "\n\n".join(text_parts)}]
        _add_pdf_parts(doc_block, parts)
        return parts if len(parts) > 1 else parts[0]["text"]
    return "\n\n".join(text_parts)


def build_scorer_user_message(ground_truth: str, candidate_answer: str) -> str:
    return f"GROUND TRUTH: {ground_truth}\n\nCANDIDATE: {candidate_answer}\n\nDoes the candidate agree with the ground truth? Output only 1 (agree) or 0 (disagree)."
