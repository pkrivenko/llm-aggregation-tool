from typing import Optional, Union

DOER_DEFAULT = "You are a research assistant. Use the provided document if present. Answer the question directly. If uncertain, say so briefly."
JUDGE_DEFAULT = "You are evaluating multiple candidate answers. Identify the best answer and why. Point out errors and missing details."
FINAL_DEFAULT = "Write the best final answer using the candidate answers and the document if present. Output only the final answer."
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
            "text": content,
            "image": doc_info.get("pdf_image"),
        }

    return f"[DOCUMENT]\ndoc_id: {doc_id}\nfilename: {filename}\nmime: {mime}\nencoding: {encoding}\ncontent: {content}\n[/DOCUMENT]"


def build_doer_user_message(question: str, doc_block: Optional[Union[str, dict]]) -> Union[str, list]:
    if doc_block and isinstance(doc_block, dict) and doc_block.get("type") == "image":
        return [
            {"type": "text", "text": f"QUESTION: {question}\n\n[DOCUMENT: {doc_block['filename']}]"},
            {"type": "image_url", "image_url": {"url": f"data:{doc_block['mime']};base64,{doc_block['content']}"}}
        ]

    if doc_block and isinstance(doc_block, dict) and doc_block.get("type") == "pdf":
        # PDF: send both extracted text and first page image
        text_content = f"""QUESTION: {question}

[PDF DOCUMENT: {doc_block['filename']}]
The following is extracted text from the PDF. An image of the first page is also provided for layout reference.

--- EXTRACTED TEXT ---
{doc_block['text']}
--- END EXTRACTED TEXT ---"""
        parts = [{"type": "text", "text": text_content}]
        if doc_block.get("image"):
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{doc_block['image']}"}})
        return parts

    msg = f"QUESTION: {question}"
    if doc_block:
        msg += f"\n\n{doc_block}"
    return msg


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
        text_parts.append(f"[PDF DOCUMENT: {doc_block['filename']}]\n--- EXTRACTED TEXT ---\n{doc_block['text']}\n--- END EXTRACTED TEXT ---")

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
    if is_pdf and doc_block.get("image"):
        return [
            {"type": "text", "text": "\n\n".join(text_parts)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{doc_block['image']}"}}
        ]
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
        text_parts.append(f"[PDF DOCUMENT: {doc_block['filename']}]\n--- EXTRACTED TEXT ---\n{doc_block['text']}\n--- END EXTRACTED TEXT ---")

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
    if is_pdf and doc_block.get("image"):
        return [
            {"type": "text", "text": "\n\n".join(text_parts)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{doc_block['image']}"}}
        ]
    return "\n\n".join(text_parts)


def build_scorer_user_message(ground_truth: str, candidate_answer: str) -> str:
    return f"GROUND TRUTH: {ground_truth}\n\nCANDIDATE: {candidate_answer}\n\nDoes the candidate agree with the ground truth? Output only 1 (agree) or 0 (disagree)."
