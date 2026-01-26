"""
Aggregation utilities for LLM response voting and selection.

This module implements state-of-the-art aggregation techniques:
- Answer extraction from verbose responses
- Majority voting (self-consistency)
- Confidence-weighted voting
- Judge selection parsing and aggregation
"""

import re
from collections import Counter, defaultdict
from typing import Optional


def extract_answer(response_text: str) -> str:
    """
    Extract the final answer from a potentially verbose response.

    Tries structured patterns first, falls back to heuristics.
    Returns the extracted answer string (normalized).
    """
    if not response_text:
        return ""

    text = response_text.strip()

    # Remove CONFIDENCE section first to avoid including it in answer
    if "CONFIDENCE:" in text:
        text = text.split("CONFIDENCE:")[0].strip()

    # Pattern 1: Explicit answer markers - capture until period, comma, or end
    # These patterns try to capture just the answer value, not surrounding text
    patterns = [
        # "The answer is X" or "Answer: X" - capture X until punctuation
        r"(?:Final [Aa]nswer|[Tt]he answer is|[Aa]nswer:?)\s*[:\-]?\s*([^.,\n]+)",
        # "Therefore, X" or "Thus, X" - common conclusion patterns
        r"(?:Therefore|Thus|So|Hence|In conclusion),?\s+(?:the answer is\s+)?([^.,\n]+)",
        # "indicates X" or "shows X" patterns
        r"(?:indicates|shows|reveals|states|is)\s+(\d+(?:\.\d+)?|\w+)(?:\s*[.,]|\s*$)",
        # Multiple choice: A, B, C, D (single letter answer)
        r"(?:^|\n)\s*([A-D])\s*[.:\)]\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            # Clean up common trailing words
            answer = re.sub(r'\s+(?:based on|according to|as stated|from).*$', '', answer, flags=re.IGNORECASE)
            if answer and len(answer) >= 1:
                return _normalize_answer(answer)

    # Pattern 2: Look for numbers at the end of sentences (common for numerical answers)
    number_match = re.search(r'(?:is|equals?|=)\s*(\d+(?:\.\d+)?)\s*[.,]?\s*$', text, re.IGNORECASE | re.MULTILINE)
    if number_match:
        return _normalize_answer(number_match.group(1))

    # Fallback: Get last substantive sentence
    sentences = re.split(r'[.!?]\s+', text)
    for sent in reversed(sentences):
        sent = sent.strip().rstrip('.,;:')
        if len(sent) > 5:  # Skip very short fragments
            return _normalize_answer(sent)

    # Ultimate fallback: return cleaned full text (truncated)
    return _normalize_answer(text[:200])


def _normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (lowercase, strip punctuation)."""
    # Remove common prefixes
    answer = re.sub(r'^(the answer is|therefore|thus|so)[:\s]*', '', answer, flags=re.IGNORECASE)
    # Strip trailing punctuation and whitespace
    answer = answer.strip().rstrip('.,;:')
    # Normalize whitespace
    answer = re.sub(r'\s+', ' ', answer)
    return answer.lower().strip()


def extract_confidence(response_text: str) -> float:
    """
    Extract confidence score from response.

    Looks for "CONFIDENCE: N" format where N is 1-10.
    Returns float between 0.1 and 1.0, or 0.5 as default.
    """
    if not response_text:
        return 0.5

    # Pattern: CONFIDENCE: N (where N is 1-10)
    match = re.search(r"CONFIDENCE:\s*(\d+)", response_text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        # Clamp to 1-10 range and convert to 0.1-1.0
        score = max(1, min(10, score))
        return score / 10.0

    # Alternative patterns
    match = re.search(r"confidence[:\s]+(?:is\s+)?(\d+)", response_text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        score = max(1, min(10, score))
        return score / 10.0

    return 0.5  # Default moderate confidence


def majority_vote(responses: list[dict], use_confidence: bool = False) -> dict:
    """
    Perform majority voting on doer responses.

    Args:
        responses: List of response dicts with 'text' and 'status' fields
        use_confidence: Whether to weight votes by confidence scores

    Returns:
        Dict with:
        - 'winner': The most common answer
        - 'vote_count': Number of votes for winner
        - 'total_votes': Total valid responses
        - 'vote_share': Winner's share of votes (0-1)
        - 'all_votes': Counter of all answer->count mappings
        - 'weighted_scores': Dict of answer->weighted_score (if use_confidence)
    """
    valid_responses = [r for r in responses if r.get("status") == "ok" and r.get("text")]

    if not valid_responses:
        return {
            'winner': None,
            'vote_count': 0,
            'total_votes': 0,
            'vote_share': 0.0,
            'all_votes': Counter(),
        }

    if use_confidence:
        # Confidence-weighted voting
        answer_weights = defaultdict(float)
        answer_counts = Counter()

        for r in valid_responses:
            answer = extract_answer(r["text"])
            confidence = extract_confidence(r["text"])
            answer_weights[answer] += confidence
            answer_counts[answer] += 1

        if answer_weights:
            winner = max(answer_weights.keys(), key=lambda a: answer_weights[a])
            return {
                'winner': winner,
                'vote_count': answer_counts[winner],
                'total_votes': len(valid_responses),
                'vote_share': answer_counts[winner] / len(valid_responses),
                'all_votes': answer_counts,
                'weighted_scores': dict(answer_weights),
            }
    else:
        # Simple majority voting
        answers = [extract_answer(r["text"]) for r in valid_responses]
        answer_counts = Counter(answers)

        if answer_counts:
            winner, count = answer_counts.most_common(1)[0]
            return {
                'winner': winner,
                'vote_count': count,
                'total_votes': len(valid_responses),
                'vote_share': count / len(valid_responses),
                'all_votes': answer_counts,
            }

    return {
        'winner': None,
        'vote_count': 0,
        'total_votes': len(valid_responses),
        'vote_share': 0.0,
        'all_votes': Counter(),
    }


def parse_judge_selection(judge_output: str) -> Optional[str]:
    """
    Extract the selected doer reference from judge output.

    Looks for "SELECTED: [doer:model_id#call_index]" format.
    Returns the doer reference string (e.g., "google/gemini-2.0-flash-001#0")
    or None if not found.
    """
    if not judge_output:
        return None

    # Pattern: SELECTED: [doer:model_id#call_index]
    match = re.search(r"SELECTED:\s*\[doer:([^\]]+)\]", judge_output, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Alternative: SELECTED: doer:model_id#call_index (without brackets)
    match = re.search(r"SELECTED:\s*doer:(\S+)", judge_output, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def aggregate_judge_selections(judge_outputs: list[dict], doer_outputs: list[dict]) -> dict:
    """
    Aggregate judge selections to find consensus on best doer response.

    Args:
        judge_outputs: List of judge response dicts
        doer_outputs: List of doer response dicts (to map selections back)

    Returns:
        Dict with:
        - 'selected_doer': The doer reference most judges selected
        - 'selected_text': The actual text from that doer
        - 'selection_count': How many judges selected the winner
        - 'total_judges': Total judges with valid selections
        - 'all_selections': Counter of all selections
    """
    valid_judges = [j for j in judge_outputs if j.get("status") == "ok" and j.get("text")]

    selections = []
    for j in valid_judges:
        selected = parse_judge_selection(j["text"])
        if selected:
            selections.append(selected)

    if not selections:
        return {
            'selected_doer': None,
            'selected_text': None,
            'selection_count': 0,
            'total_judges': len(valid_judges),
            'all_selections': Counter(),
        }

    selection_counts = Counter(selections)
    winner, count = selection_counts.most_common(1)[0]

    # Find the original doer text
    selected_text = None
    for doer in doer_outputs:
        doer_ref = f"{doer['model_id']}#{doer['call_index']}"
        if doer_ref == winner:
            selected_text = doer.get("text")
            break

    return {
        'selected_doer': winner,
        'selected_text': selected_text,
        'selection_count': count,
        'total_judges': len(valid_judges),
        'all_selections': selection_counts,
    }


def compute_agreement_score(responses: list[dict]) -> float:
    """
    Compute how much agreement there is among responses.

    Returns a score from 0 to 1 where:
    - 1.0 = all responses give the same answer
    - 0.0 = all responses give different answers

    This can be used to decide if judge stage is needed
    (high agreement = skip judges for cost savings).
    """
    valid_responses = [r for r in responses if r.get("status") == "ok" and r.get("text")]

    if len(valid_responses) <= 1:
        return 1.0  # Single response = perfect agreement

    answers = [extract_answer(r["text"]) for r in valid_responses]
    answer_counts = Counter(answers)

    # Agreement = share of most common answer
    most_common_count = answer_counts.most_common(1)[0][1]
    return most_common_count / len(valid_responses)
