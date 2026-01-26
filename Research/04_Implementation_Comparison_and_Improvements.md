# Implementation Comparison: Relai vs State-of-the-Art

**Date:** January 2026
**Purpose:** Compare current implementation against state-of-the-art and identify high-leverage improvements

---

## Executive Summary

This report compares the Relai LLM Aggregation Tool's current implementation against state-of-the-art techniques identified in our research. While the current three-stage pipeline (Doers → Judges → Final Judges) demonstrates the value of aggregation (+1.3% overall, +23% on difficult questions), several high-leverage improvements could significantly enhance reliability with minimal complexity.

**Key Findings:**
- The current implementation captures the essence of aggregation but lacks explicit voting mechanisms
- Simple improvements (answer extraction + majority voting, confidence weighting) could yield 5-15% additional gains
- The architecture is well-suited for enhancement without major restructuring
- Focus on "quality over quantity" - fewer, smarter calls beat more naive calls

---

## 1. Current Implementation Analysis

### 1.1 Architecture Overview

```
Questions × Documents → Doers → Judges → Final Judges
                         ↓           ↓          ↓
                    [responses] [evaluations] [final answer]
```

**Stage 1 - Doers:**
- Multiple parallel LLM calls (typically 10)
- Temperature 0.7 for diversity
- Simple system prompt: "Answer directly. If uncertain, say so briefly."

**Stage 2 - Judges:**
- Evaluate all doer responses together
- System prompt: "Identify the best answer and why. Point out errors."
- See all responses with labels: `[doer:model_id#call_index]`

**Stage 3 - Final Judges:**
- Synthesize final answer from all information
- System prompt: "Write the best final answer"
- See doer outputs, judge outputs, and optionally the document

### 1.2 Benchmark Results

| Configuration | Accuracy | Notes |
|--------------|----------|-------|
| Single Model (temp=0.3) | 72.7% | Baseline |
| ASAP (10 doers, 5 judges) | 74.0% | +1.3% overall |
| ASAP on difficult Q10 | 43% vs 2% | +41% improvement |
| Focused ASAP (20 doers, 7 judges) | 73% vs 50% | +23% improvement |

**Key Insight:** The aggregation approach shows its value primarily on difficult questions where the single model consistently fails.

### 1.3 Current Strengths

1. **Clean Architecture**: Separates generation, evaluation, and synthesis
2. **Label Attribution**: `[doer:model_id#call_index]` enables judge reasoning
3. **Configurable Data Flow**: Control what each stage sees
4. **Multimodal Support**: Text, PDF, images handled properly
5. **Budget Control**: Hard caps prevent cost overruns
6. **Detailed Logging**: Full attempt records for debugging

### 1.4 Implementation Gaps

| Gap | Impact | Complexity to Fix |
|-----|--------|-------------------|
| No explicit voting mechanism | High | Low |
| No answer extraction | High | Low |
| No confidence estimation | Medium | Low |
| Prose synthesis vs selection | Medium | Low |
| No chain-of-thought | Medium | Very Low |
| No pairwise comparison | Medium | Medium |
| Single model family only | Medium | Low |

---

## 2. Comparison with State-of-the-Art

### 2.1 Self-Consistency (Wang et al., 2022)

**State-of-the-Art:**
- Generate N responses with temperature sampling
- **Extract final answers** using regex/parsing
- **Majority vote** to select most common answer
- Typical gains: +6-18% on reasoning tasks

**Current Implementation:**
- ✅ Generates N responses with temperature (Doers)
- ❌ No answer extraction - responses stay as full text
- ❌ No majority voting - relies on judge synthesis
- ❌ Final judges synthesize prose, don't vote

**Gap Analysis:** The current approach lets judges see all responses but doesn't leverage the simple power of majority voting. A single model seeing 10 responses and synthesizing isn't the same as counting which answer appears most often.

### 2.2 Universal Self-Consistency (Chen et al., 2024)

**State-of-the-Art:**
- For free-form answers where voting fails
- Use LLM to select "most consistent" response
- Prompt: "Which response is most consistent with the others?"

**Current Implementation:**
- ✅ Judge stage evaluates all responses
- ⚠️ Judge prompt asks for "best answer" not "most consistent"
- ⚠️ Judge generates evaluation, not selection

**Gap Analysis:** The judge prompt could be improved to explicitly ask for consistency-based selection rather than general quality evaluation.

### 2.3 LLM-as-a-Judge Best Practices

**State-of-the-Art:**
- Pairwise comparison > absolute scoring
- Position randomization to avoid bias
- Chain-of-thought reasoning before judgment
- Multi-judge aggregation (3+ models)
- Rubric-based evaluation

**Current Implementation:**
- ❌ No pairwise comparison - judges see all at once
- ❌ No position randomization
- ⚠️ Implicit reasoning in evaluation (not explicit CoT)
- ❌ Single model family (Gemini) for all stages
- ⚠️ General prompt, not rubric-based

**Gap Analysis:** The judge stage could benefit significantly from pairwise comparison and explicit rubrics. Using diverse model families would reduce systematic biases.

### 2.4 Confidence Estimation

**State-of-the-Art:**
- Confidence-weighted voting outperforms simple majority
- Self-certainty from token probabilities
- Verbalized confidence ("I am 80% confident...")
- Inverse-entropy weighting achieves 97% optimality

**Current Implementation:**
- ❌ No confidence extraction from doers
- ❌ No confidence-based weighting
- ❌ No uncertainty signals

**Gap Analysis:** Even simple verbalized confidence ("On a scale of 1-10, how confident are you?") could significantly improve aggregation quality.

### 2.5 Chain-of-Thought Prompting

**State-of-the-Art:**
- +6-18% gains on reasoning tasks
- "Let's think step by step"
- Intermediate reasoning improves final answer quality

**Current Implementation:**
- ❌ Doer prompt: "Answer directly" (no CoT)
- ⚠️ Judges analyze but not explicitly step-by-step

**Gap Analysis:** Simply adding "Let's think step by step before answering" to doer prompts could improve individual response quality significantly.

### 2.6 Multi-Agent Debate

**State-of-the-Art:**
- Multiple rounds of critique and refinement
- Agents see others' responses and argue
- ReConcile: confidence-weighted round-table discussion
- A-HMAD: heterogeneous specialized agents

**Current Implementation:**
- ⚠️ Single round of judge evaluation (no debate)
- ⚠️ No back-and-forth refinement
- ✅ Judges do see all responses (like debate input)

**Gap Analysis:** The current approach is closer to "committee evaluation" than "debate." Adding a second round where doers see critiques could help, but research shows diminishing returns - the current single-round approach may be optimal for cost-efficiency.

---

## 3. Summary Comparison Table

| Technique | State-of-Art Gain | Current Implementation | Gap |
|-----------|-------------------|----------------------|-----|
| Self-Consistency Voting | +6-18% | Not implemented | **High** |
| Answer Extraction | Required | Not implemented | **High** |
| Chain-of-Thought | +6-18% | Not implemented | **High** |
| Confidence Weighting | +2-5% | Not implemented | Medium |
| Pairwise Comparison | +5-10% | Not implemented | Medium |
| Heterogeneous Models | +3-5% | Not implemented | Medium |
| Position Randomization | -40% bias | Not implemented | Low |
| Multi-Round Debate | Variable | Single round | Low (diminishing returns) |
| Rubric-Based Evaluation | +13% | Not implemented | Medium |

---

## 4. High-Leverage Improvements

Based on the analysis, here are **6 high-leverage improvements** ranked by impact/complexity ratio:

### 4.1 Add Chain-of-Thought to Doer Prompts

**Complexity:** Very Low (prompt change only)
**Expected Gain:** +5-10%
**Implementation:**

```python
# Current
DOER_DEFAULT = "Answer the question directly. If uncertain, say so briefly."

# Improved
DOER_DEFAULT = """Think through this step by step before giving your final answer.

1. First, identify the key information in the question and document
2. Then, reason through what the answer should be
3. Finally, state your answer clearly

If uncertain, explain why and give your best estimate."""
```

### 4.2 Add Answer Extraction + Majority Voting

**Complexity:** Low (new utility function)
**Expected Gain:** +5-15%
**Implementation Concept:**

```python
def extract_answer(response_text: str) -> str:
    """Extract final answer from response."""
    # Try structured patterns
    patterns = [
        r"(?:Final [Aa]nswer|[Tt]he answer is|[Aa]nswer:)\s*(.+?)(?:\.|$)",
        r"(?:Therefore|Thus|So),?\s*(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    # Fallback: last sentence
    return response_text.strip().split('.')[-2].strip()

def majority_vote(responses: list[dict]) -> str:
    """Select answer with most votes."""
    answers = [extract_answer(r["text"]) for r in responses if r["status"] == "ok"]
    answer_counts = Counter(answers)
    return answer_counts.most_common(1)[0][0]
```

**Integration:**
- Run after doer stage, before judges
- If clear majority (>50%), can skip judge stage entirely for cost savings
- Pass vote count to judges as additional signal

### 4.3 Add Confidence Estimation

**Complexity:** Low (prompt modification + parsing)
**Expected Gain:** +2-5%
**Implementation Concept:**

```python
# Modified doer prompt
DOER_WITH_CONFIDENCE = """Think step by step and answer the question.

After your answer, rate your confidence from 1-10:
- 10 = Absolutely certain, no doubt
- 7-9 = Very confident, unlikely to be wrong
- 4-6 = Moderately confident, could go either way
- 1-3 = Uncertain, just a guess

Format:
ANSWER: [your answer]
CONFIDENCE: [1-10]"""

def extract_confidence(response_text: str) -> float:
    """Extract confidence score from response."""
    match = re.search(r"CONFIDENCE:\s*(\d+)", response_text)
    if match:
        return int(match.group(1)) / 10.0
    return 0.5  # Default moderate confidence

def confidence_weighted_vote(responses: list[dict]) -> str:
    """Weight votes by confidence scores."""
    answer_weights = defaultdict(float)
    for r in responses:
        if r["status"] == "ok":
            answer = extract_answer(r["text"])
            confidence = extract_confidence(r["text"])
            answer_weights[answer] += confidence
    return max(answer_weights.keys(), key=lambda a: answer_weights[a])
```

### 4.4 Use Heterogeneous Models

**Complexity:** Low (configuration change)
**Expected Gain:** +3-5%
**Implementation:**

Instead of:
```python
doers = [ModelRow(model_id="google/gemini-2.0-flash-001", n_calls=10)]
```

Use:
```python
doers = [
    ModelRow(model_id="google/gemini-2.0-flash-001", n_calls=4),
    ModelRow(model_id="anthropic/claude-3-haiku-20240307", n_calls=3),
    ModelRow(model_id="openai/gpt-4o-mini", n_calls=3),
]
```

**Benefits:**
- Reduces systematic biases of any single model family
- Different models have different strengths
- Disagreement between families is a meaningful signal

### 4.5 Improve Judge Prompts with Explicit Selection

**Complexity:** Low (prompt change)
**Expected Gain:** +3-8%
**Implementation:**

```python
# Current
JUDGE_DEFAULT = "Identify the best answer and why. Point out errors."

# Improved
JUDGE_DEFAULT = """You are evaluating multiple candidate answers.

TASK:
1. Review each candidate answer carefully
2. Check each for factual accuracy against the document
3. Check each for logical consistency
4. Select which ONE answer is most likely correct
5. Explain your selection briefly

Your output MUST end with:
SELECTED: [doer:model#index]

Be decisive - pick the single best answer, even if none are perfect."""
```

### 4.6 Add Structured Output Parsing

**Complexity:** Low-Medium (parsing logic)
**Expected Gain:** Enables other improvements
**Implementation:**

```python
def parse_judge_selection(judge_output: str) -> str:
    """Extract selected doer from judge output."""
    match = re.search(r"SELECTED:\s*\[doer:([^\]]+)\]", judge_output)
    if match:
        return match.group(1)
    return None

def aggregate_judge_selections(judge_outputs: list[dict]) -> str:
    """Count judge votes for doer selections."""
    selections = []
    for j in judge_outputs:
        if j["status"] == "ok":
            selected = parse_judge_selection(j["text"])
            if selected:
                selections.append(selected)
    if selections:
        return Counter(selections).most_common(1)[0][0]
    return None
```

---

## 5. Implementation Roadmap

### Phase 1: Quick Wins (1-2 hours)

1. **Update doer prompt** with chain-of-thought
2. **Update judge prompt** with explicit selection
3. **Use heterogeneous models** in benchmark config

**Expected combined gain:** +8-15%

### Phase 2: Voting System (2-4 hours)

1. **Add answer extraction** utility
2. **Add majority voting** as primary aggregation
3. **Add confidence extraction** and weighting
4. **Update final judge** to use vote counts as input

**Expected additional gain:** +5-10%

### Phase 3: Enhanced Judging (4-8 hours)

1. **Add pairwise comparison** mode for judges
2. **Add position randomization**
3. **Add rubric-based evaluation** options
4. **Track judge consistency** metrics

**Expected additional gain:** +3-5%

---

## 6. Recommended Configuration for Maximum Impact

Based on research, here's an optimized configuration for reliable aggregation:

```python
config = RunConfig(
    questions=questions,
    docs=[doc],

    # Heterogeneous doers with CoT prompting
    doers=[
        ModelRow(model_id="google/gemini-2.0-flash-001", n_calls=4, temperature=0.7),
        ModelRow(model_id="anthropic/claude-3-haiku-20240307", n_calls=3, temperature=0.7),
        ModelRow(model_id="openai/gpt-4o-mini", n_calls=3, temperature=0.7),
    ],
    doer_system_prompt=DOER_WITH_CONFIDENCE,  # CoT + confidence

    # Judges with explicit selection
    judges=[
        ModelRow(model_id="google/gemini-2.0-flash-001", n_calls=3, temperature=0.3),
    ],
    judge_system_prompt=JUDGE_WITH_SELECTION,  # Select best

    # Single final judge for synthesis
    final_judges=[
        ModelRow(model_id="google/gemini-2.0-flash-001", n_calls=1, temperature=0.0),
    ],

    # Full context flow
    send_doc_to_judges=True,
    send_doc_to_final_judges=True,
    send_doer_responses_to_judges=True,
    send_doer_outputs_to_final_judges=True,
    send_judge_outputs_to_final_judges=True,

    # Reasonable limits
    cap_total_calls=100,
    max_output_tokens=500,
    retries=1,
)
```

---

## 7. Measurement Strategy

To validate improvements, track:

| Metric | Baseline | Target |
|--------|----------|--------|
| Overall accuracy | 74.0% | 80-85% |
| Difficult question accuracy | 43% | 60-70% |
| Cost per question | ~$0.05 | ~$0.06 (acceptable +20%) |
| Latency (p95) | ~5s | ~7s (acceptable) |
| Judge-human agreement | N/A | >80% |

---

## 8. Conclusion

The current Relai implementation demonstrates the core value proposition of LLM aggregation but leaves significant gains on the table by not implementing:

1. **Explicit answer extraction and voting** (biggest single improvement)
2. **Chain-of-thought prompting** (free improvement via prompt change)
3. **Confidence-weighted aggregation** (low-hanging fruit)
4. **Heterogeneous model ensembles** (configuration change only)

These improvements are well-aligned with state-of-the-art research and can be implemented incrementally without architectural changes. The expected combined improvement is **+15-25%** relative accuracy gain, potentially bringing difficult question accuracy from 43% to 60-70%.

For a research prototype, focusing on Phase 1 and Phase 2 improvements provides the best return on investment, demonstrating that aggregation can deliver **significant and consistent reliability improvements** over single-model approaches.

---

## References

- Wang et al. (2022). Self-Consistency Improves Chain of Thought Reasoning
- Chen et al. (2024). Universal Self-Consistency for Large Language Models
- Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench
- See full references in companion research reports (01-03)

---

*Report generated: January 2026*
*For use in Relai LLM Aggregation Tool development*
