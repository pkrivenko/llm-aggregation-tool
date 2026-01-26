# Benchmark Results: ASAP vs Single Model

**Date:** 2026-01-25 16:04:51

**Iterations per configuration:** 100

**Scoring method:** LLM-based (Gemini judges correctness)

## Configuration

### Single Model
- Model: `google/gemini-3-flash-preview`
- Doers: 1 call (temperature=0.5)
- Judges: None
- Final Judge: None

### ASAP (Aggregation System)
- Model: `google/gemini-3-flash-preview`
- Doers: 10 calls (temperature=0.7)
- Judges: 5 calls (temperature=0.5)
- Final Judge: 1 call (temperature=0.3)

> **Note:** These results may be from a previous benchmark run. Re-run `python3 benchmark_asap_vs_single.py` to generate fresh results with the current configuration.

---

## Overall Results

| Configuration | Average Success Rate |
|--------------|---------------------|
| Single Model | 100.0% |
| ASAP         | 68.1% |
| **Improvement** | **-31.9%** |

---

## Results by Document

| Document | Single Model | ASAP | Improvement |
|----------|-------------|------|-------------|
| doc1 (doc1.txt) | 100.0% | 87.5% | -12.5% |
| doc2 (doc2.pdf) | 0.0% | 94.1% | +94.1% |
| doc3 (doc3.jpg) | 0.0% | 21.7% | +21.7% |

---

## Results by Question

| Q# | Question (truncated) | Single Model | ASAP | Improvement |
|----|---------------------|-------------|------|-------------|
| Q1 | Respond with full name of the candidate. In consis... | 100.0% | 100.0% | +0.0% |
| Q2 | How many years of experience does Pavel state in t... | 100.0% | 100.0% | +0.0% |
| Q3 | What's his highest level of education? Include onl... | 0.0% | 100.0% | +100.0% |
| Q4 | What was his position at Central Bank of Russia? | 0.0% | 83.3% | +83.3% |
| Q5 | Which Cloud Platform does he use? respond with one... | 0.0% | 71.4% | +71.4% |
| Q6 | List all the structural methods mentioned | 0.0% | 75.0% | +75.0% |
| Q7 | in which journal did he publish a paper titled Ass... | 0.0% | 50.0% | +50.0% |
| Q8 | What's the title of his research paper that docume... | 0.0% | 33.3% | +33.3% |
| Q9 | Who is his coauthor on a research paper that uses ... | 0.0% | 75.0% | +75.0% |
| Q10 | How many distinct coauthors does he have? Ignore f... | 0.0% | 14.3% | +14.3% |
| Q11 | In his education business, he lists some revenue n... | 0.0% | 61.5% | +61.5% |

---

## Detailed Results by Document and Question

### Single Model Results

| Doc \ Question | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 | Q11 | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| doc1 | 100% | 100% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | **18%** |
| doc2 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | **0%** |
| doc3 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | **0%** |

### ASAP Results

| Doc \ Question | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 | Q11 | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| doc1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 100% | **91%** |
| doc2 | 100% | 0% | 100% | 100% | 100% | 100% | 0% | 100% | 100% | 50% | 100% | **77%** |
| doc3 | 100% | 100% | 100% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | **27%** |

### Improvement (ASAP - Single Model)

| Doc \ Question | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 | Q11 | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| doc1 | +0% | +0% | +100% | +100% | +100% | +100% | +100% | +100% | +100% | +0% | +100% | **+73%** |
| doc2 | +100% | +0% | +100% | +100% | +100% | +100% | +0% | +100% | +100% | +50% | +100% | **+77%** |
| doc3 | +100% | +100% | +100% | +0% | +0% | +0% | +0% | +0% | +0% | +0% | +0% | **+27%** |

---

## Summary

Single model shows a **31.9%** advantage over ASAP.

### Key Observations

- **Largest improvement:** doc1, Q3 with +100.0%
- **Largest regression:** doc3, Q11 with +0.0%

---

## Data Files

- **Detailed responses:** `benchmark_detailed_data.jsonl` (JSONL format with all model responses)
- **Summary stats:** `benchmark_raw_results.json`
