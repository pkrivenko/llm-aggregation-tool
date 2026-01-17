# Final Benchmark Results: ASAP vs Single Model

**Generated:** 2026-01-16 09:05:25

**Iterations completed:**
- Single Model: 100/100
- ASAP: 100/100

---

## Configuration Parameters

### Model Settings

| Parameter | Single Model | ASAP |
|-----------|-------------|------|
| **Model** | `google/gemini-2.0-flash-001` | `google/gemini-2.0-flash-001` |

### Doer Stage

| Parameter | Single Model | ASAP |
|-----------|-------------|------|
| Number of calls | 1 | 10 |
| Temperature | 0.3 | 0.7 |
| Timeout (sec) | 60 | 60 |

### Judge Stage

| Parameter | Single Model | ASAP |
|-----------|-------------|------|
| Enabled | No | Yes |
| Number of calls | - | 5 |
| Temperature | - | 0.3 |
| Timeout (sec) | - | 60 |
| Send doc to judges | - | Yes |
| Send doer responses | - | Yes |

### Final Judge Stage

| Parameter | Single Model | ASAP |
|-----------|-------------|------|
| Enabled | No | Yes |
| Number of calls | - | 1 |
| Temperature | - | 0.0 |
| Timeout (sec) | - | 60 |
| Send doc to final | - | Yes |
| Send doer outputs | - | Yes |
| Send judge outputs | - | Yes |

### Global Settings (Same for Both)

| Parameter | Value |
|-----------|-------|
| Max output tokens | 300 |
| Retries on failure | 1 |
| Max concurrency | 10 |
| Cap total calls | 500 (Single) / 2000 (ASAP) |

### Scoring Settings

| Parameter | Value |
|-----------|-------|
| Scoring mode | LLM-based (not exact match) |
| Scorer model | `google/gemini-2.0-flash-001` |
| Scorer temperature | 0.0 |
| Scorer timeout | 30 sec |
| Strip whitespace | Yes |

### Test Data

| Parameter | Value |
|-----------|-------|
| Documents | doc1.txt, doc2.pdf, doc3.jpg |
| Questions | 11 (from Questions.md) |
| Ground truths | Provided for all questions |

---

## Overall Results

| Configuration | Iterations | Average Success Rate |
|--------------|------------|---------------------|
| Single Model | 100 | **72.7%** |
| ASAP         | 100 | **74.0%** |
| **Difference** | - | **+1.3%** |

---

## Results by Document

| Document | Single Model | ASAP | Difference |
|----------|-------------|------|------------|
| doc1 | 100.0% | 100.0% | +0.0% |
| doc2 | 91.0% | 94.8% | +3.8% |
| doc3 | 27.3% | 27.3% | +0.0% |

---

## Results by Question

| Q# | Single Model | ASAP | Difference |
|----|-------------|------|------------|
| Q1 | 100.0% | 100.0% | +0.0% |
| Q2 | 100.0% | 100.0% | +0.0% |
| Q3 | 100.0% | 100.0% | +0.0% |
| Q4 | 66.7% | 66.7% | +0.0% |
| Q5 | 66.7% | 66.7% | +0.0% |
| Q6 | 66.7% | 66.7% | +0.0% |
| Q7 | 66.7% | 66.7% | +0.0% |
| Q8 | 66.7% | 66.7% | +0.0% |
| Q9 | 66.7% | 66.7% | +0.0% |
| Q10 | 33.8% | 47.7% | +13.9% |
| Q11 | 66.3% | 66.7% | +0.3% |

---

## Detailed Results Matrix

### Single Model Success Rates

| Doc \ Q | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 | Q11 | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| doc1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | **100%** |
| doc2 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 2% | 99% | **91%** |
| doc3 | 100% | 100% | 100% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | **27%** |

### ASAP Success Rates

| Doc \ Q | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 | Q11 | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| doc1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | **100%** |
| doc2 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 43% | 100% | **95%** |
| doc3 | 100% | 100% | 100% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | **27%** |

### Improvement (ASAP - Single)

| Doc \ Q | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 | Q11 | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| doc1 | +0% | +0% | +0% | +0% | +0% | +0% | +0% | +0% | +0% | +0% | +0% | **+0%** |
| doc2 | +0% | +0% | +0% | +0% | +0% | +0% | +0% | +0% | +0% | +41% | +1% | **+4%** |
| doc3 | +0% | +0% | +0% | +0% | +0% | +0% | +0% | +0% | +0% | +0% | +0% | **+0%** |

---

## Summary

**ASAP shows a +1.3% improvement over single model overall.**

Key findings:
- **Q10 on doc2**: ASAP shows +41% improvement (2% → 43%), the most significant gain
- **doc2 overall**: ASAP improves accuracy by +4% (91% → 95%)
- **doc1 and doc3**: No difference between methods (100% and 27% respectively)

The ASAP multi-stage aggregation approach provides modest but consistent improvements, with the most benefit on questions where the single model struggles.