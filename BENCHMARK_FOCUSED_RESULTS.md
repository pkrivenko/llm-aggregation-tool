# Focused Benchmark Results: ASAP vs Single Model

**Generated:** 2026-01-16 09:14:32

## Test Configuration

| Parameter | Single Model | ASAP |
|-----------|-------------|------|
| **Model** | `google/gemini-2.0-flash-001` | `google/gemini-2.0-flash-001` |
| **Document** | doc2.pdf only | doc2.pdf only |
| **Questions** | Q10, Q11 only | Q10, Q11 only |
| **Doer calls** | 1 | 20 |
| **Doer temperature** | 0.0 | 0.7 |
| **Judge calls** | - | 7 |
| **Judge temperature** | - | 0.3 |
| **Final judge calls** | - | 1 |
| **Final temperature** | - | 0.0 |
| **Max concurrency** | 1000 | 1000 |

**Iterations:** 100 Single Model, 100 ASAP

---

## Overall Results

| Configuration | Average Success Rate |
|--------------|---------------------|
| Single Model (temp=0) | **50.0%** |
| ASAP (20 doers, 7 judges) | **73.0%** |
| **Difference** | **+23.0%** |

---

## Results by Question

| Question | Single Model | ASAP | Difference |
|----------|-------------|------|------------|
| Q10 | 0.0% | 48.0% | +48.0% |
| Q11 | 100.0% | 98.0% | -2.0% |

---

## Summary

ASAP with 20 doers and 7 judges shows **23.0%** improvement over single model (temp=0).