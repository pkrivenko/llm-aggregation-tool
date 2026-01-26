"""
LLM Aggregation Tool - Multi-stage pipeline for reliable LLM responses.

This package implements a three-stage aggregation pipeline:
1. Doers: Generate multiple responses with chain-of-thought and confidence
2. Judges: Evaluate and select the best response
3. Final Judges: Synthesize the final answer

Key features:
- Self-consistency voting with confidence weighting
- Explicit judge selection for transparent aggregation
- Support for heterogeneous model ensembles
- Multimodal support (text, PDF, images)
"""

from .config import RunConfig, DocInfo, ModelRow, BenchmarkConfig, ScorerConfig
from .runner import run_pipeline
from .aggregation import (
    extract_answer,
    extract_confidence,
    majority_vote,
    aggregate_judge_selections,
    compute_agreement_score,
)

__all__ = [
    # Config
    "RunConfig",
    "DocInfo",
    "ModelRow",
    "BenchmarkConfig",
    "ScorerConfig",
    # Pipeline
    "run_pipeline",
    # Aggregation
    "extract_answer",
    "extract_confidence",
    "majority_vote",
    "aggregate_judge_selections",
    "compute_agreement_score",
]
