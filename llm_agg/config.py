from typing import List, Literal, Optional
from pydantic import BaseModel, Field

MAX_FILE_SIZE = 204800  # 200KB in bytes


class ModelRow(BaseModel):
    model_id: str
    timeout_s: float = Field(gt=0)
    n_calls: int = Field(ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)  # Per-model override


class ScorerConfig(BaseModel):
    model_id: str
    timeout_s: float = Field(gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class BenchmarkConfig(BaseModel):
    enabled: bool = False
    mode: Literal["exact", "llm"] = "exact"
    strip_whitespace: bool = True
    scorer: Optional[ScorerConfig] = None
    ground_truths: List[str] = Field(default_factory=list)


class DocInfo(BaseModel):
    doc_id: str
    filename: str
    mime: str
    encoding: Literal["utf-8", "base64", "base64+zlib", "image", "pdf"]
    content: str  # For text/image files; empty for PDFs using new fields
    # PDF-specific fields (user can enable multiple)
    pdf_raw: Optional[str] = None      # Base64 encoded raw PDF (send as-is)
    pdf_text: Optional[str] = None     # Extracted text from PDF
    pdf_pages: Optional[List[str]] = None  # Base64 encoded page images (all pages)


class RunConfig(BaseModel):
    questions: List[str] = Field(min_length=1)
    docs: List[DocInfo] = Field(default_factory=list, max_length=10)
    doers: List[ModelRow] = Field(default_factory=list, max_length=10)
    judges: List[ModelRow] = Field(default_factory=list, max_length=10)
    final_judges: List[ModelRow] = Field(default_factory=list, max_length=10)
    doer_system_prompt: str = ""
    judge_system_prompt: str = ""
    final_system_prompt: str = ""
    send_doc_to_judges: bool = False
    send_doc_to_final_judges: bool = False
    send_doer_responses_to_judges: bool = True
    send_doer_outputs_to_final_judges: bool = True
    send_judge_outputs_to_final_judges: bool = True
    cap_total_calls: int = Field(default=100, ge=1)
    max_output_tokens: int = Field(default=200, ge=1)
    retries: int = Field(default=0, ge=0)
    debug_mode: bool = False
    max_concurrency: int = Field(default=1000, ge=1)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
