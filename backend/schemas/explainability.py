"""
Explainability response payload schemas. Mirrors
``inference.explainability.base.ExplainabilityResult`` -- the frozen
dataclass every ``ExplainabilityEngine.generate_*`` method returns.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class ExplainabilityResultSchema(BaseModel):
    """Mirrors ``inference.explainability.base.ExplainabilityResult`` field
    for field -- no new fields, no renamed fields."""

    method: str
    sample_id: str
    success: bool
    execution_time_ms: float = 0.0
    predicted_class: Optional[str] = None
    predicted_class_index: Optional[int] = None
    target_class: Optional[str] = None
    target_class_index: Optional[int] = None
    confidence: Optional[float] = None
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
    raw_attribution_path: Optional[str] = None
    heatmap_shape: Optional[List[int]] = None
    no_nan: Optional[bool] = None
    no_inf: Optional[bool] = None
    error: Optional[str] = None
