"""
Request parameter schemas.

FastAPI cannot bind a single Pydantic model directly to a multipart request
that also carries file parts (``UploadFile`` fields must be declared
individually via ``File(...)`` in the route signature) -- so these models
are not used as automatic ``Body``/``Form`` binders. Instead, each router
receives its non-file fields as plain ``Form(...)`` parameters (explicit,
robust, no fragile "Pydantic-model-as-Form" recipe) and constructs the
matching schema below immediately afterward, getting Pydantic's type
coercion/validation/error messages for those fields in one place rather
than duplicated per route. This keeps validation logic out of the routers
themselves and out of any frozen module.
"""
from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator

#: inference.explainability.base.IG_STEPS / OCCLUSION_PATCH_SIZE /
#: OCCLUSION_STRIDE default values, mirrored here only as this schema's own
#: field defaults (not re-exported/duplicated as constants elsewhere) --
#: see inference.explainability.engine.ExplainabilityEngine.generate_integrated_gradients
#: / generate_occlusion for the frozen originals these defaults must match.
DEFAULT_IG_STEPS = 16
DEFAULT_OCCLUSION_PATCH_SIZE = 32
DEFAULT_OCCLUSION_STRIDE = 16


class ExplainRequestParams(BaseModel):
    """Validated common parameters for every ``/explain/*`` endpoint.

    ``target_class`` accepts either a class name (``str``) or a class
    index (``int``) -- exactly the ``Optional[Any]`` contract
    ``ExplainabilityEngine.generate_*``/``BaseExplainer._resolve_target``
    already accepts; this schema only narrows the HTTP-layer input (a
    plain form string) into one of those two types before it reaches the
    service, rather than changing what the service accepts.
    """

    target_class: Optional[Union[int, str]] = Field(
        default=None,
        description="Class name or class index to explain. Defaults to the model's own top prediction.",
    )
    sample_id: str = Field(default="sample", min_length=1, max_length=200)

    @field_validator("target_class", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("target_class")
    @classmethod
    def _coerce_numeric_string(cls, value: Any) -> Any:
        # Form fields always arrive as strings; a digit-only string means
        # the caller meant a class *index*, not a class literally named
        # e.g. "3". Non-numeric strings pass through as class names.
        if isinstance(value, str) and value.lstrip("-").isdigit():
            return int(value)
        return value


class IntegratedGradientsParams(BaseModel):
    """Additional parameter accepted only by ``/explain/integrated_gradients``
    (mirrors ``ExplainabilityEngine.generate_integrated_gradients``'s
    ``steps`` parameter)."""

    steps: int = Field(default=DEFAULT_IG_STEPS, ge=1, le=512)


class OcclusionParams(BaseModel):
    """Additional parameters accepted only by ``/explain/occlusion``
    (mirrors ``ExplainabilityEngine.generate_occlusion``'s ``patch_size``/
    ``stride`` parameters)."""

    patch_size: int = Field(default=DEFAULT_OCCLUSION_PATCH_SIZE, ge=1, le=1024)
    stride: int = Field(default=DEFAULT_OCCLUSION_STRIDE, ge=1, le=1024)
