"""
Explainability endpoints: ``/explain/gradcam``, ``/explain/gradcam_plus``,
``/explain/scorecam``, ``/explain/eigencam``, ``/explain/guided_backprop``,
``/explain/integrated_gradients``, ``/explain/occlusion``.

Every endpoint here simply decodes the uploaded image and calls the
matching ``ExplainabilityService.generate_*`` method -- no algorithm
implementation lives in this file. The five methods that share an
identical parameter set (target class + sample id, no extra knobs) are
registered from one shared route factory (:func:`_register_simple_route`)
rather than five near-identical copy-pasted route functions, per this
phase's "no duplicated code" requirement; ``integrated_gradients`` and
``occlusion`` each take one extra method-specific parameter and so get
their own route function.
"""
from __future__ import annotations

from typing import Annotated, Any, Callable, Dict, Optional, Union

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from PIL import Image

from backend.dependencies.explainability import get_explainability_service
from backend.schemas.explainability import ExplainabilityResultSchema
from backend.schemas.requests import (
    DEFAULT_IG_STEPS,
    DEFAULT_OCCLUSION_PATCH_SIZE,
    DEFAULT_OCCLUSION_STRIDE,
    ExplainRequestParams,
    IntegratedGradientsParams,
    OcclusionParams,
)
from backend.schemas.responses import APIResponse
from backend.settings import Settings, get_settings
from backend.utils.response import success_response
from backend.utils.validators import validate_and_decode
from services.explainability_service import ExplainabilityService

router = APIRouter(prefix="/explain", tags=["Explainability"])

_COMMON_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    422: {"description": "Invalid upload (empty, unsupported type, or undecodable image)."},
    503: {"description": "Model/runtime not ready, or explainability disabled on this deployment."},
}


async def _decode_upload(
    file: UploadFile, settings: Settings,
) -> Image.Image:
    return await validate_and_decode(file, settings.max_upload_size_bytes)


def _result_to_response(request: Request, result: Any, method_label: str) -> APIResponse[ExplainabilityResultSchema]:
    message = f"{method_label} explanation generated successfully." if result.success else f"{method_label} explanation failed."
    return success_response(request, data=result.to_dict(), message=message)


# ======================================================================
# The five parameter-identical methods, registered from one shared factory
# ======================================================================

SIMPLE_METHODS: Dict[str, Dict[str, str]] = {
    "gradcam": {"service_method": "generate_gradcam", "label": "Grad-CAM"},
    "gradcam_plus": {"service_method": "generate_gradcam_plus", "label": "Grad-CAM++"},
    "scorecam": {"service_method": "generate_scorecam", "label": "Score-CAM"},
    "eigencam": {"service_method": "generate_eigencam", "label": "Eigen-CAM"},
    "guided_backprop": {"service_method": "generate_guided_backprop", "label": "Guided Backpropagation"},
}


def _register_simple_route(path_suffix: str, service_method: str, label: str) -> None:
    async def _route(
        request: Request,
        explainability_service: Annotated[ExplainabilityService, Depends(get_explainability_service)],
        settings: Annotated[Settings, Depends(get_settings)],
        file: Annotated[UploadFile, File(description="Image to explain.")],
        target_class: Annotated[
            Optional[str], Form(description="Class name or index to explain. Defaults to the model's top prediction.")
        ] = None,
        sample_id: Annotated[str, Form(description="Identifier used when naming saved artifacts.")] = "sample",
    ) -> APIResponse[ExplainabilityResultSchema]:
        image = await _decode_upload(file, settings)
        params = ExplainRequestParams(target_class=target_class, sample_id=sample_id)
        method: Callable[..., Any] = getattr(explainability_service, service_method)
        result = method(image, target_class=params.target_class, sample_id=params.sample_id)
        return _result_to_response(request, result, label)

    router.add_api_route(
        f"/{path_suffix}",
        _route,
        methods=["POST"],
        response_model=APIResponse[ExplainabilityResultSchema],
        summary=f"{label} explanation",
        description=f"Runs {label} on one uploaded image via the frozen ExplainabilityEngine and returns the heatmap/overlay result.",
        operation_id=f"explain_{path_suffix}",
        responses=_COMMON_RESPONSES,
    )


for _suffix, _cfg in SIMPLE_METHODS.items():
    _register_simple_route(_suffix, _cfg["service_method"], _cfg["label"])


# ======================================================================
# Methods with an extra, method-specific parameter
# ======================================================================


@router.post(
    "/integrated_gradients",
    response_model=APIResponse[ExplainabilityResultSchema],
    summary="Integrated Gradients explanation",
    description="Runs Integrated Gradients on one uploaded image via the frozen ExplainabilityEngine.",
    operation_id="explain_integrated_gradients",
    responses=_COMMON_RESPONSES,
)
async def explain_integrated_gradients(
    request: Request,
    explainability_service: Annotated[ExplainabilityService, Depends(get_explainability_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(description="Image to explain.")],
    target_class: Annotated[Optional[str], Form(description="Class name or index to explain.")] = None,
    sample_id: Annotated[str, Form(description="Identifier used when naming saved artifacts.")] = "sample",
    steps: Annotated[int, Form(description="Number of interpolation steps along the integration path.")] = DEFAULT_IG_STEPS,
) -> APIResponse[ExplainabilityResultSchema]:
    image = await _decode_upload(file, settings)
    params = ExplainRequestParams(target_class=target_class, sample_id=sample_id)
    ig_params = IntegratedGradientsParams(steps=steps)
    result = explainability_service.generate_integrated_gradients(
        image, target_class=params.target_class, sample_id=params.sample_id, steps=ig_params.steps,
    )
    return _result_to_response(request, result, "Integrated Gradients")


@router.post(
    "/occlusion",
    response_model=APIResponse[ExplainabilityResultSchema],
    summary="Occlusion sensitivity explanation",
    description="Runs occlusion sensitivity analysis on one uploaded image via the frozen ExplainabilityEngine.",
    operation_id="explain_occlusion",
    responses=_COMMON_RESPONSES,
)
async def explain_occlusion(
    request: Request,
    explainability_service: Annotated[ExplainabilityService, Depends(get_explainability_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(description="Image to explain.")],
    target_class: Annotated[Optional[str], Form(description="Class name or index to explain.")] = None,
    sample_id: Annotated[str, Form(description="Identifier used when naming saved artifacts.")] = "sample",
    patch_size: Annotated[int, Form(description="Side length (px) of the square occlusion patch.")] = DEFAULT_OCCLUSION_PATCH_SIZE,
    stride: Annotated[int, Form(description="Stride (px) between successive occlusion patches.")] = DEFAULT_OCCLUSION_STRIDE,
) -> APIResponse[ExplainabilityResultSchema]:
    image = await _decode_upload(file, settings)
    params = ExplainRequestParams(target_class=target_class, sample_id=sample_id)
    occ_params = OcclusionParams(patch_size=patch_size, stride=stride)
    result = explainability_service.generate_occlusion(
        image,
        target_class=params.target_class,
        sample_id=params.sample_id,
        patch_size=occ_params.patch_size,
        stride=occ_params.stride,
    )
    return _result_to_response(request, result, "Occlusion Sensitivity")
