"""
Prediction endpoints: ``/predict`` (single image), ``/predict/batch``
(multiple images, order preserved).

Every prediction computation happens inside ``PredictionService`` -- this
router's only jobs are: (1) HTTP-layer upload validation (see
``backend.utils.validators``) before calling the service, (2) shaping the
already-computed ``PredictionResult``(s) into the standard response
envelope.
"""
from __future__ import annotations

from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from backend.dependencies.prediction import get_prediction_service
from backend.schemas.responses import APIResponse, BatchPredictionResponseSchema, PredictionResultSchema
from backend.settings import Settings, get_settings
from backend.utils.response import success_response
from backend.utils.validators import validate_and_read_bytes, validate_batch_size
from inference.postprocessing import summarize_predictions
from services.prediction_service import ImageInput, PredictionService

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=APIResponse[PredictionResultSchema],
    summary="Predict on a single image",
    description="Runs the full inference pipeline on one uploaded image and returns per-class predictions.",
    operation_id="predict_single",
    responses={
        422: {"description": "Invalid upload (empty, unsupported type, or undecodable image)."},
        503: {"description": "Model/runtime not ready."},
    },
)
async def predict_single(
    request: Request,
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(description="Image file to run inference on.")],
    identifier: Annotated[Optional[str], Form(description="Optional caller-chosen identifier for this image.")] = None,
) -> APIResponse[PredictionResultSchema]:
    data = await validate_and_read_bytes(file, settings.max_upload_size_bytes)
    result = prediction_service.predict_from_bytes(data, identifier or file.filename)
    message = "Prediction completed successfully." if result.success else "Prediction failed for this image."
    return success_response(request, data=result.to_dict(), message=message)


@router.post(
    "/predict/batch",
    response_model=APIResponse[BatchPredictionResponseSchema],
    summary="Predict on multiple images",
    description="Runs the full inference pipeline on multiple uploaded images. Result order matches upload order.",
    operation_id="predict_batch",
    responses={
        422: {"description": "Invalid upload, empty batch, or batch too large."},
        503: {"description": "Model/runtime not ready."},
    },
)
async def predict_batch(
    request: Request,
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    files: Annotated[List[UploadFile], File(description="Image files to run inference on, in order.")],
    identifiers: Annotated[
        Optional[List[str]], Form(description="Optional per-image identifiers, same order as `files`.")
    ] = None,
) -> APIResponse[BatchPredictionResponseSchema]:
    validate_batch_size(len(files), settings.max_batch_size)
    if identifiers is not None and len(identifiers) != len(files):
        raise HTTPException(
            status_code=422,
            detail=f"identifiers ({len(identifiers)}) and files ({len(files)}) length mismatch.",
        )

    byte_payloads: List[ImageInput] = [await validate_and_read_bytes(f, settings.max_upload_size_bytes) for f in files]
    resolved_identifiers = identifiers or [f.filename or f"image_{i}" for i, f in enumerate(files)]

    results = prediction_service.predict_batch(byte_payloads, resolved_identifiers)
    summary = summarize_predictions(results)

    payload = {"results": [r.to_dict() for r in results], "summary": summary}
    return success_response(
        request,
        data=payload,
        message=f"Batch prediction completed: {summary['successful']}/{summary['total']} succeeded.",
    )
