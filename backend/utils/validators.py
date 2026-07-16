"""
HTTP-facing upload validation, performed before any service is called --
per this phase's explicit requirement to validate "uploaded file type,
empty uploads, unsupported extension, corrupted image ... before calling
services."

Two different depths of validation are deliberately used across this
package's two upload paths, and both are intentional (not an oversight):

* ``/predict`` and ``/predict/batch`` validate only the cheap, HTTP-layer
  things here (empty upload, disallowed extension) and then hand raw bytes
  to ``PredictionService.predict_from_bytes``, which performs its own
  byte-for-byte corruption/channel-count validation via the already-frozen
  ``inference.preprocessing.validate_and_decode_image`` -- per-image,
  without raising, returning a structured failed ``PredictionResult``
  rather than an HTTP error. Reimplementing that same corruption check here
  would be exactly the "duplicated validation logic" this phase's
  architecture forbids (see ``services.prediction_service``'s own module
  docstring for why its input adapters are designed this way).
* ``/explain/*`` has no such downstream adapter -- ``ExplainabilityEngine``
  takes an already-decoded ``PIL.Image.Image``, not bytes, and does no
  corruption checking of its own. This module's :func:`validate_and_decode`
  is the only place that check happens for that path, so it *does* perform
  full decode/corruption validation here, via
  ``backend.utils.image.decode_to_pil``.

``inference.preprocessing.SUPPORTED_EXTENSIONS`` is imported (not
redefined) so both paths agree with the frozen pipeline on which
extensions are acceptable.
"""
from __future__ import annotations

from typing import Optional

from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

from backend.utils.image import (
    ImageDecodeError,
    decode_to_pil,
    filename_extension,
    read_upload_bytes,
)
from inference.preprocessing import SUPPORTED_EXTENSIONS

#: Multipart content-types accepted at the HTTP layer. Intentionally a
#: superset-tolerant check (some clients send a generic
#: ``application/octet-stream`` for image uploads) -- the authoritative
#: check is always the actual decode, not the client-supplied header.
ACCEPTABLE_CONTENT_TYPE_PREFIXES = ("image/", "application/octet-stream")


def validate_filename_extension(filename: Optional[str]) -> None:
    """Reject unsupported file extensions before any decode is attempted.

    Raises:
        HTTPException: 422, if ``filename`` has an extension outside
            ``inference.preprocessing.SUPPORTED_EXTENSIONS``. A missing
            filename/extension is NOT rejected here -- some multipart
            clients omit it -- content-based decoding is the real gate.
    """
    ext = filename_extension(filename)
    if ext and ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file extension '{ext}' for '{filename}'. "
                f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )


def validate_content_type(content_type: Optional[str]) -> None:
    """Best-effort content-type check (advisory -- see module docstring for
    why the real gate is always the decode step, for endpoints that
    decode)."""
    if content_type is None:
        return
    if not any(content_type.startswith(prefix) for prefix in ACCEPTABLE_CONTENT_TYPE_PREFIXES):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported content type '{content_type}'. Expected an image upload.",
        )


async def validate_and_read_bytes(upload: UploadFile, max_size_bytes: int) -> bytes:
    """Full HTTP-layer validation (extension, content-type, empty, size
    cap) for the ``/predict*`` path, returning raw bytes for
    ``PredictionService`` to decode/validate itself.

    Raises:
        HTTPException: 422, on any validation failure.
    """
    validate_filename_extension(upload.filename)
    validate_content_type(upload.content_type)
    try:
        return await read_upload_bytes(upload, max_size_bytes)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def validate_and_decode(upload: UploadFile, max_size_bytes: int) -> Image.Image:
    """Full validation AND decode (extension, content-type, empty, size
    cap, corruption) for the ``/explain/*`` path, which needs an actual
    ``PIL.Image.Image`` to pass to ``ExplainabilityService``.

    Raises:
        HTTPException: 422, on any validation or decode failure.
    """
    data = await validate_and_read_bytes(upload, max_size_bytes)
    try:
        return decode_to_pil(data)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def validate_batch_size(count: int, max_batch_size: int) -> None:
    """Reject empty or oversized batches before touching any service.

    Raises:
        HTTPException: 422, if ``count`` is 0 or exceeds ``max_batch_size``.
    """
    if count == 0:
        raise HTTPException(status_code=422, detail="No files were uploaded.")
    if count > max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {count} exceeds the maximum allowed batch size ({max_batch_size}).",
        )
