"""
Image transport helpers: reading upload bytes and decoding them into a
``PIL.Image.Image`` for endpoints that need an in-memory image object
(the explainability endpoints -- see ``backend.routers.explainability``).

This module performs no resizing, normalization, tensor construction, or
any other ML preprocessing -- that is exclusively
``inference.preprocessing``'s job. It performs the minimum decode needed to
hand a valid ``PIL.Image.Image`` to ``ExplainabilityEngine.generate()``,
which (unlike ``PredictionService``) has no bytes-in / file-path-in
adapter of its own and expects an already-decoded image.

Prediction endpoints do NOT use :func:`decode_to_pil` -- they pass raw
bytes straight to ``PredictionService.predict_from_bytes``, which performs
its own byte-for-byte-validated decode via
``inference.preprocessing.validate_and_decode_image`` (frozen, already
tested). Duplicating that decode here for the prediction path would be
exactly the kind of parallel validation logic this phase's "no duplicated
code" requirement forbids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image
from starlette.datastructures import UploadFile

try:
    from PIL import UnidentifiedImageError
except ImportError:  # pragma: no cover - older Pillow fallback
    UnidentifiedImageError = OSError  # type: ignore[assignment,misc]


class ImageDecodeError(ValueError):
    """Raised when upload bytes cannot be decoded into a valid image.
    Caught by ``backend.utils.validators`` and translated into a clean
    ``HTTPException`` -- never propagated raw to a router."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class DecodedUpload:
    """A validated, decoded upload: the original filename (for logging/
    identifiers) and the decoded, verified image."""

    filename: str
    image: Image.Image


async def read_upload_bytes(upload: UploadFile, max_size_bytes: int) -> bytes:
    """Read an ``UploadFile``'s full contents, enforcing ``max_size_bytes``.

    Raises:
        ImageDecodeError: on an empty upload, or one exceeding
            ``max_size_bytes``.
    """
    data = await upload.read()
    if not data:
        raise ImageDecodeError("Uploaded file is empty.")
    if len(data) > max_size_bytes:
        raise ImageDecodeError(
            f"Uploaded file ({len(data)} bytes) exceeds the maximum allowed size "
            f"({max_size_bytes} bytes)."
        )
    return data


def decode_to_pil(data: bytes) -> Image.Image:
    """Decode raw image bytes into a verified ``PIL.Image.Image``.

    Mirrors ``inference.preprocessing.validate_and_decode_image``'s
    structural-integrity check (``Image.open`` + ``.verify()`` + reopen --
    ``verify()`` invalidates the file handle) without importing that
    function directly, since this operates on in-memory bytes rather than
    a file path and has no ``expected_channels``/RGB-conversion concerns of
    its own (``ExplainabilityEngine`` / ``BaseExplainer._to_input_tensor``
    does its own ``.convert("RGB")``).

    Raises:
        ImageDecodeError: if the bytes cannot be decoded as an image.
    """
    import io

    try:
        buffer = io.BytesIO(data)
        img = Image.open(buffer)
        img.verify()
        buffer.seek(0)
        img = Image.open(buffer)
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageDecodeError(f"Failed to decode image: {exc}") from exc
    return img


def filename_extension(filename: Optional[str]) -> str:
    """Lowercased file extension (including the leading dot), or ``""`` if
    ``filename`` is ``None``/has none."""
    if not filename:
        return ""
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()
