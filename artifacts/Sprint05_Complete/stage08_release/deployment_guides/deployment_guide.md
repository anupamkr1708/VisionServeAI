# VisionServeAI Chest X-Ray Multi-Label Classifier -- Deployment Guide

## 1. Package Folder Structure

## 2. Environment Setup

- Python: 3.12.13 (supported range >=3.9,<3.13)
- Torch: 2.10.0+cu128 (minimum major version 2)
- Torchvision: 0.25.0+cu128
- ONNX opset: 17
- CUDA available: True (configured device: cuda)

```bash
pip install torch torchvision onnx onnxruntime pillow numpy
```

## 3. Loading TorchScript

```python
import torch
model = torch.jit.load("/kaggle/working/sprint05_deployment/stage03_export/model.ts", map_location="cpu")
model.eval()
```

## 4. Loading ONNX

```python
import onnxruntime as ort
session = ort.InferenceSession("/kaggle/working/sprint05_deployment/stage03_export/model.onnx", providers=["CPUExecutionProvider"])
```

## 5. Loading Native PyTorch Checkpoint

```python
import torch
checkpoint = torch.load("/kaggle/input/datasets/anupsharma1730/visionserveai-sprint04-artifacts/visionserveai/sprint04/checkpoints/best_model.pt", map_location="cpu")
# Reconstruct the densenet121 architecture (see Stage 02), then:
# model.load_state_dict(state_dict, strict=True); model.eval()
```

## 6. Preprocessing

- Resize to: 224 x 224 (export_registry.model_signature.input_shape)
- Channels: 3
- Normalization mean: [0.485, 0.456, 0.406] (fallback_imagenet_default (no normalization recorded in training_summary.json))
- Normalization std: [0.229, 0.224, 0.225] (fallback_imagenet_default (no normalization recorded in training_summary.json))

```python
from PIL import Image
import numpy as np
import torch

img = Image.open(path).convert("RGB").resize(
    (224, 224), Image.BILINEAR
)
array = np.asarray(img, dtype=np.float32) / 255.0
tensor = torch.from_numpy(array).permute(2, 0, 1)
mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
tensor = (tensor - mean) / std
```

## 7. Thresholds

Per-class deployment thresholds (source: `/kaggle/input/datasets/anupsharma1730/visionserve-sprint04-evaluation/sprint04_evaluation/stage06_threshold_calibration_engine/optimal_thresholds.json`):

```json
{
  "Atelectasis": 6.325932986328553e-07,
  "Cardiomegaly": 0.014227046631276608,
  "Consolidation": 0.9983059167861938,
  "Edema": 0.7518303990364075,
  "Effusion": 2.956914010518452e-12,
  "Emphysema": 0.0004060639475937933,
  "Fibrosis": 8.525514564980869e-36,
  "Hernia": 3.139132888409754e-17,
  "Infiltration": 0.9965029954910278,
  "Mass": 4.485758893224556e-07,
  "Nodule": 1.502111953922614e-22,
  "Pleural_Thickening": 6.367942875480995e-15,
  "Pneumonia": 1.920059570321639e-13,
  "Pneumothorax": 3.2106581282498325e-10
}
```

## 8. Single Image Inference

```python
result = engine.predict_image("/path/to/image.png")
```

## 9. Batch Inference

```python
results = engine.predict_batch(
    ["/path/a.png", "/path/b.png"],
    image_identifiers=["patient_a", "patient_b"],
)
```

## 10. Expected Output (JSON Response)

```json
{
  "image_identifier": "patient_a",
  "success": true,
  "predicted_diseases": ["Effusion"],
  "confidence_scores": {"Effusion": 0.81},
  "probabilities": {"...": "..."},
  "thresholds_used": {"...": "..."},
  "inference_timestamp_utc": "2026-07-05T00:00:00+00:00",
  "model_fingerprint_sha256": "74ad647587bb8b8297072d03f1aed20ba5243899dd502a7e6dbdf8bfc6a59010",
  "model_version": "densenet121-74ad647587bb",
  "error": null
}
```

## 11. Error Handling

On decode/preprocessing failure, `success` is `false` and `error.error_type` is one of:

- `missing_file`
- `unsupported_extension`
- `corrupted_image`
- `zero_byte_image`
- `wrong_channel_count`
- `invalid_tensor_dimensions`

## 12. Deployment Checklist

- [x] model_reconstructed
- [x] torchscript_export_validated
- [x] onnx_export_validated
- [x] numerical_validation_passed
- [x] runtime_validated
- [x] explainability_available
- [x] metadata_complete
- [x] manifests_generated
- [x] checksums_verified
- [x] compatibility_passed
- [x] integrity_passed

Overall readiness: **READY** (100.0/100)
