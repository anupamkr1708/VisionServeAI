---
model_name: VisionServeAI Chest X-Ray Multi-Label Classifier
version: None
tags: [chest-xray, multi-label-classification, medical-imaging, pytorch, onnx, torchscript]
---
# VisionServeAI Chest X-Ray Multi-Label Classifier

## Model Description
A densenet121-based multi-label image classifier that estimates the probability of 14 thoracic disease findings from a single chest X-ray image, following the NIH ChestX-ray14 label taxonomy.

**Purpose:** Research-oriented decision-support signal for thoracic disease screening on chest radiographs.

## Architecture
- Backbone: densenet121
- Total parameters: 6,968,206
- Output head: 14-way sigmoid multi-label classifier

## Training Dataset
- Dataset: NIH Chest X-ray14
- Training metadata available: True
- Evaluation metadata available: True

## Input Resolution
- 224 x 224 x 3

## Output Classes
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Effusion
- Emphysema
- Fibrosis
- Hernia
- Infiltration
- Mass
- Nodule
- Pleural_Thickening
- Pneumonia
- Pneumothorax

## Thresholding
- Source: `/kaggle/input/datasets/anupsharma1730/visionserve-sprint04-evaluation/sprint04_evaluation/stage06_threshold_calibration_engine/optimal_thresholds.json`
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

## Inference Pipeline
decode -> resize -> normalize -> forward pass (sigmoid) -> per-class threshold

## Deployment Formats
- TorchScript (script): success=True
- ONNX (opset 17): success=True

## Supported Hardware
```json
{
  "pytorch": {
    "cpu": true,
    "cuda": true
  },
  "torchscript": {
    "cpu": true,
    "cuda": true
  },
  "onnxruntime": {
    "cpu": true,
    "cuda": false
  }
}
```

## Expected Inputs
RGB image file (png/jpg/jpeg/bmp/tiff), any resolution -- resized internally.

## Expected Outputs
JSON PredictionResult: predicted_diseases, confidence_scores, probabilities, thresholds_used, model_version, model_fingerprint_sha256, inference_timestamp_utc.

## Known Limitations
- Single frontal-view chest radiograph input only; no lateral-view or multi-view fusion.
- Multi-label thresholds are calibrated on the Sprint 04 evaluation split and may not transfer directly to a different patient population, scanner, or acquisition protocol.
- The model has not undergone prospective clinical validation.

## Failure Cases (structured error types)
- `missing_file`
- `unsupported_extension`
- `corrupted_image`
- `zero_byte_image`
- `wrong_channel_count`
- `invalid_tensor_dimensions`

## Ethical Considerations
Predictions reflect patterns in historical, retrospectively-labeled data and may encode dataset-specific biases (acquisition site, patient demographics, label noise inherited from the original NLP-derived NIH labels). Outputs must not be used as the sole basis for a clinical decision.

## Clinical Disclaimer
This model is NOT approved for clinical diagnosis, triage, or treatment decisions. It has not been cleared by any regulatory body (e.g., FDA, CE). Any clinical use requires independent validation, regulatory clearance, and qualified human oversight.

## Research Use Disclaimer
Provided for research and engineering demonstration purposes only. Users are responsible for independently validating performance on their own data before any downstream use.

**Version:** None  
**Authors:** VisionServeAI ML Platform Engineering Team  
**License:** Research use only -- not licensed for clinical deployment.
