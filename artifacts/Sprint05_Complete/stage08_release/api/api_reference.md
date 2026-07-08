# VisionServeAI Chest X-Ray Multi-Label Classifier -- API Reference

## Endpoint: `POST /v1/predict`

### Request Schema
```json
{
  "type": "object",
  "properties": {
    "images": {
      "type": "array",
      "items": {
        "type": "string",
        "description": "Absolute or accessible image path/URI."
      },
      "minItems": 1
    },
    "image_identifiers": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Optional caller-supplied identifiers, one per image (defaults to the path)."
    }
  },
  "required": [
    "images"
  ]
}
```

### Response Schema
```json
{
  "type": "object",
  "properties": {
    "predictions": {
      "type": "array",
      "items": {
        "$ref": "#/definitions/PredictionObject"
      }
    }
  }
}
```

### Prediction Object
```json
{
  "image_identifier": "string",
  "success": "boolean",
  "predicted_diseases": "array[string]",
  "confidence_scores": "object<string, float 0-1>",
  "probabilities": "object<string, float 0-1>",
  "thresholds_used": "object<string, float 0-1>",
  "inference_timestamp_utc": "string (ISO-8601 UTC) | null",
  "model_fingerprint_sha256": "string | null",
  "model_version": "string | null",
  "error": "object{error_type, message} | null"
}
```

### Error Codes
- `400`: Malformed request body / missing required field.
- `415`: Unsupported media type (see `unsupported_extension`).
- `422`: Image failed decode/validation (see per-item `error.error_type`).
- `500`: Unhandled server/inference error.

### Validation Rules
- images must be a non-empty array.
- Each image must decode to 1, 3, or 4 source bands and convert to RGB.
- Each image must be non-zero-byte and have a supported extension (.png, .jpg, .jpeg, .bmp, .tiff, .tif).

### Example Request
```json
{
  "images": [
    "/data/patient_001.png",
    "/data/patient_002.png"
  ],
  "image_identifiers": [
    "patient_001",
    "patient_002"
  ]
}
```

### Example Response
```json
{
  "predictions": [
    {
      "image_identifier": "patient_001",
      "success": true,
      "predicted_diseases": [
        "Atelectasis"
      ],
      "confidence_scores": {
        "Atelectasis": 0.81
      },
      "model_version": "densenet121-74ad647587bb",
      "error": null
    },
    {
      "image_identifier": "patient_002",
      "success": false,
      "error": {
        "error_type": "corrupted_image",
        "message": "Failed to decode image: ..."
      }
    }
  ]
}
```
