# VisionServeAI Sprint04 — Chest X-Ray Evaluation Report

_Generated 2026-07-05T07:04:55.502351+00:00 by Stage 8 (Production Report Generator)._

## Executive Summary

- **Model**: densenet121 + Linear(14)
- **Test samples**: 25596
- **Classes**: 14
- **Macro AUROC**: 0.46394449390805065  •  **Micro AUROC**: 0.584349289971795
- **Macro F1**: 0.04750639120267578  •  **Subset accuracy**: 0.0001562744178777934
- **Macro ECE**: 0.2819920381767572  •  **Macro MCE**: 0.35877962858267926
- **Explainability samples**: 119 across 7 attribution methods
- **Engineering health**: UNHEALTHY ({'n_checks': 127, 'PASS': 123, 'FAIL': 1, 'WARNING': 3, 'UNKNOWN': 0})
- **Deployment recommendation**: DO_NOT_DEPLOY (score 50/100)
- **Publication readiness**: 73/100

## Model

- Backbone: `densenet121 + Linear(14)`
- Checkpoint param counts match: False
- Checkpoint metadata cross-check: {'epoch': {'best_model_metadata': 10, 'training_summary': 10, 'match': True}, 'val_macro_auroc': {'best_model_metadata': 0.6291324087507639, 'training_summary': 0.6291324087507639, 'match': True}, 'val_loss': {'best_model_metadata': 1.2928808805067944, 'training_summary': 1.2928808805067944, 'match': True}}

## Dataset

- Usable test samples: 25596 (of 25596 in the raw manifest)
- Image size: [288, 288]
- Batch size: 32

## Training Summary

Training was performed upstream of this evaluation pipeline; only the checkpoint's self-reported metadata is available here (Stage 8 never re-runs or re-loads training). Cross-check against best_model_metadata: {'epoch': {'best_model_metadata': 10, 'training_summary': 10, 'match': True}, 'val_macro_auroc': {'best_model_metadata': 0.6291324087507639, 'training_summary': 0.6291324087507639, 'match': True}, 'val_loss': {'best_model_metadata': 1.2928808805067944, 'training_summary': 1.2928808805067944, 'match': True}}

## Inference Summary

- Device: cuda (Tesla T4)
- Total inference time: 279.474 sec
- Throughput: 1107.49 samples/sec (avg)
- Peak GPU memory: 320.44 MB
- All inference validation checks passed: True

## Metrics

| auroc_macro | auroc_micro | auroc_weighted | f1_macro | f1_micro | subset_accuracy | hamming_loss | mcc_macro | balanced_accuracy_macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.46394449390805065 | 0.584349289971795 | 0.48197892142198034 | 0.04750639120267578 | 0.15714730307447872 | 0.0001562744178777934 | 0.28719331145491483 | -0.0005616525689169788 | 0.4995022633276626 |

## Per-class Metrics

| class_name | support | positive_count | negative_count | positive_prevalence | negative_prevalence | predicted_positive_count | prediction_prevalence | auroc | auroc_valid | average_precision | average_precision_valid | precision | recall_sensitivity | specificity | f1 | accuracy | mcc | mcc_valid | balanced_accuracy | balanced_accuracy_valid | true_positive | true_negative | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Atelectasis | 3279 | 3279 | 22317 | 0.1281059540553211 | 0.8718940459446789 | 0 | 0.0 | 0.5343564477922521 | True | 0.1415235025057405 | True | 0.0 | 0.0 | 1.0 | 0.0 | 0.8718940459446789 | 0.0 | True | 0.5 | True | 0 | 22317 | 0 | 3279 |
| Cardiomegaly | 1069 | 1069 | 24527 | 0.0417643381778402 | 0.9582356618221596 | 18108 | 0.7074542897327707 | 0.4725423916667998 | True | 0.0386389175223478 | True | 0.0409763640379942 | 0.6941066417212348 | 0.2919639580870061 | 0.0773843666892631 | 0.3087591811220503 | -0.0061252712156405 | True | 0.4930352999041205 | True | 742 | 7161 | 17366 | 327 |
| Consolidation | 1815 | 1815 | 23781 | 0.0709095171120487 | 0.9290904828879512 | 25596 | 1.0 | 0.5001063306899516 | True | 0.0711581901675061 | True | 0.0709095171120487 | 1.0 | 0.0 | 0.1324285870635876 | 0.0709095171120487 | 0.0 | True | 0.5 | True | 1815 | 0 | 23781 | 0 |
| Edema | 925 | 925 | 24671 | 0.0361384591342397 | 0.9638615408657604 | 25595 | 0.9999609313955306 | 0.4697003922977738 | True | 0.0339269101066725 | True | 0.036139871068568 | 1.0 | 4.053341980462892e-05 | 0.0697586726998491 | 0.0361775277387091 | 0.0012103191999251 | True | 0.5000202667099023 | True | 925 | 1 | 24670 | 0 |
| Effusion | 4658 | 4658 | 20938 | 0.1819815596186904 | 0.8180184403813096 | 1 | 3.906860446944835e-05 | 0.4820270346920908 | True | 0.1712961883526434 | True | 0.0 | 0.0 | 0.9999522399465088 | 0.0 | 0.8179793717768401 | -0.0029481839491223 | True | 0.4999761199732543 | True | 0 | 20937 | 1 | 4658 |
| Emphysema | 1093 | 1093 | 24503 | 0.042701984685107 | 0.957298015314893 | 0 | 0.0 | 0.5171969158583528 | True | 0.0461691745539887 | True | 0.0 | 0.0 | 1.0 | 0.0 | 0.957298015314893 | 0.0 | True | 0.5 | True | 0 | 24503 | 0 | 1093 |
| Fibrosis | 435 | 435 | 25161 | 0.01699484294421 | 0.98300515705579 | 0 | 0.0 | 0.4076284817727855 | True | 0.0139387742726127 | True | 0.0 | 0.0 | 1.0 | 0.0 | 0.98300515705579 | 0.0 | True | 0.5 | True | 0 | 25161 | 0 | 435 |
| Hernia | 86 | 86 | 25510 | 0.0033598999843725 | 0.9966401000156274 | 0 | 0.0 | 0.3725506641262432 | True | 0.0025885460277979 | True | 0.0 | 0.0 | 1.0 | 0.0 | 0.9966401000156274 | 0.0 | True | 0.5 | True | 0 | 25510 | 0 | 86 |
| Infiltration | 6112 | 6112 | 19484 | 0.2387873105172683 | 0.7612126894827317 | 25596 | 1.0 | 0.5025129190443279 | True | 0.2378948315299623 | True | 0.2387873105172683 | 1.0 | 0.0 | 0.3855178503847609 | 0.2387873105172683 | 0.0 | True | 0.5 | True | 6112 | 0 | 19484 | 0 |
| Mass | 1748 | 1748 | 23848 | 0.0682919206125957 | 0.9317080793874044 | 0 | 0.0 | 0.4584971481280758 | True | 0.0620171606982679 | True | 0.0 | 0.0 | 1.0 | 0.0 | 0.9317080793874044 | 0.0 | True | 0.5 | True | 0 | 23848 | 0 | 1748 |
| Nodule | 1623 | 1623 | 23973 | 0.0634083450539146 | 0.9365916549460852 | 0 | 0.0 | 0.4275942855099952 | True | 0.0523137179110037 | True | 0.0 | 0.0 | 1.0 | 0.0 | 0.9365916549460852 | 0.0 | True | 0.5 | True | 0 | 23973 | 0 | 1623 |
| Pleural_Thickening | 1143 | 1143 | 24453 | 0.0446554149085794 | 0.9553445850914204 | 0 | 0.0 | 0.4315242349501225 | True | 0.0372074333766488 | True | 0.0 | 0.0 | 1.0 | 0.0 | 0.9553445850914204 | 0.0 | True | 0.5 | True | 0 | 24453 | 0 | 1143 |
| Pneumonia | 555 | 555 | 25041 | 0.0216830754805438 | 0.9783169245194562 | 0 | 0.0 | 0.4814276478467205 | True | 0.020562839046717 | True | 0.0 | 0.0 | 1.0 | 0.0 | 0.9783169245194562 | 0.0 | True | 0.5 | True | 0 | 25041 | 0 | 555 |
| Pneumothorax | 2665 | 2665 | 22931 | 0.1041178309110798 | 0.8958821690889202 | 0 | 0.0 | 0.4375580203372169 | True | 0.0897947049049075 | True | 0.0 | 0.0 | 1.0 | 0.0 | 0.8958821690889202 | 0.0 | True | 0.5 | True | 0 | 22931 | 0 | 2665 |

## Calibration

- Macro ECE: 0.2819920381767572, Macro MCE: 0.35877962858267926, Macro Brier: 0.2760408649903308


| class_name | ece | mce | brier_score |
| --- | --- | --- | --- |
| Atelectasis | 0.1277009160311611 | 0.1347759068012237 | 0.1279787960967566 |
| Cardiomegaly | 0.6235429010796091 | 0.935488765626693 | 0.5370334490604984 |
| Consolidation | 0.9290693987950924 | 0.9290693987950924 | 0.92905304244185 |
| Edema | 0.9631707093792374 | 0.963743674717273 | 0.9626938199000088 |
| Effusion | 0.1809169244488839 | 0.6130049228668213 | 0.1816765340490709 |
| Emphysema | 0.0419209191530296 | 0.0849515348672866 | 0.0426320743161485 |
| Fibrosis | 0.0169948426133981 | 0.0169948426133981 | 0.0169948429440534 |
| Hernia | 0.0033598999842805 | 0.0033598999842805 | 0.0033598999843725 |
| Infiltration | 0.7611899368556929 | 0.9118099808692932 | 0.7611783231298336 |
| Mass | 0.0667078331595495 | 0.0667096129414658 | 0.0681251777805606 |
| Nodule | 0.063408302632332 | 0.063408302632332 | 0.0634083425412456 |
| Pleural_Thickening | 0.0446481066570867 | 0.0446481066570867 | 0.0446550407505448 |
| Pneumonia | 0.0211696585064837 | 0.1508616656064987 | 0.0216693612939558 |
| Pneumothorax | 0.1040881851787634 | 0.1040881851787634 | 0.104113405575731 |

## Threshold Optimization

| class_name | f1_optimal_threshold | f1_optimal_value | balanced_accuracy_optimal_threshold | balanced_accuracy_optimal_value | youden_j_optimal_threshold | youden_j_optimal_value |
| --- | --- | --- | --- | --- | --- | --- |
| Atelectasis | 6.325932986328553e-07 | 0.2314777080446654 | 7.706509677518625e-06 | 0.5248613906883847 | 7.706509677518625e-06 | 0.0497227813767693 |
| Cardiomegaly | 0.0142270466312766 | 0.0808865971447262 | 0.0142270466312766 | 0.5050354770251283 | 0.0142270466312766 | 0.0100709540502565 |
| Consolidation | 0.9983059167861938 | 0.1324798829553767 | 1.0 | 0.5033380353299617 | 1.0 | 0.0066760706599233 |
| Edema | 0.7518303990364075 | 0.0697902519993964 | 0.7518303990364075 | 0.50026346722873 | 0.7518303990364075 | 0.0005269344574601 |
| Effusion | 2.956914010518452e-12 | 0.3083493345692908 | 8.105603654939841e-08 | 0.5027453110352464 | 8.105603654939841e-08 | 0.0054906220704928 |
| Emphysema | 0.0004060639475937 | 0.0854719209706861 | 0.0004060639475937 | 0.522454594969214 | 0.0004060639475937 | 0.0449091899384279 |
| Fibrosis | 8.525514564980869e-36 | 0.0337104773713577 | 8.525514564980869e-36 | 0.504431461388657 | 8.525514564980869e-36 | 0.008862922777314 |
| Hernia | 3.139132888409754e-17 | 0.0080482897384305 | 3.139132888409754e-17 | 0.5036114428450311 | 3.139132888409754e-17 | 0.0072228856900622 |
| Infiltration | 0.9965029954910278 | 0.3856394725219257 | 0.9999970197677612 | 0.5138324834392242 | 0.9999970197677612 | 0.0276649668784483 |
| Mass | 4.485758893224556e-07 | 0.1280856838939221 | 0.0249402541667222 | 0.5015485661669598 | 0.0249402541667222 | 0.0030971323339195 |
| Nodule | 1.5021119539226142e-22 | 0.1197457910580889 | 1.5021119539226142e-22 | 0.5025601814980856 | 1.5021119539226142e-22 | 0.0051203629961711 |
| Pleural_Thickening | 6.367942875480995e-15 | 0.0857884637092222 | 6.367942875480995e-15 | 0.5019387988720769 | 6.367942875480995e-15 | 0.0038775977441538 |
| Pneumonia | 1.9200595703216391e-13 | 0.042465281762883 | 0.0036928530316799 | 0.501924411532654 | 0.0036928530316799 | 0.0038488230653079 |
| Pneumothorax | 3.210658128249832e-10 | 0.1890066130982009 | 3.210658128249832e-10 | 0.501477660815058 | 3.210658128249832e-10 | 0.0029553216301159 |

## Interpretability

- Methods run: ['gradcam', 'gradcam_plus', 'scorecam', 'eigencam', 'guided_backprop', 'integrated_gradients', 'occlusion']
- Samples explained: 119
- PNGs written: 952
- Selection reasons: {'most_confident_wrong_prediction': 14, 'representative_correct_positive_per_class': 13, 'hard_false_positive': 12, 'hard_false_negative': 12, 'least_confident_correct_prediction': 10, 'largest_probability_error': 9, 'hard_true_positive': 8, 'false_negative_gallery': 8, 'hard_true_negative': 7, 'worst_calibrated_class_error': 6, 'rare_disease_failure(Pneumonia)': 5, 'false_positive_gallery': 4, 'class_confusion(pred=Edema,true=Infiltration)': 4, 'class_confusion(pred=Pneumothorax,true=Infiltration)': 4, 'class_confusion(pred=Pleural_Thickening,true=Infiltration)': 3}

## Engineering Validation

| stage_row | stage | PASS | FAIL | WARNING | UNKNOWN |
| --- | --- | --- | --- | --- | --- |
| 0 | stage01_environment | 4 | 0 | 0 | 0 |
| 1 | stage02_artifact_loader | 6 | 1 | 2 | 0 |
| 2 | stage03_dataset_builder | 4 | 0 | 1 | 0 |
| 3 | stage04_inference_engine | 27 | 0 | 0 | 0 |
| 4 | stage05_metrics_engine | 27 | 0 | 0 | 0 |
| 5 | stage06_threshold_calibration_engine | 22 | 0 | 0 | 0 |
| 6 | stage07_interpretability_engine | 33 | 0 | 0 | 0 |

## Deployment Readiness

- Score: 50/100
- Recommendation: **DO_NOT_DEPLOY**
- Blockers: ['checkpoint_verified', "14/14 classes flagged DO_NOT_DEPLOY: ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']"]

## Publication Readiness

- Score: 73/100

**Strengths:**
- Macro AUROC of 0.46394449390805065 across 14 pathologies on 25596 held-out test samples.
- 7-method explainability suite (Grad-CAM, Grad-CAM++, Score-CAM, Eigen-CAM, Guided Backprop, Integrated Gradients, Occlusion) with zero method failures reported in interpretability_summary.json.
- All Stage 5-7 engineering validations are fully green (0 failed checks).
- Deterministic, seeded, resumable pipeline with an explicit frozen artifact contract between every stage.

**Weaknesses:**
- Explainability coverage is bounded to 119 of 25596 test samples (0.46%) for computational reasons.
- Class imbalance ratio (max/min support) of 71.06976744186046; rare classes have wide uncertainty in their AUROC/ECE estimates.
- Single held-out split from one public dataset; no external or multi-institution validation cohort.
- Primary metrics are reported at a fixed 0.5 threshold even though Stage 6 computes per-class optimal thresholds that differ substantially from 0.5.

**Future work:**
- External validation on an independent chest X-ray cohort (different scanner population, different labeling protocol).
- Per-class threshold deployment (using Stage 6's optimal thresholds) rather than a single global 0.5 cut-off, with prospective monitoring of the resulting confusion matrix.
- Targeted data collection or class-balanced sampling for the lowest-support pathologies.
- Quantitative agreement study between explainability heatmaps and radiologist-annotated regions of interest.

## Limitations

**Threshold Limitations**

- Headline precision/recall/F1/accuracy in evaluation_metrics.json use a single fixed operating threshold of 0.5 for every class, while Stage 6 computes a distinct balanced-accuracy-optimal threshold per class (see optimal_thresholds.csv). Any deployment should use the per-class thresholds, not 0.5.

**Dataset Limitations**

- Evaluation is limited to a single frozen test split of 25596 samples from one public dataset; no external or multi-institution cohort was evaluated.
- The underlying chest X-ray label set is the standard NIH ChestXray14 labeling scheme, whose known text-mining label-extraction caveats (as documented publicly for this dataset) are inherited unchanged by this evaluation.

**Class Imbalance**

- {"lowest_support_classes": [{"class_name": "Hernia", "support": 86}, {"class_name": "Fibrosis", "support": 435}, {"class_name": "Pneumonia", "support": 555}], "note": "Classes with low positive support have wider-variance AUROC/ECE estimates and should be interpreted with additional caution."}

**Calibration Limitations**

- {"worst_calibrated_classes": [{"class_name": "Edema", "ece": 0.9631707093792374, "mce": 0.963743674717273}, {"class_name": "Consolidation", "ece": 0.9290693987950924, "mce": 0.9290693987950924}, {"class_name": "Infiltration", "ece": 0.7611899368556929, "mce": 0.9118099808692932}], "note": "Macro ECE=0.2819920381767572, macro MCE=0.35877962858267926; per-class calibration is uneven and should be re-checked before any probability output is shown to an end user as a literal risk percentage."}

**Interpretability Limitations**

- Only 119 of 25596 test samples received attribution maps (bounded by max_total_samples=150 in Stage 7's configuration) — attribution quality has not been assessed across the full test set.
- Rare-disease failure sampling captured 5 case(s) under reason(s) ['rare_disease_failure(Pneumonia)'], which is too small a sample to draw statistically robust conclusions about rare-class failure modes.
- Attribution maps (Grad-CAM family, Guided Backprop, Integrated Gradients, Occlusion) have not been quantitatively validated against radiologist-annotated regions of interest.

**Reproducibility Limitations**

- cuDNN determinism and a fixed random seed are enforced, but exact bit-for-bit reproducibility across different GPU models/driver versions is not guaranteed.

## Recommendations

**Engineering Improvements**

- Persist each stage's own StageTimer duration into its JSON summary so Stage 8's runtime_summary.json does not have to leave Stages 1/2/3/5/6 execution time as 'unavailable'.
- Add a lightweight schema-version field to every stage's summary JSON to make future cross-stage compatibility checks (like the ones in Stage 8) explicit rather than implicit.

**Training Improvements**

- Investigate and, if needed, retrain/fine-tune specifically for the classes flagged DO_NOT_DEPLOY: ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'].
- Consider post-hoc calibration (e.g. temperature scaling) focused on the worst-calibrated class (Edema).

**Deployment Improvements**

- Switch production inference from the fixed 0.5 threshold to the per-class optimal thresholds already computed in optimal_thresholds.csv.

**Research Improvements**

- Run an external-cohort validation study to test generalization beyond the current single-dataset test split.
- Run a quantitative attribution-agreement study between explainability heatmaps and radiologist-marked regions of interest.

**Future Experiments**

- Class-balanced fine-tuning or focal-loss retraining for the lowest-support classes.
- Expand the explainability sample pool beyond the current bounded cap to improve statistical confidence in rare-class failure-mode analysis.

**Prioritized**

- [P0] Fix or retrain for DO_NOT_DEPLOY classes: ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
- [P1] Adopt per-class optimal thresholds in place of the fixed 0.5 cut-off.
- [P2] Recalibrate probabilities for Edema.
- [P2] Pursue external-cohort validation before broader deployment.

## Appendix: Artifact Inventory

| key | stage | kind | required | exists | file_size_bytes |
| --- | --- | --- | --- | --- | --- |
| environment_summary | stage01_environment | json | True | True | 1879 |
| artifact_validation | stage02_artifact_loader | json | True | True | 4353 |
| dataset_summary | stage03_dataset_builder | json | True | True | 2748 |
| dataset_validation | stage03_dataset_builder | json | True | True | 1357 |
| class_distribution_csv | stage03_dataset_builder | csv | False | True | 634 |
| inference_summary | stage04_inference_engine | json | True | True | 139270 |
| predictions_parquet | stage04_inference_engine | binary | True | True | 2014642 |
| raw_logits_pt | stage04_inference_engine | binary | True | True | 1511806 |
| prediction_metadata | stage04_inference_engine | json | True | True | 2265 |
| evaluation_summary | stage05_metrics_engine | json | True | True | 3035 |
| evaluation_metrics | stage05_metrics_engine | json | True | True | 10310 |
| macro_metrics | stage05_metrics_engine | json | True | True | 588 |
| micro_metrics | stage05_metrics_engine | json | True | True | 717 |
| weighted_metrics | stage05_metrics_engine | json | True | True | 502 |
| per_class_metrics_csv | stage05_metrics_engine | csv | True | True | 3358 |
| stage5_engineering_validation | stage05_metrics_engine | json | True | True | 5242 |
| stage06_summary | stage06_threshold_calibration_engine | json | True | True | 1809 |
| optimal_thresholds_csv | stage06_threshold_calibration_engine | csv | True | True | 2012 |
| calibration_summary_csv | stage06_threshold_calibration_engine | csv | True | True | 1015 |
| deployment_recommendations_csv | stage06_threshold_calibration_engine | csv | True | True | 1665 |
| roc_pr_threshold_curves_csv | stage06_threshold_calibration_engine | csv | False | True | 6062733 |
| probability_distribution_summary_csv | stage06_threshold_calibration_engine | csv | False | True | 5691 |
| pathology_report_csv | stage06_threshold_calibration_engine | csv | False | True | 1981 |
| stage6_engineering_validation | stage06_threshold_calibration_engine | json | True | True | 4567 |
| interpretability_summary | stage07_interpretability_engine | json | True | True | 4826 |
| explainability_metadata | stage07_interpretability_engine | json | False | True | 546 |
| selected_samples_csv | stage07_interpretability_engine | csv | True | True | 11107 |
| sample_predictions_csv | stage07_interpretability_engine | csv | True | True | 11107 |
| error_gallery_csv | stage07_interpretability_engine | csv | True | True | 109142 |
| gradcam_metadata_csv | stage07_interpretability_engine | csv | False | True | 10817 |
| integrated_gradients_csv | stage07_interpretability_engine | csv | False | True | 5442 |
| occlusion_summary_csv | stage07_interpretability_engine | csv | False | True | 5814 |
| guided_backprop_summary_csv | stage07_interpretability_engine | csv | False | True | 5072 |
| class_confusion_matrix_csv | stage07_interpretability_engine | csv | True | True | 1090 |
| stage7_engineering_validation | stage07_interpretability_engine | json | True | True | 4664 |
