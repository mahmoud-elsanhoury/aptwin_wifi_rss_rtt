# aptwin_wifi_rss_rtt
Machine/deep learning AP-link digital twin for Wi-Fi RSS/RTT indoor positioning using geometry, floor-plan features, and multi-scenario evaluation.

# AP Digital Twin for Wi-Fi RTT/RSS Indoor Positioning

This repository contains the cleaned, standalone AP-link digital twin pipeline prepared from the IPIN 2026 six-environment notebook.

The repository converts the original notebook workflow into a reproducible GitHub-style scaffold. It builds AP-link datasets, extracts geometry and floor-plan-derived features, trains several machine learning and deep learning model families, computes final test metrics, summarizes multi-seed stability, and generates paper-ready CDF figures.

The focus is a data-driven AP-link digital twin for Wi-Fi RTT/RSS indoor positioning. The default pipeline intentionally excludes conventional/pathloss equation baselines for now. The current study is not a propagation-equation comparison. Those baselines can be added later as a separate diagnostic branch if needed.


## Dataset Attribution

This repository builds on the publicly available WiFi RTT/RSS indoor positioning dataset released by Feng, Nguyen, and Luo. The dataset contains WiFi Round-Trip Time (RTT) and Received Signal Strength (RSS) measurements collected in real indoor environments with ground-truth reference positions. In this repository, the dataset is used to construct AP-RP link-level digital twin experiments across the available indoor scenarios.

Please cite the original dataset and the associated journal article when using this repository or any results derived from it:

```bibtex
@dataset{feng2024wifi,
  author    = {Feng, Xu and Nguyen, Khuong An and Luo, Zhiyuan},
  title     = {WiFi RTT RSS dataset for indoor positioning},
  year      = {2024},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.11558192}
}

@article{feng2023wifi,
  title     = {WiFi round-trip time (RTT) fingerprinting: an analysis of the properties and the performance in non-line-of-sight environments},
  author    = {Feng, Xu and Nguyen, Khuong An and Luo, Zhiyuan},
  journal   = {Journal of Location Based Services},
  volume    = {17},
  number    = {4},
  pages     = {307--339},
  year      = {2023},
  publisher = {Taylor \& Francis},
  doi       = {10.1080/17489725.2023.2239748}
}

```

## 1. Scientific scope

The goal is to predict valid Wi-Fi Access Point to Reference Point measurements across six indoor scenarios.

The target variables are:

- **RSS**: Received Signal Strength, measured in dBm.
- **RTT**: Wi-Fi Round-Trip Time ranging estimate, represented in meters.

The input information includes:

- receiver coordinates,
- AP coordinates,
- AP-RP relative geometry,
- distance,
- LOS/NLOS indicators,
- wall-obstruction features,
- endpoint wall-context features,
- LAMS image-like floor-plan representations for the CNN branch.

The basic modeling idea is:

```text
AP position + RP position + geometry + floor-plan context  ->  RSS / RTT behavior
```

This is called an **AP digital twin** because the model learns how access points behave spatially across indoor reference points under different environmental layouts.

---

## 2. Six indoor scenarios

The benchmark contains six indoor scenarios from two dataset batches.

| Scenario | Batch | Coordinate block | General character |
|---|---:|---|---|
| `building` | batch 1 | floor | Large full-floor environment |
| `office` | batch 1 | office | Small mostly-LOS room |
| `apartment` | batch 1 | apartment | Small mixed LOS/NLOS apartment |
| `lecture_theatre` | batch 2 | lecture theatre | Large open/LOS-dominant environment |
| `corridor` | batch 2 | corridor | Long narrow NLOS-sensitive environment |
| `office2` | batch 2 | office | Medium mixed office environment |

The scenarios are deliberately diverse. This avoids evaluating the digital twin only in a single simple room and makes the benchmark more useful for cross-environment analysis.

---

## 3. Data protocol

The pipeline follows a strict train/validation/test protocol.

1. Keep the official train/test CSV files unchanged.
2. Detect structurally dead APs using training data only.
3. Convert wide scenario CSV files into long AP-link rows.
4. Exclude sentinel RSS/RTT values from regression targets.
5. Create validation data only from the official training split.
6. Use RP-level grouped validation to reduce repeated-scan leakage.
7. Fit all scalers on subtrain data only.
8. Use validation for model selection, representative-seed selection, and calibration.
9. Use test data only for final reporting.
10. Report scenario-macro metrics as the primary metrics because the dataset is scenario-imbalanced.

Invalid placeholder values are treated as missing targets:

```text
RSS = -200 dBm
RTT = 100000 m
```

These values are not used as valid regression targets.

---

## 4. Evaluation levels

The repository evaluates models at two levels.

### 4.1 Row level

Row-level evaluation uses repeated scan rows. Each row is an individual measurement instance.

This level preserves the measurement density of the original data and reflects scan-level prediction behavior.

### 4.2 Geometry level

Geometry-level evaluation aggregates repeated scan rows into unique AP-RP links.

This level answers the question:

```text
How well does the model predict the average behavior of each AP-RP link?
```

Both levels are important:

- **Row level** measures repeated-scan prediction behavior.
- **Geometry level** measures the quality of the AP-link digital twin itself.

---

## 5. Feature schemes

The repository uses a controlled feature ladder.

### 5.1 Geometry-only feature scheme

```text
RX_x
RX_y
AP_x
AP_y
dx
dy
distance
LOS_flag
LOS_known
```

Explanation:

- `RX_x`, `RX_y`: receiver/reference point coordinates.
- `AP_x`, `AP_y`: access point coordinates.
- `dx`, `dy`: AP-RP displacement components.
- `distance`: Euclidean AP-RP distance.
- `LOS_flag`: binary LOS/NLOS indicator when known.
- `LOS_known`: indicator showing whether LOS/NLOS information is available.

This feature scheme is used by the core geometry-only tabular models.

### 5.2 Wall-obstruction feature scheme

Additional features:

```text
wall_cross_count_total
wall_cross_count_internal
wall_cross_count_external
obstructed_path
```

Explanation:

- `wall_cross_count_total`: number of walls crossed by the AP-RP line segment.
- `wall_cross_count_internal`: number of internal walls crossed.
- `wall_cross_count_external`: number of external walls crossed.
- `obstructed_path`: binary indicator for whether the AP-RP path crosses at least one wall.

This feature scheme tests whether explicit floor-plan obstruction information improves RSS/RTT prediction.

### 5.3 Wall-context feature scheme

Additional endpoint-context features:

```text
AP_dist_nearest_wall
AP_dist_nearest_external_wall
AP_dist_nearest_internal_wall
RP_dist_nearest_wall
RP_dist_nearest_external_wall
RP_dist_nearest_internal_wall
AP_near_wall_1m
RP_near_wall_1m
```

Explanation:

- distance from the AP to the nearest wall,
- distance from the AP to the nearest external wall,
- distance from the AP to the nearest internal wall,
- distance from the RP to the nearest wall,
- distance from the RP to the nearest external wall,
- distance from the RP to the nearest internal wall,
- binary indicator for AP being near a wall,
- binary indicator for RP being near a wall.

This scheme adds local endpoint context around the AP and RP, not only the AP-RP line obstruction.

### 5.4 LAMS feature scheme

The CNN branch uses LAMS representations.

In this repository, **LAMS** refers to image-like AP-RP local spatial representations derived from floor-plan information. They are used to test whether an image-assisted branch can learn spatial propagation patterns that are not fully captured by tabular geometry features.

The LAMS branch is used by:

```text
training/train_cnn_lams.py
```

---

## 6. LOS/NLOS handling

The current main pipeline keeps:

```text
LOS_flag
LOS_known
```

These are treated as explicit floor-plan/context features.

They are retained because they are part of the original notebook feature protocol. However, they should be examined later through a separate diagnostic ablation, not by modifying the current core pipeline.

Recommended future diagnostic file:

```text
experiments/los_nlos_ablation.py
```

The diagnostic should compare:

```text
With LOS_flag and LOS_known
vs.
Without LOS_flag and LOS_known
```

The purpose is to quantify how strongly the digital twin depends on explicit LOS/NLOS labels. If removing LOS/NLOS causes only a small performance degradation, then geometry and wall-context features already capture much of the propagation structure. If the degradation is large, then explicit visibility information is important for accurate modeling.

This ablation should be interpreted as a sensitivity analysis, not automatically as a replacement for the main model.

---

## 7. Model families

The repository contains several model families.
*Important Remark:* Random Forest and XGBoost are treated as deterministic tabular regressors optimized for point prediction. They output predicted RSS/RTT values only and are therefore evaluated using point-error metrics such as RMSE, MAE, and P95. In contrast, the Gaussian MLP is formulated as a probabilistic regressor that predicts both the conditional mean and uncertainty. Therefore, it is trained using Gaussian negative log-likelihood and calibrated on the validation split. This distinction is intentional: RF/XGBoost serve as deterministic baselines, while the Gaussian MLP provides the calibrated probabilistic branch.

### 7.1 Random Forest

Script:

```text
training/train_rf.py
```

Purpose:

- classical machine learning baseline,
- geometry-only AP-link regression,
- multi-output prediction for RSS and RTT,
- useful as a strong non-neural reference model.

### 7.2 XGBoost

Script:

```text
training/train_xgboost.py
```

Purpose:

- strong tabular model family,
- compares geometry-only, wall-obstruction, and wall-context features,
- includes scenario-weighted variants,
- expected to be competitive for structured tabular AP-link data.

### 7.3 Gaussian MLP

Script:

```text
training/train_mlp.py
```

Purpose:

- neural tabular baseline,
- probabilistic output structure,
- validation-only calibration,
- includes raw pooled and scenario-weighted variants.

### 7.4 CNN/LAMS

Script:

```text
training/train_cnn_lams.py
```

Purpose:

- image-assisted floor-plan comparison family,
- tests whether LAMS representations help RSS/RTT prediction,
- includes LAMS-only and hybrid CNN variants.

### 7.5 GraphSAGE GNN

Script:

```text
training/train_gnn.py
```

Purpose:

- graph learning comparison,
- represents AP-RP relationships as graph edges,
- optional because it requires `torch_geometric`.

The GNN branch is optional. If `torch_geometric` is unavailable, the pipeline can be run with:

```bash
python run_pipeline.py --skip-optional
```

No fake GNN output should be produced if the dependency is unavailable.

---

## 8. Multi-seed protocol

The training scripts use a fixed multi-seed protocol:

```text
11, 22, 33, 44, 55
```

The purpose is to avoid relying on a single lucky or unlucky random run.

The reporting protocol is:

1. Train each stochastic model using the same fixed seed set.
2. Compute validation metrics for each seed.
3. Compute test metrics for each seed.
4. Report test mean ± standard deviation across seeds.
5. Use the validation-median seed only for representative CDF visualization.
6. Never select the best seed using test performance.

The representative seed is selected using validation scenario-macro RMSE. It is only used to choose one run for CDF visualization and selected-family plots.

Main claims should be based on the multi-seed mean ± standard deviation tables.

---

## 9. Metrics

The primary metric is:

```text
Scenario-macro RMSE
```

Secondary metrics are:

```text
Scenario-macro MAE
Scenario-macro P95
```

### 9.1 Why scenario-macro metrics?

The dataset is scenario-imbalanced. Some scenarios contain many more AP-link samples than others.

Micro metrics are computed over all samples at once and may be dominated by the largest scenario. Scenario-macro metrics first compute the metric separately per scenario, then average across scenarios.

This gives each scenario equal weight.

### 9.2 Metric definitions

| Metric | Meaning |
|---|---|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| P95 | 95th percentile absolute error |
| Scenario-macro | Average of scenario-wise metrics |
| Micro | Metric computed over all samples at once |

Bias and R² are not used for final ranking in the current multi-seed reporting layer.

---

## 10. Expected repository layout

```text
ap_locations/
    Floor+office+apartment_AP_coords.txt
    lecture theatre+office+corridor_AP_position.txt
    parse_ap_locations.py

data/
    first_batch/
        database_building_train.csv
        database_building_test.csv
        database_office_train.csv
        database_office_test.csv
        database_apartment_train.csv
        database_apartment_test.csv

    second_batch/
        database_lecture_theatre_train.csv
        database_lecture_theatre_test.csv
        database_corridor_train.csv
        database_corridor_test.csv
        database_office2_train.csv
        database_office2_test.csv

    build_dataset.py

floorplan_images/
    build_wall_features.py
    build_lams.py

training/
    common_training.py
    train_rf.py
    train_xgboost.py
    train_mlp.py
    train_cnn_lams.py
    train_gnn.py

performance_metrics/
    compute_metrics.py

plotting/
    plot_error_cdf.py

output_csvs/
    processed_features/
    predictions/
    audits/
    pipeline_checkpoints/

models/
    saved_models/
    registry/
    pretraining_setup/

results/
    tables/
    figures/

run_pipeline.py
```

---

## 11. Running the pipeline

First inspect the pipeline order:

```bash
python run_pipeline.py --dry-run
```

Run the default pipeline:

```bash
python run_pipeline.py
```

If checkpoints already exist, the script asks:

```text
1 = Continue from latest checkpoint
2 = Start from scratch
```

Option 1 resumes from completed checkpointed steps.

Option 2 starts a fresh run. Before rerunning, it archives old prediction CSV files into a timestamped folder under:

```text
output_csvs/predictions/
```

This prevents stale prediction files from mixing with new multi-seed outputs.

To skip the optional GNN branch:

```bash
python run_pipeline.py --skip-optional
```

To ignore checkpoints and rerun all selected steps:

```bash
python run_pipeline.py --force-rerun
```

To start manually from a specific step:

```bash
python run_pipeline.py --start-from train_xgboost
```

To list pipeline steps and checkpoint status:

```bash
python run_pipeline.py --list-steps
```

To clear checkpoints manually:

```bash
python run_pipeline.py --clear-checkpoints
```

---

## 12. Pipeline steps

| Step ID | Script | Purpose |
|---|---|---|
| `parse_ap_locations` | `ap_locations/parse_ap_locations.py` | Parse AP coordinate files |
| `build_dataset` | `data/build_dataset.py` | Build AP-link row/geometry datasets |
| `build_wall_features` | `floorplan_images/build_wall_features.py` | Add wall and endpoint context features |
| `build_lams` | `floorplan_images/build_lams.py` | Build LAMS representations |
| `train_rf` | `training/train_rf.py` | Train Random Forest models |
| `train_xgboost` | `training/train_xgboost.py` | Train XGBoost models |
| `train_mlp` | `training/train_mlp.py` | Train Gaussian MLP models |
| `train_cnn_lams` | `training/train_cnn_lams.py` | Train CNN/LAMS models |
| `train_gnn` | `training/train_gnn.py` | Train optional GraphSAGE GNN |
| `compute_metrics` | `performance_metrics/compute_metrics.py` | Compute final metrics and multi-seed summaries |
| `plot_error_cdf` | `plotting/plot_error_cdf.py` | Generate final CDF figures |

---

## 13. Processed dataset outputs

`data/build_dataset.py` writes:

```text
output_csvs/processed_features/all_ap_link_table.csv
output_csvs/processed_features/dead_ap_report.csv
output_csvs/processed_features/scenario_load_summary.csv
output_csvs/processed_features/dataset_audit_summary.csv
output_csvs/processed_features/imbalance_report.csv

output_csvs/processed_features/row_train.csv
output_csvs/processed_features/row_val.csv
output_csvs/processed_features/row_test.csv

output_csvs/processed_features/geometry_train.csv
output_csvs/processed_features/geometry_val.csv
output_csvs/processed_features/geometry_test.csv
```

The row-level files contain repeated scan rows.

The geometry-level files contain unique AP-RP link records.

---

## 14. Wall-feature outputs

`floorplan_images/build_wall_features.py` writes wall-aware feature tables, including:

```text
row_train_wall_obstruction.csv
row_val_wall_obstruction.csv
row_test_wall_obstruction.csv

row_train_wall_context.csv
row_val_wall_context.csv
row_test_wall_context.csv

geometry_train_wall_obstruction.csv
geometry_val_wall_obstruction.csv
geometry_test_wall_obstruction.csv

geometry_train_wall_context.csv
geometry_val_wall_context.csv
geometry_test_wall_context.csv
```

These files are used mainly by wall-aware XGBoost variants and related feature-ladder comparisons.

---

## 15. LAMS outputs

`floorplan_images/build_lams.py` writes LAMS datasets and metadata used by the CNN/LAMS branch.

Typical outputs include:

```text
floorplan_images/lams/
output_csvs/processed_features/*lams*.csv
models/pretraining_setup/*lams*.json
```

The exact file names may depend on the current LAMS export configuration.

---

## 16. Prediction outputs

Training scripts write prediction CSV files under:

```text
output_csvs/predictions/
```

The multi-seed convention is:

```text
*_seed11_*_test_predictions.csv
*_seed22_*_test_predictions.csv
*_seed33_*_test_predictions.csv
*_seed44_*_test_predictions.csv
*_seed55_*_test_predictions.csv
```

Representative prediction files use:

```text
*_representative_*_test_predictions.csv
```

Seed-specific files are used for mean ± standard deviation reporting.

Representative files are used for final CDF visualization.

---

## 17. Final metric outputs

`performance_metrics/compute_metrics.py` writes:

```text
results/tables/all_candidate_metrics_test.csv
results/tables/selected_best_model_per_family_test.csv
results/tables/detailed_micro_scenario_macro_metrics_test.csv
results/tables/prediction_file_audit.csv
results/tables/multiseed_prediction_file_audit.csv
results/tables/multiseed_test_metrics_all_seeds.csv
results/tables/multiseed_test_summary_mean_std.csv

output_csvs/predictions/_prediction_registry_long.csv
```

Key files:

| File | Meaning |
|---|---|
| `all_candidate_metrics_test.csv` | Metrics for all representative candidate models |
| `selected_best_model_per_family_test.csv` | One selected representative per family, level, and target |
| `detailed_micro_scenario_macro_metrics_test.csv` | Micro, scenario-wise, and scenario-macro details |
| `multiseed_test_metrics_all_seeds.csv` | Scenario-macro metrics for each seed |
| `multiseed_test_summary_mean_std.csv` | Mean ± std across seeds |
| `_prediction_registry_long.csv` | Normalized long-format prediction registry for plotting |

---

## 18. Final figure outputs

`plotting/plot_error_cdf.py` writes:

```text
results/figures/row_level_best_family_abs_error_cdf.png
results/figures/row_level_best_family_abs_error_cdf.pdf

results/figures/geometry_level_best_family_abs_error_cdf.png
results/figures/geometry_level_best_family_abs_error_cdf.pdf

results/tables/cdf_selected_models_used.csv
```

The CDF plots show empirical absolute-error distributions for the selected family representatives.

---

## 19. Checkpoint behavior

The pipeline stores checkpoint files under:

```text
output_csvs/pipeline_checkpoints/
```

After each successful step, a checkpoint JSON file is written.

If the pipeline stops because of an error, fix the error and rerun:

```bash
python run_pipeline.py
```

The pipeline resumes from the failed step.

When choosing a fresh restart through the interactive prompt, old prediction CSV files are archived before rerunning. This protects final metric computation from stale prediction files.

---

## 20. Important methodological notes

- The official train/test split is preserved.
- Validation is created only from training data.
- RP-level validation grouping is used to reduce repeated-scan leakage.
- Scalers are fitted on subtrain only.
- Test data are used only for final reporting.
- Scenario-macro RMSE is the primary ranking metric.
- Scenario-macro MAE and P95 are secondary metrics.
- Micro metrics are secondary because scenario sample counts are imbalanced.
- CNN/LAMS and GNN are retained as comparison families, not forced headline results.
- Weaker deep learning results should be reported honestly if they occur.
- LOS/NLOS ablation should be added later as a separate diagnostic experiment.
- Conventional/pathloss baselines are excluded from the current default pipeline and should not be mixed into the main data-driven comparison unless added as a separate controlled branch.

---

## 21. Dependencies

Core dependencies:

```text
numpy
pandas
scikit-learn
matplotlib
joblib
```

Model-specific dependencies:

```text
xgboost
torch
torchvision
torch_geometric
```

The GNN branch requires `torch_geometric`. If unavailable, use:

```bash
python run_pipeline.py --skip-optional
```

---

## 22. Reproducibility notes

The repository is designed to avoid notebook-memory dependency.

Each stage is a standalone script. Scripts communicate through saved CSV files, model files, metadata, and result tables.

The pipeline avoids:

- hidden notebook state,
- test-based seed selection,
- fabricated missing outputs,
- silent skipping of failed required stages,
- mixing stale prediction files with fresh reruns,
- choosing a best test seed.

---

## 23. Suggested citation

If you use this repository, please cite the associated article once it is published.

Temporary BibTeX entry:

```bibtex
@article{elsanhoury2026aptwin,
  title   = {AP-Twin: Generative Indoor Wi-Fi RTT and RSS Measurements at Known Reference Points Using ML/DL Algorithms and Floorplan Images},
  author  = {Elsanhoury, Mahmoud and others},
  journal = {To be updated after publication},
  year    = {2026},
  note    = {Code repository: AP Digital Twin for Wi-Fi RTT/RSS Indoor Positioning}
}
```

After publication, we will replace the placeholder fields with the final paper title, full author list, venue, DOI, and repository URL.

---

## 24. Repository status

This scaffold is prepared for the IPIN 2026 AP Digital Twin study.

The current default pipeline supports:

- six-scenario AP-link dataset construction,
- row-level and geometry-level evaluation,
- geometry-only and floor-plan-aware feature schemes,
- Random Forest, XGBoost, Gaussian MLP, CNN/LAMS, and optional GraphSAGE GNN model families,
- fixed multi-seed training,
- scenario-macro metric reporting,
- representative-seed CDF visualization,
- checkpointed full-pipeline execution,
- automatic archiving of stale prediction CSVs before fresh reruns.
