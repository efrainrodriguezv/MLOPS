# AutoML results

**Data:** 24,023 train / 6,006 test rows (same split as Part A) · 14 features · target `total_lift`
**Budgets:** FLAML 300s per run · H2O 300s per run · validation = 5-fold CV RMSE on train

## Platform (MLflow + FLAML)
- Best model: **xgb_limitdepth** · CV RMSE 143.4 lb
- Top-5 features (permutation): gender_male, age, bmi, weight, howlong_ord
- Top-3 subset: gender_male, age, bmi

### Top-3 by validation score — all features
|    | model          |   val_rmse |   test_rmse |   test_r2 |   fit_s |   predict_ms_per_1k |
|---:|:---------------|-----------:|------------:|----------:|--------:|--------------------:|
|  1 | xgb_limitdepth |    143.351 |     143.891 |     0.730 |   0.322 |               0.732 |
|  2 | xgboost        |    143.367 |     143.955 |     0.730 |   2.997 |               3.416 |
|  3 | lgbm           |    143.663 |     144.208 |     0.729 |   0.339 |               1.243 |

### Top-3 by validation score — top-3 features
|    | model          |   val_rmse |   test_rmse |   test_r2 |   fit_s |   predict_ms_per_1k |
|---:|:---------------|-----------:|------------:|----------:|--------:|--------------------:|
|  1 | xgb_limitdepth |    167.165 |     167.557 |     0.634 |   0.189 |               0.438 |
|  2 | xgboost        |    167.203 |     167.727 |     0.634 |   0.214 |               0.407 |
|  3 | rf             |    167.411 |     167.622 |     0.634 |   0.317 |               4.520 |

### Top-3 by speed (fit time) — all features
|    | model      |   val_rmse |   test_rmse |   test_r2 |   fit_s |   predict_ms_per_1k |
|---:|:-----------|-----------:|------------:|----------:|--------:|--------------------:|
|  1 | lassolars  |    153.037 |     152.468 |     0.697 |   0.003 |               0.086 |
|  2 | kneighbor  |    179.876 |     179.832 |     0.579 |   0.006 |               6.625 |
|  3 | extra_tree |    154.894 |     155.953 |     0.683 |   0.030 |               2.345 |

### Top-3 by speed (fit time) — top-3 features
|    | model     |   val_rmse |   test_rmse |   test_r2 |   fit_s |   predict_ms_per_1k |
|---:|:----------|-----------:|------------:|----------:|--------:|--------------------:|
|  1 | enet      |    178.640 |     177.259 |     0.591 |   0.003 |               0.070 |
|  2 | kneighbor |    199.037 |     200.908 |     0.474 |   0.004 |               2.610 |
|  3 | lassolars |    178.640 |     177.259 |     0.591 |   0.005 |               0.053 |

## H2O AutoML
- Leader: **StackedEnsemble_AllModels_6_AutoML_1_20261109_171518** · best single model GBM_grid_1_AutoML_1_20261109_171518_model_19
- Top-5 features: gender_male, bmi, height, weight, age

### Top-3 by validation score — all features
|    | model_id                                             | algo            |   val_rmse |   test_rmse |   test_r2 |   fit_s |   predict_ms_per_1k |
|---:|:-----------------------------------------------------|:----------------|-----------:|------------:|----------:|--------:|--------------------:|
|  1 | StackedEnsemble_AllModels_6_AutoML_1_20261109_171518 | StackedEnsemble |    143.062 |     143.687 |     0.731 |   1.130 |              33.364 |
|  2 | StackedEnsemble_AllModels_4_AutoML_1_20261109_171518 | StackedEnsemble |    143.265 |     144.029 |     0.730 |   0.337 |               9.615 |
|  3 | StackedEnsemble_Best1000_1_AutoML_1_20261109_171518  | StackedEnsemble |    143.267 |     144.033 |     0.730 |   0.228 |               9.573 |

### Top-3 by validation score — top-3 features
|    | model_id                                                | algo            |   val_rmse |   test_rmse |   test_r2 |   fit_s |   predict_ms_per_1k |
|---:|:--------------------------------------------------------|:----------------|-----------:|------------:|----------:|--------:|--------------------:|
|  1 | StackedEnsemble_BestOfFamily_7_AutoML_2_20261109_172044 | StackedEnsemble |    174.809 |     174.743 |     0.602 |   0.265 |               6.459 |
|  2 | StackedEnsemble_Best1000_1_AutoML_2_20261109_172044     | StackedEnsemble |    174.811 |     174.671 |     0.603 |   0.440 |              15.620 |
|  3 | GBM_grid_1_AutoML_2_20261109_172044_model_57            | GBM             |    174.845 |     174.699 |     0.603 |   0.069 |               2.875 |

### Top-3 by speed (fit time, excl. stacked ensembles) — all features
|    | model_id                                             | algo         |   val_rmse |   test_rmse |   test_r2 |   fit_s |   predict_ms_per_1k |
|---:|:-----------------------------------------------------|:-------------|-----------:|------------:|----------:|--------:|--------------------:|
|  1 | GBM_grid_1_AutoML_1_20261109_171518_model_61         | GBM          |    274.882 |     274.563 |     0.019 |   0.003 |               0.153 |
|  2 | GBM_grid_1_AutoML_1_20261109_171518_model_60         | GBM          |    181.370 |     181.390 |     0.572 |   0.030 |               1.012 |
|  3 | DeepLearning_grid_1_AutoML_1_20261109_171518_model_7 | DeepLearning |    199.434 |     158.050 |     0.675 |   0.048 |               1.065 |

### Top-3 by speed (fit time, excl. stacked ensembles) — top-3 features
|    | model_id                                              | algo         |   val_rmse |   test_rmse |   test_r2 |   fit_s |   predict_ms_per_1k |
|---:|:------------------------------------------------------|:-------------|-----------:|------------:|----------:|--------:|--------------------:|
|  1 | GLM_1_AutoML_2_20261109_172044                        | GLM          |    183.887 |     182.457 |     0.567 |   0.019 |               0.071 |
|  2 | GBM_grid_1_AutoML_2_20261109_172044_model_62          | GBM          |    207.207 |     209.445 |     0.429 |   0.021 |               0.673 |
|  3 | DeepLearning_grid_1_AutoML_2_20261109_172044_model_19 | DeepLearning |    199.852 |     188.823 |     0.536 |   0.049 |               0.623 |

## vs Part A (RandomForest · v2 · hp_light)
| model                          |   val_rmse |   test_rmse |   test_r2 |   fit_s |   predict_ms_per_1k |   Δ val_rmse vs Part A |   speed-up vs Part A (fit) |
|:-------------------------------|-----------:|------------:|----------:|--------:|--------------------:|-----------------------:|---------------------------:|
| Part A · RF v2 hp_light        |    146.493 |     146.900 |     0.719 |   0.228 |               2.392 |                  0.000 |                      1.000 |
| FLAML · xgb_limitdepth (all)   |    143.351 |     143.891 |     0.730 |   0.322 |               0.732 |                 -3.143 |                      0.707 |
| FLAML · xgboost (all)          |    143.367 |     143.955 |     0.730 |   2.997 |               3.416 |                 -3.126 |                      0.076 |
| FLAML · lgbm (all)             |    143.663 |     144.208 |     0.729 |   0.339 |               1.243 |                 -2.830 |                      0.673 |
| FLAML · xgb_limitdepth (top-3) |    167.165 |     167.557 |     0.634 |   0.189 |               0.438 |                 20.671 |                      1.206 |
| H2O · StackedEnsemble (all)    |    143.062 |     143.687 |     0.731 |   1.130 |              33.364 |                 -3.431 |                      0.202 |
| H2O · StackedEnsemble (top-3)  |    174.809 |     174.743 |     0.602 |   0.265 |               6.459 |                 28.316 |                      0.860 |
