# Dashboard data: malaria detection with CNNs

Small JSON exports of the key results from this repo's notebooks, for building
interactive dashboards. Every number comes from the notebook outputs; nothing
is invented. See `source_notebook` in each file for provenance.

## model_evaluation.json
From `web/analysis.ipynb`: honest evaluation of the two shipped CNNs.

- `dataset`: name, image count, class balance, and the caveat that no train/test split is recorded (numbers are an upper bound).
- `models`: `my_model` and `malaria_cnn`, each with `accuracy`, 95% CI (`accuracy_ci95`), `auc`, `sensitivity`, `specificity`, `preprocess` description, and notes.
- `bootstrap`: method and resample count behind the CIs.
- `calibration`: expected calibration error (`ece_my_model`) over 15 bins, with a note on the direction of miscalibration.
- `test_time_augmentation`: `n_cells`, `views_per_cell`, mean SD of P(infected), mean 95% interval width overall and split by correct/misclassified.
- `error_analysis`: `n_errors` plus per-property (brightness, contrast, saturation, redness) means and SDs for misclassified vs correctly classified cells.

## data_cleaning.json
From `pages/data_cleaning.ipynb`: cleaning the WHO-style incidence table.

- `input`, `columns`: source CSV and its columns.
- `ranges`: max values seen (`max_country_alpha`, `max_year`, `max_cases`, `max_deaths`, `max_deaths_region`).
- `cleaning`: imputation method (`missing_value_fill`), the median used, and null counts after filling.
- `who_regions`: the six regions present.
- `south_east_asia_subset`: row/column counts, year range, columns.
- `outputs`: the cleaned CSVs this notebook writes.

## image_processing.json
From `notebooka4911bf708.ipynb`: the exploratory image-processing pass.

- `dataset`: source, class loaded, image count, resize, and the note that the data is not in the repo.
- `pipeline`: the processing steps in order.
- `result_single_cell`: contours found and the area-ratio percentage on the one demo image.
