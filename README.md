# SmartOdds Take-Home Project

This repository contains a complete, reproducible workflow for analysing horse-racing data and addressing all tasks specified in the SmartOdds assignment.

The project emphasises:

- clean modular code  
- strict time-ordered modelling (no leakage)  
- robust validation and testing  
- clear, defensible reasoning  
- a polished, well-structured final report  

---

## Repository Structure

```text
.
├── src/                    # Modular Python code
│   ├── cleaning.py         # Load data, clean, basic transforms
│   ├── features.py         # Feature construction (non-leaky)
│   ├── ratings.py          # Horse/jockey/trainer rating models
│   ├── model.py            # Winner-probability model
│   ├── evaluation.py       # Backtests & market comparison
│   └── utils.py            # Shared helpers
│
├── tests/                  # Automated tests & validation
│   ├── test_schema.py      # Columns, types, basic sanity checks
│   ├── test_invariants.py  # Logical invariants (prob sums, ranking)
│   ├── test_leakage.py     # Guards against future-data leakage
│   └── test_pipeline.py    # End-to-end integration tests
│
├── notebooks/              # Exploratory analysis
│   └── eda.ipynb
│
├── report/                 # Final written report & figures
│   ├── report.ipynb        # Main deliverable (HTML/PDF export)
│   └── figures/
│
├── data/
│   ├── raw/
│   │   └── test_dataset.csv
│   └── processed/          # Cleaned + feature datasets
│
├── requirements.txt
└── README.md
```

The **final report is written in parallel with the analysis**:  
each major step generates both code and narrative material for `report/report.ipynb`.

---

## 1. Data Cleaning & Integrity Validation

Goals:

- Produce a reliable, leakage-free cleaned dataset  
- Validate correctness using automated tests  

Steps:

1. Inspect schema, types, missingness, anomalies  
2. Clean & normalise fields (`age`, `draw`, weight fields, race distances, categorical codes)  
3. Remove or flag corrupted rows  
4. Write cleaned dataset into `data/processed/clean.csv`  
5. Tests:
   - schema conformity  
   - stable invariants (e.g., runner counts)  
   - absence of future leakage  

---

## 2. Exploratory Data Analysis

Goals:

- Understand distributional structure  
- Identify relationships relevant to peak-age analysis and modelling  

Outputs included in report:

- distributions of age by race type  
- performance vs age curves  
- jockey/trainer effects  
- going/distance influences  

---

## 3. Peak Age Analysis

Requirements:

- For each `race_type_simple`, estimate age vs. performance curves  
- Identify peak performance age  

Method options (justify in report):

- smoothed curve (LOESS / GAM)  
- spline regression  
- binned means with confidence intervals  

Output:

- peak age table  
- plots included in `report/figures/`  

---

## 4. Rating System Construction

Goals:

- Build time-ordered ratings for horses, jockeys, and trainers  
- No use of `obs__*` current-race variables (per assignment rules)  

Possible models (choose & justify):

- Elo-style update model  
- Bayesian hierarchical update  
- Exponential-decay performance index  

Outputs:

- ratings dataset saved in `data/processed/ratings.csv`  
- corresponding tests:
  - monotonic time ordering  
  - deterministic reproducibility  
  - proper update rules  

---

## 5. Winner-Probability Model

Design:

- features include ratings, race characteristics, historical form  
- exclude current-race `obs__*` variables  
- model could be:
  - multinomial logistic regression  
  - gradient boosted model  
  - neural softmax model  

Outputs:

- per-race predicted probability vector  
- test:
  - probability rows sum to 1  
  - no leakage from future races  

---

## 6. Market Comparison & Statistical Evaluation

Tasks:

- Compare model predictions to Betfair Starting Price (`obs__bsp`)  
- Assess profitability under zero-commission assumption  
- Evaluate:
  - log-loss vs market  
  - calibration curves  
  - Brier score  
  - ROC/AUC for win classification  
- Explore profitable subsets  
- Test statistical significance (bootstrap / permutation test)  

---

## 7. Final Report

Delivered as `report/report.ipynb` (exported to PDF or HTML).

Includes:

- motivation & overview  
- methodology (with equations where appropriate)  
- results for all four assignment questions  
- interpretation & reasoning  
- discussion of limitations and further work  

---

## 8. Reproducibility

- One-command environment setup using `requirements.txt`  
- Deterministic seeds in all modelling steps  
- Tests runnable via:

```text
pytest -q
```

---

## 9. Usage

Example workflow:

```text
python src/cleaning.py
python src/features.py
python src/ratings.py
python src/model.py
python src/evaluation.py
```

Full pipeline tests:

```text
pytest
```

---

## Contact

For questions about the project structure or execution, see notes in each module header.

