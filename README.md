SmartOdds Take-Home Project

This repository contains a complete, reproducible workflow for analysing horse-racing data and addressing all tasks specified in the SmartOdds take-home assignment.
The project emphasises:

clean modular code

strict time-ordered modelling (no leakage)

robust validation and testing

clear, defensible reasoning

a polished, well-structured final report

Project Plan
0. Repository Structure
.
├── src/                 # Modular Python code
│     ├── cleaning.py
│     ├── features.py
│     ├── ratings.py
│     ├── model.py
│     ├── evaluation.py
│     ├── utils.py
│
├── tests/               # Automated tests & validation
│     ├── test_schema.py
│     ├── test_invariants.py
│     ├── test_leakage.py
│     ├── test_ratings.py
│     ├── test_softmax.py
│     ├── test_end_to_end.py
│
├── notebooks/           # Exploratory analysis notebooks
│     └── eda.ipynb
│
├── report/              # Final report + figures
│     ├── report.ipynb (or report.qmd/report.md)
│     ├── figures/
│
├── data/
│     ├── raw/
│     └── processed/
│
├── requirements.txt
└── README.md


The final report is developed in parallel throughout the project.

1. Data Cleaning & Integrity Validation (Q0 Foundation)
Objectives

Build a reliable, leakage-free cleaned dataset.

Validate structural consistency and domain constraints.

Key Steps

Parse and enforce data types.

Validate race-level invariants:

shared date, venue, distance, going, race type

row count matches n_runners

Standardise handling of non-finishers (PU, UR, F, etc.).

Detect and treat anomalous or impossible values.

Guarantee strict temporal ordering for all features.

Save cleaned dataset to data/processed/.

Testing

Schema tests

Uniqueness tests (race_id, horse_id)

Race-level invariant tests

Finishing position validity tests

Temporal leakage tests

Report (in parallel)

Document cleaning logic and summarise dataset characteristics.

2. Exploratory Data Analysis (Q0 Question)
Objectives

Understand data structure and identify modelling signals.

Analyses

Distributions of age, weight, official ratings, BSP, distances, going.

Outcome patterns (win rate by BSP decile, finishing position behaviour).

Basic correlations + conditional summaries.

Report

Include key plots and concise commentary.

Highlight insights relevant to later modelling.

3. Peak Age Analysis (Q1)
Objectives

Determine age ranges where horses exhibit peak performance per race discipline.

Method

Define a normalised performance metric, e.g.:

Perf=1−finish_position−1n_runners−1
Perf=1−
n_runners−1
finish_position−1
	​


Compute mean performance by age × race_type_simple.

Fit smooth curves (LOESS/GAM).

Identify peak ages across Flat / Hurdle / Chase.

Validation

Sensitivity checks to metric choice and smoothing.

Sanity checks (e.g., younger peak ages on Flat, older in Chases).

Report

Performance curves and interpretation.

4. Time-Ordered Rating Systems (Q2)

(Horse, jockey, and trainer ratings)

Objectives

Construct information-accumulating rating systems usable as predictive features.

Horse Ratings

Recency-weighted update:

Rt+1=α⋅Perft+(1−α)Rt
R
t+1
	​

=α⋅Perf
t
	​

+(1−α)R
t
	​


Optional distance- or going-specific variants.

Strict chronological updates to avoid leakage.

Jockey & Trainer Ratings

Rolling/recency-weighted aggregates with shrinkage to global means.

Stabilise small-sample behaviour.

Testing

Deterministic synthetic tests verifying rating updates.

Temporal ordering tests.

Distributional sanity checks.

Report

Document rating formulas, examples, and trajectories.

5. Predictive Modelling (Q3)
Objective

Predict win probabilities for each runner pre-race.

Model

Create a feature set using:

horse, jockey, trainer ratings

official rating

age, weight, draw

race attributes

Fit a race-wise softmax model:

P(i)=esi∑j∈raceesj
P(i)=
∑
j∈race
	​

e
s
j
	​

e
s
i
	​

	​


Alternative: Plackett–Luce ranking model.

Training Protocol

Forward-chaining CV (time-respecting folds).

Metrics: log-loss, Brier score, calibration curves, decile reliability.

Testing

Probabilities sum to 1 per race.

Higher scores correlate with higher win rates.

Strictly no leakage from future races.

Report

Include model explanation, calibration plots, and evaluation results.

6. Market Comparison & Efficiency Analysis (Q4)
Objective

Compare model predictions to betting market (BSP) implied probabilities.

Steps

Compute implied probability:

pBSP=1BSP
p
BSP
	​

=
BSP
1
	​


Compare model vs market calibration.

Identify value bets (overlays).

Compute hypothetical profit (zero commission).

Use bootstrap to assess statistical significance.

Testing

Validate overlay logic and return calculations.

Consistency checks across time segments.

Report

Plots and tables comparing model and market efficiency.

Profitability curves and interpretation.

7. Limitations & Future Work

Discuss limitations such as:

Small-sample instability for connections

Noise in finishing positions

Simplifying assumptions in ratings and models

Potential extensions: hierarchical Bayesian ratings, BT models, pace/sectional data

8. Reproducibility & Packaging
Deliverables

Complete modular codebase (src/)

Automated test suite (tests/)

Cleaned datasets (data/processed/)

Final report (HTML/PDF)

Environment file (requirements.txt)

A single zipped submission as required

Pipeline

Run full workflow: cleaning → ratings → modelling → evaluation

Ensure all tests pass

Regenerate report deterministically

Package everything into the final archive

Status

This document defines the full development plan.
Implementation proceeds sequentially, with the final report built continuously throughout the project.# SmartOdds-Task
