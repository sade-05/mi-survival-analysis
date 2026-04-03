# Myocardial Infarction: Survival Analysis
### Kaplan-Meier Curves · Cox Proportional Hazards · Model Diagnostics

[![R](https://img.shields.io/badge/R-4.3%2B-276DC3?logo=r&logoColor=white)](https://www.r-project.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset: UCI](https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-blue)](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications)

---

## Overview

This project applies survival analysis to a dataset of **1,700 myocardial infarction
patients** admitted to Krasnoyarsk Interdistrict Clinical Hospital No. 20
(Russia, 1992–1995). The goal is to examine not just *who* dies during the hospital
stay, but *when*, and which patient characteristics are most strongly associated
with the timing of death.

The analysis uses five predictors identified as the strongest independent risk
factors in a companion logistic regression analysis:

- Age group
- Obesity history
- Bronchial asthma history
- Chronic heart failure history
- Right ventricular myocardial infarction

The dataset is publicly available from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications)
(DOI: [10.24432/C53P5M](https://doi.org/10.24432/C53P5M)).

---

## Methods

### Kaplan-Meier estimation

The **Kaplan-Meier estimator** calculates the probability of surviving past each
observed event time, accounting for patients who are right-censored (discharged
alive before end of follow-up):
```
S(t) = ∏ (1 - dᵢ / nᵢ)  for all tᵢ ≤ t
```

KM curves are plotted separately for each of the five key predictors.
Group differences are tested with the **log-rank test** (H₀: survival curves
are equal across groups). A p-value < 0.05 indicates a statistically significant
difference in survival.

### Cox proportional hazards model

The **Cox model** adjusts for all five predictors simultaneously:
```
h(t | X) = h₀(t) × exp(β₁·age_group + β₂·obesity + β₃·asthma
                        + β₄·chf_history + β₅·right_vent_mi)
```

where h₀(t) is the unspecified baseline hazard (semiparametric). Exponentiating
each coefficient gives a **hazard ratio (HR)**: HR > 1 increases risk of death;
HR < 1 is protective. The proportional hazards assumption is tested using
Schoenfeld residuals via `cox.zph()`.

### Diagnostics

- **ROC curve** — the Cox linear predictor is used as a continuous risk score;
  AUC summarises overall discrimination
- **Calibration plot** — predicted probabilities vs. observed death rates across
  ten bins; points on the diagonal indicate perfect calibration
- **Variable importance** — predictors ranked by absolute log-hazard coefficient

---

## Outputs

| File | Description |
|---|---|
| `figures/km_age_group.png` | KM curves by age group |
| `figures/km_obesity.png` | KM curves by obesity history |
| `figures/km_right_vent_mi.png` | KM curves by right ventricular MI |
| `figures/cox_forest_plot.png` | Cox hazard ratios with 95% CI |
| `figures/variable_importance.png` | Predictors ranked by log-hazard magnitude |
| `figures/roc_curve.png` | ROC curve with AUC |
| `figures/calibration_plot.png` | Predicted vs. observed event rates |
| `outputs/cox_results.csv` | Cox model coefficient table |
| `outputs/descriptive_table.csv` | Patient cohort summary by outcome |

---

## How to Run

**1. Install packages**
```r
install.packages(c("tidyverse", "survival", "broom", "pROC"))
```

**2. Run the script**

The dataset downloads automatically from UCI — no manual download needed.
```r
source("R/mi_survival_simple.R")
```

Figures save to `figures/` and tables to `outputs/` automatically.

---

## Dependencies

| Package | Purpose |
|---|---|
| `tidyverse` | Data wrangling and ggplot2 visualisations |
| `survival` | `Surv()`, `survfit()`, `coxph()`, `cox.zph()`, `basehaz()` |
| `broom` | Tidy model output for plots and tables |
| `pROC` | ROC curve and AUC |

---

## Citation

> Golovenkin, S., Shulman, V., Rossiev, D., Shesternya, P., Nikulina, S.,
> Orlova, Y., & Voino-Yasenetsky, V. (2020). *Myocardial infarction complications*
> [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53P5M

---

## License

Code: [MIT License](LICENSE) | Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
