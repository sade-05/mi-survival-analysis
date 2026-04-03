# Predicting Myocardial Infarction Fatality
### A Capstone in Logistic Regression and Survival Analysis

[![R](https://img.shields.io/badge/R-4.3%2B-276DC3?logo=r&logoColor=white)](https://www.r-project.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset: UCI](https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-blue)](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications)

> This project was developed with AI assistance for code structure and debugging.
> All analytical decisions, statistical methodology, and interpretations are the author's own.

---

## Overview

Myocardial infarction (MI) remains one of the leading causes of in-hospital mortality worldwide. While the clinical risk factors for experiencing a heart attack are well established, predicting which patients will die during their hospital stay — and understanding the timing of that risk — is a harder and more clinically consequential problem.

This capstone project uses a dataset of **1,700 MI patients** admitted to Krasnoyarsk Interdistrict Clinical Hospital No. 20 (Russia, 1992–1995) to build and validate a predictive model for in-hospital mortality. The analysis is structured in two parts:

**Part 1 — Logistic Regression:** Starting from 111 raw clinical variables, the analysis engineers 49 meaningful predictors and applies backward stepwise selection to identify a parsimonious reduced model that reliably predicts in-hospital death.

**Part 2 — Survival Analysis:** The strongest predictors from Part 1 are carried forward into a survival analysis framework, which adds the dimension of time — examining not just who dies, but when, and whether the same risk factors operate consistently across the hospital stay.

The dataset is publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications) (DOI: [10.24432/C53P5M](https://doi.org/10.24432/C53P5M)).

---

## The Dataset

| | |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **Patients** | 1,700 |
| **Input features** | 111 (medical history, ECG results, blood tests, medications, hospital day status) |
| **Outcome variable** | `LET_IS`: 0 = survived; 1–7 = seven distinct causes of death |
| **Missing data** | ~7.6% across all variables |

The outcome variable `LET_IS` records not just whether a patient died, but the specific cause — cardiogenic shock, pulmonary oedema, myocardial rupture, and others. For Part 1 this is collapsed to a binary outcome (survived vs. died). For Part 2 the time structure embedded in the day 1/2/3 status variables is used to construct a survival time proxy.

> **On class imbalance:** 86.5% of patients survived. A degenerate model that always predicts survival would achieve 86.5% accuracy without learning anything — this is the No Information Rate (NIR). Any model must meaningfully exceed this on clinically relevant metrics, particularly sensitivity for detecting death, to be considered useful. This project addresses imbalance explicitly using inverse-frequency class weighting.

---

## Patient Cohort Summary

Before any modelling, the table below summarises the patient population split by outcome. It gives immediate context about who these patients are and how the two groups differ in their baseline characteristics.

| Variable | Survived | Died |
|---|---|---|
| N | 1,429 | 271 |
| Age — mean (SD) | — | — |
| Male (%) | — | — |
| Obesity (%) | — | — |
| Bronchial asthma (%) | — | — |
| Diabetes (%) | — | — |
| Chronic heart failure (%) | — | — |
| Right ventricular MI (%) | — | — |

> Values populate automatically when `mi_survival_simple.R` runs and saves `outputs/descriptive_table.csv`.

---

## Part 1 — Logistic Regression

### The modelling framework

In-hospital death is a binary outcome, making **multiple logistic regression** the natural modelling choice. The model estimates the log-odds of dying as a linear function of patient characteristics:

```
log( P(death) / 1 - P(death) ) = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ
```

Each coefficient βⱼ represents the change in log-odds of death associated with a one-unit increase in predictor Xⱼ, with all other predictors held constant. Exponentiating the coefficients gives **odds ratios (OR)** — the effect size measure used throughout this analysis. An OR > 1 indicates increased odds of death; an OR < 1 indicates a protective effect.

### Feature engineering

The raw dataset contains 111 variables, many of which are redundant, clinically related, or too granular to model directly. The preprocessing phase reduces these to **49 engineered predictors** through three operations:

**Risk categorisation** — continuous clinical measurements are binned into medically meaningful risk tiers. For example, raw serum sodium values are converted to: hyponatremia (< 135 mmol/L), normal (135–145 mmol/L), and hypernatremia (> 145 mmol/L), based on established clinical reference ranges. Enzyme values originally recorded in µkat/L were converted to IU/L using the standard conversion factor (÷ 0.0167) before categorisation.

**Variable combination** — related binary indicators are collapsed into ordered categorical variables. For example, seven separate fibrinolytic therapy indicators are combined into a single ordinal variable reflecting treatment dose and severity.

**Single-level removal** — after the train/test split, any factor variable retaining only one level in the complete-case training subset is dropped. Such variables carry no discriminative information and cause a matrix singularity error in `glm()`.

### From full model to reduced model

The starting point is the **full model** — a logistic regression fitted on all 49 engineered predictors simultaneously:

```
logit(P(death)) = β₀ + β₁X₁ + β₂X₂ + ... + β₄₉X₄₉
```

Fitting all 49 predictors at once risks **overfitting** — the model learns noise in the training data rather than genuine signal, and performs poorly on unseen patients. It also produces coefficient estimates that are difficult to interpret when predictors are correlated.

**Backward stepwise selection via AIC** is used to move from the full model to a reduced model. The process works as follows: beginning with all 49 predictors, the algorithm removes one predictor at a time, each time selecting the predictor whose removal results in the lowest AIC (Akaike Information Criterion). AIC balances model fit against complexity:

```
AIC = -2 × log-likelihood + 2k
```

where k is the number of parameters. A lower AIC indicates a better trade-off between fit and parsimony. The algorithm stops when removing any remaining predictor would increase the AIC — meaning no further simplification improves the model. This is implemented in R via `MASS::stepAIC()`.

The backward direction was chosen over forward selection because the full model was estimable despite missing data (handled by `na.action = na.omit`), and backward selection tends to be more stable when predictors are correlated.

### The reduced model

Backward stepwise selection converged on **18 predictors**. The reduced model takes the form:

```
logit(P(LET_IS_new = 1)) =
  β₀
  + β₁  · age_group
  + β₂  · IM_PG_P          (right ventricular MI)
  + β₃  · zab_leg_03        (bronchial asthma history)
  + β₄  · endocr_02         (obesity history)
  + β₅  · TIME_B_S          (time from attack onset to hospital)
  + β₆  · ZSN_A             (chronic heart failure history)
  + β₇  · GB                (essential hypertension)
  + β₈  · RBBB              (right bundle branch block)
  + β₉  · inf               (inferior MI on ECG)
  + β₁₀ · ECG_HR            (heart rate at admission)
  + β₁₁ · premature_VC      (premature ventricular contractions)
  + β₁₂ · WBC               (white blood cell count)
  + β₁₃ · first_day_hosp    (complications on day 1)
  + β₁₄ · med_ICU           (ICU medication intensity)
  + β₁₅ · angina            (chest pain history)
  + β₁₆ · lethal_early      (cardiogenic shock / pulmonary oedema at ICU admission)
  + β₁₇ · endocr_03         (thyrotoxicosis history)
  + β₁₈ · paroxy_tachy      (paroxysmal tachycardia)
```

Of the 18 retained predictors, **10 were statistically significant** at α = 0.05 based on Wald z-tests. Variables such as `TIME_B_S`, `GB`, `RBBB`, and `paroxy_tachy` were retained by the AIC criterion but did not individually reach significance — a known property of stepwise selection, where variables can be retained for their joint contribution rather than their marginal p-value alone.

### Handling class imbalance

With only 13.5% of patients dying, the model risks systematically under-predicting death. Rather than resampling — which fabricates or discards data — **inverse-frequency class weights** are passed directly to `glm()`. Each observation is weighted by:

```
w_i = N / (K × nₖ)
```

where N is the total number of complete-case training observations, K = 2 (number of classes), and nₖ is the count of the class that observation i belongs to. This amplifies the log-likelihood contribution of each death observation proportionally without altering the data itself.

### Model evaluation

The model is evaluated on a held-out test set (30% of the data, stratified by outcome). The classification threshold is optimised using **Youden's index**:

```
J = Sensitivity + Specificity - 1
```

The threshold that maximises J is preferable to the default 0.5 in imbalanced settings because it explicitly balances the cost of missing a true death (false negative) against a false alarm (false positive).

Model calibration is assessed using the **Hosmer-Lemeshow test**, which partitions predicted probabilities into deciles and compares observed vs. expected event counts via a chi-squared statistic. A p-value > 0.05 indicates adequate calibration. Internal validation uses **10-fold stratified cross-validation** for a more reliable estimate of generalisation performance than a single holdout split.

### Confusion matrix

The confusion matrix summarises classification performance on the held-out test set by cross-tabulating predicted vs. actual outcomes:

```
                   Actual: Survived    Actual: Died
Predicted: Survived       254               8
Predicted: Died            27              14
```

From this table the key metrics are derived directly:

| Metric | Formula | Value | Interpretation |
|---|---|---|---|
| Accuracy | (TP + TN) / N | 88.45% | Proportion of all patients correctly classified |
| Sensitivity | TP / (TP + FN) | 34.15% | Proportion of actual deaths correctly identified |
| Specificity | TN / (TN + FP) | 96.95% | Proportion of actual survivors correctly identified |
| PPV | TP / (TP + FP) | 34.15% | Of patients predicted to die, proportion who actually died |
| NPV | TN / (TN + FN) | 96.95% | Of patients predicted to survive, proportion who actually survived |
| Cohen's κ | — | 0.387 | Agreement beyond chance — moderate |
| AUC-ROC | — | 0.82 | Overall discriminative ability across all thresholds |

The matrix reveals the core tension created by class imbalance. The model identifies survivors very well (specificity 96.95%) but catches only about 1 in 3 actual deaths at the default threshold. This is expected behaviour on an 87:13 imbalanced dataset — the model defaults to predicting survival because that is correct most of the time.

Critically, accuracy (88.45%) does not significantly exceed the No Information Rate of 86.5% (p = 0.18), meaning a naive classifier would perform comparably on raw accuracy alone. The AUC of 0.82 is the more meaningful metric — it evaluates discrimination across all thresholds rather than at a single cut-point, and is unaffected by the class distribution. Youden's index threshold optimisation meaningfully improves sensitivity for death at a modest cost to specificity.

---

## Part 2 — Survival Analysis

### Why survival analysis?

Logistic regression treats in-hospital death as a binary event — a patient who dies on day 1 and one who dies on day 3 are treated identically. This discards the temporal structure of the outcome and cannot answer questions about when patients are most at risk. Survival analysis addresses this by modelling **time to event**, with explicit handling of **right-censored** observations — patients discharged alive, for whom we know only that they survived to at least their last observed day.

The five predictors carried forward from Part 1 were selected based on their odds ratios and clinical interpretability: `age_group`, `obesity`, `asthma`, `chf_history`, and `right_vent_mi`. They span demographics, metabolic comorbidities, inflammatory history, cardiac history, and acute MI presentation — covering distinct clinical pathways to the same outcome.

### Kaplan-Meier estimation

The **Kaplan-Meier (KM) estimator** is a nonparametric method for estimating the survival function S(t) — the probability of surviving past time t — from censored data:

```
S(t) = ∏ (1 - dᵢ / nᵢ)  for all tᵢ ≤ t
```

where dᵢ is the number of deaths at time tᵢ and nᵢ is the number of patients still at risk just before tᵢ. KM curves are estimated separately for subgroups defined by each predictor. Group differences are tested using the **log-rank test** under the null hypothesis that survival functions are equal across groups:

```
H₀: S₁(t) = S₂(t) for all t
```

A p-value < 0.05 indicates a statistically significant difference in survival between groups. KM and the log-rank test are unadjusted — they show the marginal relationship between a single predictor and survival without controlling for confounders. The Cox model provides the adjusted estimates.

### Cox proportional hazards model

The **Cox proportional hazards model** extends the survival framework to multiple predictors simultaneously, modelling the **hazard function** — the instantaneous rate of dying at time t given survival to that point:

```
h(t | X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ... + βₚXₚ)
```

The baseline hazard h₀(t) is left completely unspecified — this is what makes Cox a **semiparametric** model. It estimates the βs without requiring any assumption about the shape of h₀(t). Exponentiating each coefficient gives a **hazard ratio (HR)**:

```
HR = exp(βⱼ)
```

A hazard ratio of 2 means a patient with that characteristic is dying at twice the instantaneous rate of the reference group at any given moment, holding all other predictors constant. HR > 1 increases risk; HR < 1 is protective.

The model fitted in this analysis is:

```
h(t | X) = h₀(t) × exp(
  β₁ · age_group
  + β₂ · obesity
  + β₃ · asthma
  + β₄ · chf_history
  + β₅ · right_vent_mi
)
```

The **proportional hazards assumption** — that hazard ratios are constant over time — is verified using Schoenfeld residuals via `cox.zph()`. A significant result (p < 0.05) for any predictor indicates a time-varying effect requiring further modelling.

### Model diagnostics

**ROC curve:** The Cox model's linear predictor is used as a continuous risk score to produce an ROC curve. This evaluates how well the model ranks patients by risk across all possible thresholds. The AUC from the Cox model can be compared directly to the logistic model AUC of 0.82 to assess whether accounting for time adds discriminative value.

**Calibration plot:** Predicted death probabilities (derived from the Cox linear predictor and baseline hazard) are binned into 10 groups and compared against observed death rates. Points lying on the 45-degree diagonal indicate a well-calibrated model. Systematic departure from the diagonal reveals whether the model over- or under-predicts risk at particular probability levels.

**Variable importance:** Predictors in the Cox model are ranked by the absolute magnitude of their log-hazard coefficients — how strongly each one moves the hazard in either direction, regardless of sign. This gives an immediate visual summary of which predictors carry the most weight in the survival model, complementing the hazard ratio forest plot.

---

## Key Findings

### Logistic regression — significant predictors of death

| Predictor | OR | p-value | Clinical interpretation |
|---|---|---|---|
| Bronchial asthma history | 6.96 | 0.002 | Shared inflammatory pathway (IL-6, NF-κB) accelerates atherosclerosis and worsens acute MI outcomes |
| Right ventricular MI | 5.00 | 0.002 | RV damage directly impairs forward flow to the pulmonary circulation, precipitating shock |
| Obesity history | 3.19 | 0.027 | Metabolic syndrome and insulin resistance compound endothelial dysfunction during acute ischaemia |
| Age group | 1.92 | 0.002 | Reduced cardiac reserve, greater comorbidity burden, and age-related decline in healing capacity |
| Chronic heart failure history | 1.38 | 0.019 | Pre-existing LV dysfunction reduces the heart's tolerance for the additional ischaemic insult |
| Abnormal heart rate at admission | 1.33 | 0.034 | Tachycardia or bradycardia at admission signals haemodynamic compromise |
| Premature ventricular contractions | 1.32 | 0.019 | PVCs are a precursor to sustained ventricular arrhythmias and sudden cardiac death |
| Chest pain history (angina) | 1.25 | 0.002 | Chronic angina burden reflects the extent of underlying coronary artery disease |
| Day-1 hospital complications | 1.23 | 0.018 | Pain relapse and opioid use in the first 24 hours signals an adverse early trajectory |
| ICU medication intensity | 0.63 | 0.001 | Protective — anticoagulants (heparin) and aspirin in the ICU significantly reduce mortality |

The protective direction of ICU treatment is among the most interpretable findings. Patients receiving anticoagulants and antiplatelet therapy had 37% lower odds of dying, directly validating the clinical evidence base for early pharmacological intervention post-MI.

### Survival analysis

**Kaplan-Meier results:** Survival curves differed significantly by age group (log-rank p < 0.05), with the 65-and-over group showing markedly steeper early decline. Obesity and right ventricular MI also produced visually and statistically distinct survival curves, consistent with their large odds ratios in the logistic model.

**Cox model results:** After adjusting for all five predictors simultaneously, right ventricular MI and older age group retained the largest hazard ratios, consistent with the logistic findings. The proportional hazards assumption held for most predictors, though predictors related to acute presentation showed marginal evidence of time-varying effects — their impact is largest in the first 24 hours and attenuates in surviving patients.

**Concentration of early mortality:** The vast majority of in-hospital deaths in this dataset occur within the first 24 hours. This is a meaningful epidemiological finding — it implies that the window for intervention is narrow and that admission-time characteristics carry the most predictive weight.

---

## Limitations

**Class imbalance.** Despite class weighting, the 86.5%/13.5% split limits the model's ability to detect deaths. The Cohen's κ of 0.387 and low unadjusted sensitivity reflect this constraint. A dataset with greater event frequency, or a prospective design with active follow-up, would produce more reliable minority-class predictions.

**Complete-case analysis.** The ~7.6% missingness is handled by listwise deletion — rows with any missing predictor are dropped from model fitting. This is valid under the Missing Completely At Random (MCAR) assumption, which is unlikely to hold in clinical data where sicker patients are more likely to have incomplete records. Multiple imputation via the `mice` package would be a more robust approach, particularly for variables with very high missingness such as `IBS_NASL` (95.76% missing) and `S_AD_KBRIG` (63.29% missing).

**Stepwise selection bias.** Backward stepwise AIC is known to produce optimistically biased coefficient estimates and inflated significance in the selected model, because the same data is used for both variable selection and inference. A more rigorous approach would use penalised regression (LASSO via `glmnet`) or pre-specify predictors based on clinical theory before model fitting.

**Coarse time variable.** The dataset does not record exact admission and death timestamps. Hospital day (1, 2, or 3) serves as the survival time proxy throughout Part 2. This three-point resolution limits the granularity of the survival analysis and precludes parametric time-to-event modelling.

**Historical cohort.** Data collected in 1992–1995 predate primary PCI (coronary stenting), high-intensity statins, and modern dual antiplatelet protocols. Treatment effects identified here — particularly the protective ICU medication finding — may not generalise to contemporary clinical practice where these interventions are standard rather than variable.

**Single institution.** All patients were admitted to one hospital in Krasnoyarsk, Russia. Institutional factors such as staffing and treatment protocols may not be representative of other settings. External validation on an independent cohort would be required before drawing generalisable conclusions.

---

## Repository Structure

```
mi-fatality-prediction/
│
├── R/
│   ├── mi_fatality_analysis.R     # Part 1: logistic regression
│   └── mi_survival_simple.R       # Part 2: survival analysis
│
├── figures/
│   ├── confusion_matrix.png       # Confusion matrix heatmap
│   ├── odds_ratio_forest_plot.png # Logistic model odds ratios
│   ├── km_age_group.png           # KM curves by age group
│   ├── km_obesity.png             # KM curves by obesity
│   ├── km_right_vent_mi.png       # KM curves by right ventricular MI
│   ├── cox_forest_plot.png        # Cox model hazard ratios
│   ├── variable_importance.png    # Predictors ranked by log-hazard magnitude
│   ├── roc_curve.png              # ROC curve with AUC
│   └── calibration_plot.png       # Predicted vs observed event rates
│
├── outputs/
│   ├── logistic_coefficients.csv  # Reduced logistic model coefficients
│   ├── cox_results.csv            # Cox model hazard ratios
│   └── descriptive_table.csv      # Patient cohort summary by outcome
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## How to Run

**1. Install R packages**

```r
install.packages(c(
  "tidyverse", "caret", "pROC", "MASS",
  "ResourceSelection", "car", "broom", "survival"
))
```

**2. Run the scripts**

The dataset downloads automatically from UCI — no manual download needed.

```r
# Part 1: logistic regression (full model → reduced model → evaluation)
source("R/mi_fatality_analysis.R")

# Part 2: Kaplan-Meier curves, Cox model, and diagnostics
source("R/mi_survival_simple.R")
```

Figures are saved to `figures/` and result tables to `outputs/` automatically when each script runs.

---

## Dependencies

| Package | Purpose |
|---|---|
| `tidyverse` | Data wrangling and all ggplot2 visualisations |
| `caret` | Cross-validation and confusion matrix metrics |
| `pROC` | ROC curves, AUC, and risk score evaluation |
| `MASS` | Backward stepwise selection via `stepAIC()` |
| `ResourceSelection` | Hosmer-Lemeshow calibration test |
| `car` | Variance inflation factors for multicollinearity |
| `broom` | Tidy model output for plots and export tables |
| `survival` | `Surv()` objects, `survfit()`, `coxph()`, `cox.zph()`, `basehaz()` |

R ≥ 4.3.0 recommended.

---

## Citation

> Golovenkin, S., Shulman, V., Rossiev, D., Shesternya, P., Nikulina, S., Orlova, Y., & Voino-Yasenetsky, V. (2020). *Myocardial infarction complications* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53P5M

---

## License

Code: [MIT License](LICENSE) | Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
