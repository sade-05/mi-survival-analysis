# =============================================================================
# Myocardial Infarction: Survival Analysis
# Dataset: UCI Myocardial Infarction Complications
# https://doi.org/10.24432/C53P5M
# =============================================================================


# -----------------------------------------------------------------------------
# 0. PACKAGES
# -----------------------------------------------------------------------------

# install.packages(c("tidyverse", "survival", "broom", "pROC"))

library(tidyverse)
library(survival)
library(broom)


# -----------------------------------------------------------------------------
# 1. DATA INGESTION
# -----------------------------------------------------------------------------
# Downloads directly from UCI 

zip_url  <- "https://archive.ics.uci.edu/static/public/579/myocardial+infarction+complications.zip"
zip_path <- tempfile(fileext = ".zip")
data_dir <- tempdir()

download.file(zip_url, destfile = zip_path, mode = "wb", quiet = TRUE)
unzip(zip_path, exdir = data_dir)

col_names <- c(
  "ID", "AGE", "SEX", "INF_ANAM", "STENOK_AN", "FK_STENOK", "IBS_POST",
  "IBS_NASL", "GB", "SIM_GIPERT", "DLIT_AG", "ZSN_A", "nr11", "nr01",
  "nr02", "nr03", "nr04", "nr07", "nr08", "np01", "np04", "np05", "np07",
  "np08", "np09", "np10", "endocr_01", "endocr_02", "endocr_03",
  "zab_leg_01", "zab_leg_02", "zab_leg_03", "zab_leg_04", "zab_leg_06",
  "S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT", "O_L_POST",
  "K_SH_POST", "MP_TP_POST", "SVT_POST", "GT_POST", "FIB_G_POST",
  "ant_im", "lat_im", "inf_im", "post_im", "IM_PG_P",
  "ritm_ecg_p_01", "ritm_ecg_p_02", "ritm_ecg_p_04", "ritm_ecg_p_06",
  "ritm_ecg_p_07", "ritm_ecg_p_08",
  "n_r_ecg_p_01", "n_r_ecg_p_02", "n_r_ecg_p_03", "n_r_ecg_p_04",
  "n_r_ecg_p_05", "n_r_ecg_p_06", "n_r_ecg_p_08", "n_r_ecg_p_09",
  "n_r_ecg_p_10", "n_p_ecg_p_01", "n_p_ecg_p_03", "n_p_ecg_p_04",
  "n_p_ecg_p_05", "n_p_ecg_p_06", "n_p_ecg_p_07", "n_p_ecg_p_08",
  "n_p_ecg_p_09", "n_p_ecg_p_10", "n_p_ecg_p_11", "n_p_ecg_p_12",
  "fibr_ter_01", "fibr_ter_02", "fibr_ter_03", "fibr_ter_05",
  "fibr_ter_06", "fibr_ter_07", "fibr_ter_08",
  "GIPO_K", "K_BLOOD", "GIPER_Na", "Na_BLOOD",
  "ALT_BLOOD", "AST_BLOOD", "KFK_BLOOD", "L_BLOOD", "ROE",
  "TIME_B_S", "R_AB_1_n", "R_AB_2_n", "R_AB_3_n",
  "NA_KB", "NOT_NA_KB", "LID_KB", "NITR_S",
  "NA_R_1_n", "NA_R_2_n", "NA_R_3_n",
  "NOT_NA_1_n", "NOT_NA_2_n", "NOT_NA_3_n",
  "LID_S_n", "B_BLOK_S_n", "ANT_CA_S_n", "GEPAR_S_n", "ASP_S_n",
  "TIKL_S_n", "TRENT_S_n",
  "FIBR_PREDS", "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", "A_V_BLOK",
  "OTEK_LANC", "RAZRIV", "DRESSLER", "ZSN", "REC_IM", "P_IM_STEN",
  "LET_IS"
)

raw_data <- read.csv(
  file.path(data_dir, "MI.data"),
  header           = FALSE,
  na.strings       = "?",
  col.names        = col_names,
  stringsAsFactors = FALSE
)

cat("Rows:", nrow(raw_data), "| Columns:", ncol(raw_data), "\n")


# -----------------------------------------------------------------------------
# 2. OUTCOME & PREDICTOR ENGINEERING
# -----------------------------------------------------------------------------
# LET_IS codes: 0 = survived, 1-7 = various causes of death.
# Collapse to a binary event (0 = censored/survived, 1 = died).
#
# Time proxy: the dataset records patient status at hospital days 1, 2, and 3.
# time_days = the last day a patient was documented. Survivors are
# right-censored at day 3 — we know they were alive at last follow-up
# but do not observe them beyond the hospital window.

df <- raw_data %>%
  mutate(

    # --- Survival outcome ---
    time_days = case_when(
      !is.na(NA_R_3_n) | !is.na(NOT_NA_3_n) ~ 3,
      !is.na(NA_R_2_n) | !is.na(NOT_NA_2_n) ~ 2,
      TRUE                                    ~ 1
    ),
    died = as.integer(LET_IS > 0),

    # --- Predictors ---
    age_group = case_when(
      AGE < 45             ~ "Under 45",
      AGE >= 45 & AGE < 65 ~ "45 to 64",
      AGE >= 65            ~ "65 and over"
    ),
    age_group     = factor(age_group,
                           levels = c("Under 45", "45 to 64", "65 and over")),
    obesity       = factor(endocr_02, levels = c(0, 1),
                           labels = c("No obesity", "Obesity")),
    asthma        = factor(zab_leg_03, levels = c(0, 1),
                           labels = c("No asthma", "Asthma")),
    chf_history   = factor(ZSN_A > 0, levels = c(FALSE, TRUE),
                           labels = c("No CHF", "CHF history")),
    right_vent_mi = factor(IM_PG_P, levels = c(0, 1),
                           labels = c("No", "Yes"))
  )

cat("\nEvent distribution:\n")
cat("Survived:", sum(df$died == 0), "\n")
cat("Died:    ", sum(df$died == 1), "\n")

# Output directories
dir.create("figures", showWarnings = FALSE)
dir.create("outputs", showWarnings = FALSE)


# =============================================================================
# PART A: KAPLAN-MEIER SURVIVAL CURVES
# =============================================================================
# The Kaplan-Meier estimator calculates the probability of surviving past
# each observed event time, accounting for patients who are censored
# (discharged alive) before the end of follow-up.
#
# The survival probability at time t is:
#   S(t) = product of (1 - deaths / at-risk) for all event times up to t
#
# We plot KM curves for three key risk factors, each with a log-rank test.
# The log-rank test asks: are the survival curves statistically different
# between groups? A p-value < 0.05 suggests a meaningful difference.
# =============================================================================


# This is a helper function to avoid repeating the same plotting code three times.
# It takes a fitted survfit object and group label, and returns a ggplot.

plot_km <- function(km_fit, group_label, palette) {

  km_tidy <- tidy(km_fit) %>%
    mutate(group = str_extract(strata, "(?<=\\=).*"))

  ggplot(km_tidy, aes(x = time, y = estimate, colour = group)) +
    geom_step(linewidth = 0.9) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high, fill = group),
                alpha = 0.1, colour = NA) +
    scale_colour_manual(values = palette) +
    scale_fill_manual(values   = palette) +
    scale_x_continuous(breaks  = 1:3) +
    scale_y_continuous(limits  = c(0, 1),
                       labels  = scales::percent_format()) +
    labs(
      x      = "Hospital day",
      y      = "Survival probability",
      colour = group_label,
      fill   = group_label
    ) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "bottom")
}


# -----------------------------------------------------------------------------
# A1. KM CURVES BY AGE GROUP
# -----------------------------------------------------------------------------

km_age <- survfit(Surv(time_days, died) ~ age_group, data = df)

lr_age  <- survdiff(Surv(time_days, died) ~ age_group, data = df)
p_age   <- 1 - pchisq(lr_age$chisq, df = length(lr_age$n) - 1)
cat("\nLog-rank test — age group: p =", round(p_age, 4), "\n")

plot_km(km_age, "Age group",
        palette = c("steelblue", "goldenrod", "tomato")) +
  labs(title    = "Survival by age group",
       subtitle = paste0("Log-rank p = ", round(p_age, 4)))

ggsave("figures/km_age_group.png", width = 7, height = 5, dpi = 150)


# -----------------------------------------------------------------------------
# A2. KM CURVES BY OBESITY
# -----------------------------------------------------------------------------

km_obesity <- survfit(Surv(time_days, died) ~ obesity,
                      data = df %>% filter(!is.na(obesity)))

lr_obesity  <- survdiff(Surv(time_days, died) ~ obesity,
                        data = df %>% filter(!is.na(obesity)))
p_obesity   <- 1 - pchisq(lr_obesity$chisq, df = 1)
cat("Log-rank test — obesity: p =", round(p_obesity, 4), "\n")

plot_km(km_obesity, "Obesity history",
        palette = c("steelblue", "tomato")) +
  labs(title    = "Survival by obesity history",
       subtitle = paste0("Log-rank p = ", round(p_obesity, 4)))

ggsave("figures/km_obesity.png", width = 7, height = 5, dpi = 150)


# -----------------------------------------------------------------------------
# A3. KM CURVES BY RIGHT VENTRICULAR MI
# -----------------------------------------------------------------------------

km_rvmi <- survfit(Surv(time_days, died) ~ right_vent_mi,
                   data = df %>% filter(!is.na(right_vent_mi)))

lr_rvmi  <- survdiff(Surv(time_days, died) ~ right_vent_mi,
                     data = df %>% filter(!is.na(right_vent_mi)))
p_rvmi   <- 1 - pchisq(lr_rvmi$chisq, df = 1)
cat("Log-rank test — right ventricular MI: p =", round(p_rvmi, 4), "\n")

plot_km(km_rvmi, "Right ventricular MI",
        palette = c("steelblue", "tomato")) +
  labs(title    = "Survival by right ventricular MI",
       subtitle = paste0("Log-rank p = ", round(p_rvmi, 4)))

ggsave("figures/km_right_vent_mi.png", width = 7, height = 5, dpi = 150)


# =============================================================================
# PART B: COX PROPORTIONAL HAZARDS MODEL
# =============================================================================
# The KM curves above show unadjusted survival differences between groups.
# The Cox model adjusts for all predictors simultaneously, isolating the
# independent effect of each one.
#
# The model estimates the hazard function:
#   h(t | X) = h0(t) x exp(b1X1 + b2X2 + ... + bpXp)
#
# h0(t) is the baseline hazard — left unspecified (semiparametric).
# exp(b) gives the hazard ratio (HR) for each predictor:
#   HR > 1 = higher instantaneous risk of death
#   HR < 1 = lower instantaneous risk of death
#   HR = 1 = no effect
#
# The proportional hazards assumption states that HRs are constant over time. 
# We check using Schoenfeld residuals via cox.zph().
# =============================================================================


# -----------------------------------------------------------------------------
# B1. FIT THE COX MODEL
# -----------------------------------------------------------------------------

cox_model <- coxph(
  Surv(time_days, died) ~ age_group + obesity + asthma +
    chf_history + right_vent_mi,
  data = df
)

cat("\n--- Cox proportional hazards model ---\n")
summary(cox_model)


# -----------------------------------------------------------------------------
# B2. PROPORTIONAL HAZARDS ASSUMPTION CHECK
# -----------------------------------------------------------------------------
# A significant p-value (< 0.05) for a predictor suggests its effect
# is not constant over time and a violation of the Cox model assumption.

cat("\n--- Proportional hazards test (Schoenfeld residuals) ---\n")
tryCatch(
  print(cox.zph(cox_model)),
  error = function(e) {
    cat("PH test could not be computed:", conditionMessage(e), "\n")
  }
)


# -----------------------------------------------------------------------------
# B3. FOREST PLOT OF HAZARD RATIOS
# -----------------------------------------------------------------------------
# Hazard ratios with 95% confidence intervals for all predictors.
# The dashed vertical line at HR = 1 is the reference (no effect).
# Points to the right indicate increased risk; left indicates protection.

cox_tidy <- tidy(cox_model, exponentiate = TRUE, conf.int = TRUE)

ggplot(cox_tidy, aes(x = estimate, y = reorder(term, estimate))) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey60") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high),
                 height = 0.25, colour = "grey40") +
  geom_point(aes(colour = estimate > 1), size = 3) +
  scale_colour_manual(values = c("steelblue", "tomato"),
                      labels = c("Protective (HR < 1)", "Risk-increasing (HR > 1)"),
                      name   = NULL) +
  scale_x_log10() +
  labs(
    title    = "Cox model: hazard ratios with 95% confidence intervals",
    subtitle = "HR > 1 indicates higher risk of in-hospital death",
    x        = "Hazard ratio (log scale)",
    y        = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

ggsave("figures/cox_forest_plot.png", width = 8, height = 5, dpi = 150)


# -----------------------------------------------------------------------------
# B4. SAVE RESULTS TABLE
# -----------------------------------------------------------------------------

cox_tidy %>%
  dplyr::select(term, estimate, conf.low, conf.high, p.value) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3))) %>%
  write.csv("outputs/cox_results.csv", row.names = FALSE)

cat("\nAnalysis complete. Figures saved to figures/ and table to outputs/\n")


# =============================================================================
# ADDITIONAL OUTPUTS
# =============================================================================


# -----------------------------------------------------------------------------
# 1. DESCRIPTIVE STATISTICS TABLE
# -----------------------------------------------------------------------------
# Summarises the patient cohort split by outcome: survived vs. died.
# Gives any reader immediate context about who the patients are before
# any modelling results are presented.

desc_table <- df %>%
  mutate(outcome = if_else(died == 1, "Died", "Survived")) %>%
  group_by(outcome) %>%
  summarise(
    N                   = n(),
    `Age mean (SD)`     = paste0(round(mean(AGE, na.rm = TRUE), 1),
                                 " (", round(sd(AGE, na.rm = TRUE), 1), ")"),
    `Male (%)`          = paste0(round(100 * mean(SEX == 1, na.rm = TRUE), 1), "%"),
    `Obesity (%)`       = paste0(round(100 * mean(endocr_02 == 1, na.rm = TRUE), 1), "%"),
    `Asthma (%)`        = paste0(round(100 * mean(zab_leg_03 == 1, na.rm = TRUE), 1), "%"),
    `Diabetes (%)`      = paste0(round(100 * mean(endocr_01 == 1, na.rm = TRUE), 1), "%"),
    `CHF history (%)`   = paste0(round(100 * mean(ZSN_A > 0, na.rm = TRUE), 1), "%"),
    `Right vent MI (%)` = paste0(round(100 * mean(IM_PG_P == 1, na.rm = TRUE), 1), "%"),
    .groups = "drop"
  )

write.csv(desc_table, "outputs/descriptive_table.csv", row.names = FALSE)

cat("\nDescriptive statistics by outcome:\n")
print(desc_table)


# -----------------------------------------------------------------------------
# 2. VARIABLE IMPORTANCE PLOT
# -----------------------------------------------------------------------------
# Ranks the Cox model predictors by their absolute log-hazard coefficient.
# Absolute value removes the sign so predictors are ranked purely by
# magnitude — how strongly they move the hazard in either direction.

cox_raw <- tidy(cox_model, exponentiate = FALSE, conf.int = TRUE) %>%
  mutate(
    direction = if_else(estimate > 0, "Risk-increasing", "Protective"),
    abs_coef  = abs(estimate),
    term      = str_replace_all(term, "_", " ")
  )

ggplot(cox_raw, aes(x = abs_coef, y = reorder(term, abs_coef),
                    fill = direction)) +
  geom_col(width = 0.65, alpha = 0.85) +
  scale_fill_manual(values = c("Protective"      = "steelblue",
                                "Risk-increasing" = "tomato")) +
  labs(
    title    = "Variable importance — Cox proportional hazards model",
    subtitle = "Ranked by absolute log-hazard coefficient",
    x        = "Absolute log-hazard coefficient",
    y        = NULL,
    fill     = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

ggsave("figures/variable_importance.png", width = 8, height = 5, dpi = 150)


# -----------------------------------------------------------------------------
# 3. ROC CURVE
# -----------------------------------------------------------------------------
# Uses the Cox model's linear predictor as a continuous risk score.
# The ROC curve plots sensitivity vs. 1-specificity across all thresholds,
# and the AUC summarises overall discriminative ability.

library(pROC)

# Extract the complete-case rows the Cox model was fitted on,
# then align the linear predictor with the actual outcome.
cox_data   <- model.frame(cox_model)
risk_score <- predict(cox_model, type = "lp")
actual     <- as.integer(cox_data[["Surv(time_days, died)"]][, "status"])

roc_obj <- roc(actual, risk_score, quiet = TRUE)

roc_df <- data.frame(
  fpr = 1 - roc_obj$specificities,
  tpr = roc_obj$sensitivities
)

ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "grey50") +
  geom_line(colour = "steelblue", linewidth = 0.9) +
  annotate("text", x = 0.65, y = 0.15,
           label = paste0("AUC = ", round(auc(roc_obj), 3)),
           size = 4.5, colour = "steelblue") +
  coord_fixed() +
  labs(
    title    = "ROC curve — Cox proportional hazards model",
    subtitle = "Linear predictor used as continuous risk score",
    x        = "False positive rate (1 - Specificity)",
    y        = "True positive rate (Sensitivity)"
  ) +
  theme_minimal(base_size = 13)

ggsave("figures/roc_curve.png", width = 6, height = 6, dpi = 150)


# -----------------------------------------------------------------------------
# 4. CALIBRATION PLOT
# -----------------------------------------------------------------------------
# Compares predicted risk against observed death rates across ten equal bins.
# Points on the 45-degree diagonal indicate perfect calibration.

base_surv  <- basehaz(cox_model, centered = FALSE)
median_haz <- base_surv$hazard[which.min(abs(base_surv$time -
                                              median(df$time_days)))]
pred_prob  <- 1 - exp(-exp(risk_score) * median_haz)

cal_df <- data.frame(
  pred_prob = pred_prob,
  actual    = actual
) %>%
  filter(!is.na(pred_prob)) %>%
  mutate(bin = cut(pred_prob, breaks = 10, include.lowest = TRUE)) %>%
  group_by(bin) %>%
  summarise(
    mean_pred = mean(pred_prob),
    obs_rate  = mean(actual),
    n         = n(),
    .groups   = "drop"
  )

ggplot(cal_df, aes(x = mean_pred, y = obs_rate)) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "grey50") +
  geom_point(aes(size = n), colour = "steelblue", alpha = 0.8) +
  geom_line(colour = "steelblue") +
  scale_size_continuous(name = "Patients in bin") +
  coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title    = "Calibration plot — Cox proportional hazards model",
    subtitle = "Points on the dashed line indicate perfect calibration",
    x        = "Mean predicted probability of death",
    y        = "Observed proportion who died"
  ) +
  theme_minimal(base_size = 13)

ggsave("figures/calibration_plot.png", width = 6, height = 6, dpi = 150)


