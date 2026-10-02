# Credit Card Default Risk Engine: Machine Learning & XAI Analysis

[![R](https://img.shields.io/badge/R-4.x-276DC3.svg?logo=r&logoColor=white)](https://www.r-project.org/)
[![Tidymodels](https://img.shields.io/badge/Tidymodels-Machine_Learning-276DC3.svg)](https://www.tidymodels.org/)
[![Random_Forest](https://img.shields.io/badge/Random_Forest-Best_Model_AUC_0.760-green.svg)]()
[![renv](https://img.shields.io/badge/renv-Package_Management-blue.svg)](https://rstudio.github.io/renv/)

## Project Overview

This repository contains an end-to-end Machine Learning and Explainable AI (XAI) pipeline built in **R** to assess credit card default risk (`default payment next month`). 

Analyzing historical payment behaviors, credit limits, and billing metrics across 12,000 clients, this project evaluates multiple statistical and ensemble models (Random Forest, FDA, Elastic Net, Group Lasso, SVM). It focuses on optimizing predictive discrimination while uncovering complex non-linear feature interactions for financial risk decision-making.

---

## Financial & Business Context

In credit risk modeling, **imbalanced classification** (~22.7% default rate) presents a key challenge: the financial cost of a **False Negative** (approving a loan for a client who defaults) is vastly higher than that of a **False Positive** (rejecting a creditworthy client).

To address this asymmetrical cost structure:
* **Custom Decision Threshold:** All models are evaluated at an adjusted decision boundary of **0.30** (instead of the default 0.50) to maximize recall on high-risk dahlis without collapsing precision.
* **Evaluation Metrics:** Primary ranking relies on **AUC-ROC** for overall discrimination power and **F1-Score** at the 0.30 threshold.

---

## Machine Learning Pipeline & Methodology

1. **Data Ingestion & Cleaning (`data/raw/`)**
   * Preprocessing 12,000 credit records spanning 6 months (April – September 2005).
   * Categorical anomaly resolution: collapsed sparse/undocumented categories in `EDUCATION` (0, 5, 6 $\rightarrow$ 4: *Others*) and `MARRIAGE` (0 $\rightarrow$ 3: *Others*).

2. **Feature Transformation & Scaling**
   * Applied `signed log` transformation on highly right-skewed monetary variables (`BILL_AMT1`–`6` and `PAY_AMT1`–`6`).
   * Z-score standardization for numerical features to support distance-based and regularized algorithms (Elastic Net, SVM).

3. **Model Suite & Training (`R/modeling_and_analysis.R`)**
   * **Tree Ensembles:** Random Forest (`ranger`) with hyperparameter tuning (`num.trees = 300`, `mtry = 2`, `min.node.size = 20`).
   * **Flexible Non-Parametric:** Flexible Discriminant Analysis with MARS (`earth`, `discrim`).
   * **Regularized Generalized Linear Models:** Elastic Net (`glmnet`) and Group Lasso (`grpreg`).
   * **Kernel Methods:** Support Vector Machines with Radial Basis Kernel (`kernlab`).

4. **Explainable AI & Diagnostics (XAI)**
   * **Variable Importance (VIP):** Quantified predictor contribution via Gini impurity reduction.
   * **Partial Dependence Plots (PDP) & Partial Interaction Maps:** Analyzed decision boundaries and Friedman's $H$-statistic for non-linear interactions.

---

## Model Performance & Benchmark

All models were evaluated on an independent test dataset using a **0.30 threshold**:

| Rank | Model Architecture | AUC-ROC | F1-Score (@ 0.30 threshold) | Key Characteristic |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Random Forest** | **0.760** | **0.525** | Top performer; captures complex interactions |
| **2** | **FDA (MARS)** | 0.756 | 0.513 | Strong non-linear splines modeling |
| **3** | **Elastic Net** | 0.731 | 0.502 | L1/L2 regularized baseline |
| **4** | **Group Lasso** | 0.730 | 0.500 | Grouped penalty across categorical dummies |
| **5** | **SVM (Radial Kernel)** | 0.708 | 0.481 | Non-linear boundary mapping |

---

## Key XAI & Analytical Findings

* **Dominance of Recent Payment History:** The most recent payment status (`PAY_0`, September status) is by far the single most influential feature (VIP score **213.5**), followed by `PAY_2` (100.1), `PAY_AMT1` (93.2), and `BILL_AMT1` (93.1).
* **Demographic Irrelevance:** Static demographic variables (`AGE`, `SEX`, `EDUCATION`, `MARRIAGE`) had negligible feature importance compared to dynamic financial behaviors.
* **Critical Risk Boundary:** Partial dependence analysis identified `PAY_0 >= 2` (a 2-month payment delay) as a severe tipping point for default probability.
* **Non-Linear Feature Interactions:** Friedman's $H$-statistic confirmed strong interactions between payment delays and billing/payment amounts:
  * `PAY_0` & `PAY_2` ($H = 0.492$) — repeated delinquency compounding risk.
  * `PAY_0` & `PAY_AMT1` ($H = 0.404$) — risk mitigation via recent partial payments.
  * `PAY_0` & `BILL_AMT1` ($H = 0.368$) — debt load impact on delayed payments.

---

## Repository Structure

```text
.
├── .gitignore                # Git exclusion rules (data, models, renv libs)
├── .Rprofile                 # Automatic renv environment bootstrapping
├── renv.lock                 # Reproducible environment dependency lockfile
├── README.md                 # Project documentation
│
├── data/
│   └── dane                  # Raw dataset
│           
├── R/
│   └── modeling_and_analysis.R # Core ML training, evaluation & XAI script
│
├── models/
│   └── final_model.rds       # Serialized Random Forest model object
│
├── reports/
│  ├── analysis_report.Rmd   # Source RMarkdown analysis document
│  └── final_report.pdf      # Compiled final PDF report
│
└── renv/                    # Isolated virtual environment configuration
```

## Key Technical Skills Demonstrated

* **Data Preprocessing & Feature Engineering:** Handling highly right-skewed financial features via `signed log` transformations, Z-score standardization, resolving categorical anomalies, and handling severe class imbalance (~22.7% default rate).
* **Machine Learning & Statistical Modeling:** Comparative benchmark of 5 distinct algorithm families (Random Forest, Elastic Net, Group Lasso, FDA/MARS, SVM), custom decision threshold optimization (0.30 cutoff) for asymmetric financial risk costs, and evaluation using AUC-ROC and F1-Score.
* **Explainable AI (XAI) & Interpretability:** Advanced model diagnostics using Gini-based Variable Importance (VIP), Partial Dependence Plots (PDP), decision boundary visualizations, and non-linear feature interaction detection via Friedman's $H$-statistic.
* **Reproducibility & Software Engineering:** Project environment isolation using `renv` (lockfiles & dependency management), modular R code structure (`tidymodels`), and automated PDF report compilation via RMarkdown/`knitr`.

## Installation & Execution

```bash
git clone https://github.com/piotrjsk/credit-card-default-analysis.git
cd credit-card-default-analysis

# Restore reproducible package dependencies
Rscript -e "renv::restore()"

# Render the executive report
Rscript -e "rmarkdown::render('reports/analysis_report.Rmd', output_dir = 'reports')"
```

---
*Developed by Piotr Jasiak & Mateusz Panek | [LinkedIn Profile](https://www.linkedin.com/in/piotrjasiak)*


