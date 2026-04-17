# Project 1 - Q2(2) Report

## 0. Task and Goal

- Task: Q2(2) Survival Analysis report.
- Requirement: Explain the end-to-end survival analysis process and record all results produced in `Q2.ipynb`.
- Analysis notebook: `/data/Project1/Q2.ipynb`
- Runtime context: PySpark + Python lifelines workflow.

---

## 1. Data Preparation Process

### 1.1 Data source

- Source CSV: `ibm-telco-churn/Telco-Customer-Churn.csv`
- Dataset domain: IBM Telco customer churn.

### 1.2 Bronze table construction

- A fixed schema was defined and CSV was loaded into Spark DataFrame `bronze_df`.
- Columns include customer profile, service subscriptions, billing/payment, tenure, and churn label.

Key load output observations:
- First 5 rows were displayed successfully.
- Spark schema was printed successfully.
- Spark emitted a header warning because file header uses `Churn` while schema uses `churnString`.
- Despite header case mismatch warning, the pipeline proceeded and produced downstream tables/models.

### 1.3 Silver table transformation

From `bronze_df`, the notebook created `silver_df` by:
1. Mapping churn label:
- `churnString == 'Yes' -> churn = 1`
- `churnString == 'No' -> churn = 0`
2. Dropping `churnString`.
3. Filtering to month-to-month contracts only:
- `contract == 'Month-to-month'`
4. Filtering to customers with internet service:
- `internetService != 'No'`

This creates the analysis cohort for survival models.

---

## 2. Kaplan-Meier Survival Analysis

### 2.1 Population-level KM curve

- Model: `KaplanMeierFitter()`
- Duration: `tenure`
- Event indicator: `churn` (float)

Population-level median survival time:
- `kmf.median_survival_time_ = 34.0`

Interpretation:
- The estimated median time-to-churn for the filtered cohort is 34 months.

### 2.2 Covariate-level KM + log-rank tests

The notebook defined helper functions to:
- Plot subgroup KM curves (`plot_km(col)`)
- Perform pairwise log-rank tests (`print_logrank(col)`)

#### A) Gender

Log-rank result:
- Test statistic: `2.038938`
- p-value: `0.153317`
- `-log2(p)`: `2.705414`

Interpretation:
- No statistically significant survival difference by gender at common thresholds (for this cohort).

#### B) Online Security

Log-rank result (`No` vs `Yes`):
- Test statistic: `141.60316`
- p-value: `1.187554e-32`
- `-log2(p)`: `106.053706`

Interpretation:
- Very strong survival difference by online security status.

#### C) Payment Method (pairwise)

| Comparison | Test Statistic | p-value | -log2(p) | Significant? |
|---|---:|---:|---:|---|
| Bank transfer (automatic) vs Credit card (automatic) | 0.061543 | 8.040732e-01 | 0.314601 | No |
| Bank transfer (automatic) vs Electronic check | 91.191889 | 1.303937e-21 | 69.377616 | Yes |
| Bank transfer (automatic) vs Mailed check | 43.536998 | 4.160192e-11 | 34.484559 | Yes |
| Credit card (automatic) vs Electronic check | 79.991082 | 3.761035e-19 | 61.205504 | Yes |
| Credit card (automatic) vs Mailed check | 39.684613 | 2.984678e-10 | 31.641706 | Yes |
| Electronic check vs Mailed check | 0.898320 | 3.432326e-01 | 1.542741 | No |

Interpretation:
- Electronic-check customers differ strongly from auto-payment groups.
- Auto-payment methods are not significantly different from each other.
- Electronic check and mailed check are not significantly different in this result table.

### 2.3 Survival probabilities example (DSL customers)

`pd.DataFrame(sp_internet_dsl.survival_function_at_times(range(0,10)))` produced:

| Time (month) | Survival Probability (DSL) |
|---:|---:|
| 0 | 1.000000 |
| 1 | 0.902698 |
| 2 | 0.864380 |
| 3 | 0.834702 |
| 4 | 0.810522 |
| 5 | 0.794352 |
| 6 | 0.783900 |
| 7 | 0.776362 |
| 8 | 0.768486 |
| 9 | 0.750833 |

---

## 3. Cox Proportional Hazards Model

### 3.1 Modeling setup

- One-hot encoding used for selected categorical variables.
- Final Cox input features:
- `dependents_Yes`
- `internetService_DSL`
- `onlineBackup_Yes`
- `techSupport_Yes`

Model fit:
- `CoxPHFitter(alpha=0.05)`
- `cph.fit(survival_pd, 'tenure', 'churn')`

### 3.2 Cox summary results

Dataset/model scale:
- Number of observations: `3351`
- Number of events observed: `1556`
- Right-censored: `1795`
- Partial log-likelihood: `-11315.95`

Coefficients:

| Covariate | coef | exp(coef) | p-value | Interpretation |
|---|---:|---:|---|---|
| dependents_Yes | -0.33 | 0.72 | <0.005 | Lower hazard (reduced churn risk) |
| internetService_DSL | -0.22 | 0.80 | <0.005 | Lower hazard |
| onlineBackup_Yes | -0.78 | 0.46 | <0.005 | Strongly lower hazard |
| techSupport_Yes | -0.64 | 0.53 | <0.005 | Lower hazard |

Overall fit metrics:
- Concordance: `0.64`
- Partial AIC: `22639.90`
- Log-likelihood ratio test: `337.77` on `4` df
- `-log2(p)` of LL-ratio test: `236.24`

Interpretation:
- All included covariates are statistically significant and associated with lower churn hazard.

### 3.3 Proportional hazards (PH) assumption check

`check_assumptions` output flagged violations for:
1. `internetService_DSL` (p-value `<5e-05`)
2. `onlineBackup_Yes` (p-value `<5e-05`)
3. `techSupport_Yes` (p-value `0.0002`)

lifelines advice from output:
- Consider stratification for these binary covariates (for example, `strata=[...]`).

Interpretation:
- The baseline Cox model gives useful associations, but strict PH assumption is not fully satisfied for multiple predictors.

---

## 4. Accelerated Failure Time (AFT) Model

### 4.1 Modeling setup

- Model used: `LogLogisticAFTFitter()`
- Duration: `tenure`
- Event: `churn`
- Features include partner, multiple lines, internet service, online security/backup, device protection, tech support, and payment method indicators.

### 4.2 AFT outputs

Printed median survival value:
- `Median Survival Time: 135.51`

Model summary scale:
- Number of observations: `3351`
- Number of events observed: `1556`
- Log-likelihood: `-6838.36`

Selected `alpha_` effects (all significant with p < 0.005):
- `deviceProtection_Yes`: exp(coef) `1.62`
- `internetService_DSL`: exp(coef) `1.47`
- `multipleLines_Yes`: exp(coef) `1.94`
- `onlineBackup_Yes`: exp(coef) `2.25`
- `onlineSecurity_Yes`: exp(coef) `2.37`
- `partner_Yes`: exp(coef) `1.97`
- `paymentMethod_Bank transfer (automatic)`: exp(coef) `2.10`
- `paymentMethod_Credit card (automatic)`: exp(coef) `2.22`
- `techSupport_Yes`: exp(coef) `1.99`

AFT fit metrics:
- Concordance: `0.73`
- AIC: `13698.72`
- Log-likelihood ratio test: `877.49` on `9` df
- `-log2(p)` of LL-ratio test: `605.78`

Interpretation:
- The AFT model shows stronger predictive ranking (higher concordance than Cox in this run) and strong, significant time-extending effects for multiple service/protection variables.

---

## 5. Customer Lifetime Value (CLV) Section Status in Notebook

In the CLV section, three result cells failed with the same runtime error:
- `get_payback_df()[0:25]` -> `TypeError: unhashable type: 'list'`
- `sns.barplot(...)` cell -> `TypeError: unhashable type: 'list'`
- `sns.lineplot(...)` cell -> `TypeError: unhashable type: 'list'`

Therefore:
- The CLV tabular and plotting outputs were not successfully produced in the current notebook execution state.

---

## 6. End-to-End Process Summary

This case follows a standard industrial survival-analysis pipeline:
1. Build a clean event-time cohort from raw subscription data.
2. Use Kaplan-Meier to estimate population survival and compare groups.
3. Use log-rank tests to quantify subgroup survival differences.
4. Fit Cox PH model for hazard-ratio interpretation.
5. Check PH assumptions and identify violated covariates.
6. Fit AFT model for time-scale interpretation and compare model quality.

Main findings from this notebook run:
- Cohort-level median survival is 34 months (KM).
- Online security and payment method are major stratifiers of survival behavior.
- Cox model indicates reduced churn hazard for dependents/DSL/backup/support indicators.
- PH assumption is violated for several binary predictors, suggesting model refinement (for example stratified Cox).
- Log-logistic AFT model gives strong, significant covariate effects and higher concordance (0.73).

---

## 7. Reproducibility Note

All values above are transcribed from stored outputs in `Q2.ipynb` cells, including:
- KM summary/statistical tests
- Cox summary and PH diagnostics
- AFT summary
- CLV section runtime errors

No additional external post-processing was used for these recorded results.

---

## 8. Q2(3) MySQL + Text-to-SQL Experiment Log

### 8.1 Current Completion Status

- MySQL data import has been completed. The import script is available at: `/data/Project1/convertcsv.sql`
- This section organizes Text-to-SQL cases based on the imported Telco dataset, without running any Python code.
- Selected LLM: OpenAI GPT (used to generate SQL drafts, followed by manual verification).

### 8.2 Database Schema (Used in All Cases)

```sql
CREATE TABLE Telco_Customer_Churn(
	 customerID       VARCHAR(10) NOT NULL PRIMARY KEY,
	 gender           VARCHAR(6) NOT NULL,
	 SeniorCitizen    BIT NOT NULL,
	 Partner          VARCHAR(3) NOT NULL,
	 Dependents       VARCHAR(3) NOT NULL,
	 tenure           INTEGER NOT NULL,
	 PhoneService     VARCHAR(3) NOT NULL,
	 MultipleLines    VARCHAR(16) NOT NULL,
	 InternetService  VARCHAR(11) NOT NULL,
	 OnlineSecurity   VARCHAR(19) NOT NULL,
	 OnlineBackup     VARCHAR(19) NOT NULL,
	 DeviceProtection VARCHAR(19) NOT NULL,
	 TechSupport      VARCHAR(19) NOT NULL,
	 StreamingTV      VARCHAR(19) NOT NULL,
	 StreamingMovies  VARCHAR(19) NOT NULL,
	 Contract         VARCHAR(14) NOT NULL,
	 PaperlessBilling VARCHAR(3) NOT NULL,
	 PaymentMethod    VARCHAR(25) NOT NULL,
	 MonthlyCharges   NUMERIC(6,2) NOT NULL,
	 TotalCharges     NUMERIC(7,2),
	 Churn            VARCHAR(3) NOT NULL
);
```

### 8.3 Three Failure Cases Where the LLM Generated Incorrect SQL

#### Case 1: Missing Key Filters Caused Incorrect Statistical Scope

- User question:
	Among customers with `Contract = 'Month-to-month'` and `InternetService <> 'No'`, compute churn rate (`Churn='Yes'`) by `PaymentMethod`, and return the top 3 methods in descending order of churn rate.

- LLM-generated incorrect SQL (issue: missing business filters):

```sql
SELECT
	PaymentMethod,
	AVG(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churn_rate
FROM Telco_Customer_Churn
GROUP BY PaymentMethod
ORDER BY churn_rate DESC
LIMIT 3;
```

- Correct SQL:

```sql
SELECT
	PaymentMethod,
	AVG(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churn_rate
FROM Telco_Customer_Churn
WHERE Contract = 'Month-to-month'
	AND InternetService <> 'No'
GROUP BY PaymentMethod
ORDER BY churn_rate DESC
LIMIT 3;
```


- Failure analysis:
	The LLM focused on grouping and churn-rate calculation but ignored required cohort filters in the prompt. This is a semantic-constraint omission error.

#### Case 2: Use of Unsupported or Incompatible Function (Median)

- User question:
	Compute the median `tenure` of churned users (`Churn='Yes'`) for each `Contract` type.

- LLM-generated incorrect SQL (issue: direct use of `MEDIAN`, which is typically unavailable in MySQL):

```sql
SELECT
	Contract,
	MEDIAN(tenure) AS median_tenure
FROM Telco_Customer_Churn
WHERE Churn = 'Yes'
GROUP BY Contract;
```

- Correct SQL (MySQL 8 window-function implementation):

```sql
WITH ranked AS (
	SELECT
		Contract,
		tenure,
		ROW_NUMBER() OVER (PARTITION BY Contract ORDER BY tenure) AS rn,
		COUNT(*) OVER (PARTITION BY Contract) AS cnt
	FROM Telco_Customer_Churn
	WHERE Churn = 'Yes'
)
SELECT
	Contract,
	AVG(tenure) AS median_tenure
FROM ranked
WHERE rn IN (FLOOR((cnt + 1) / 2), FLOOR((cnt + 2) / 2))
GROUP BY Contract;
```


- Failure analysis:
	The LLM tends to apply generic SQL syntax directly to MySQL and overlooks dialect differences such as function availability and window-function details.

#### Case 3: Incorrect Denominator Definition Caused Ratio Bias

- User question:
	Within the `Contract='Two year'` cohort, compute the paperless billing ratio (`PaperlessBilling='Yes'`) for each `PaymentMethod`.

- LLM-generated incorrect SQL (issue: denominator used total cohort size instead of per-group size):

```sql
SELECT
	PaymentMethod,
	SUM(CASE WHEN PaperlessBilling = 'Yes' THEN 1 ELSE 0 END) /
	(SELECT COUNT(*) FROM Telco_Customer_Churn WHERE Contract = 'Two year') AS paperless_rate
FROM Telco_Customer_Churn
WHERE Contract = 'Two year'
GROUP BY PaymentMethod;
```

- Correct SQL:

```sql
SELECT
	PaymentMethod,
	SUM(CASE WHEN PaperlessBilling = 'Yes' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS paperless_rate
FROM Telco_Customer_Churn
WHERE Contract = 'Two year'
GROUP BY PaymentMethod;
```


- Failure analysis:
	For grouped-ratio tasks, the LLM can confuse global and group-level denominators. The SQL may run, but the business meaning is wrong.

### 8.4 Assumptions About When LLMs May Fail in SQL Data Analysis

1. When prompts contain many business constraints, required filters (time window, cohort, scope) are easy to miss.
2. During cross-dialect adaptation, compatibility judgments for functions and syntax are often unstable.
3. For ratio-based metrics, denominator definitions are frequently incorrect.
4. When column names and semantics are similar (for example, `churn` vs `churnString`), the wrong field may be referenced.
5. If data-quality details (NULL values, empty strings, case sensitivity, enum formatting) are not explicit, SQL logic can drift from intended semantics.

### 8.5 Prompts Used in This Task

The following representative prompt templates were used in this section (reproducible):

1. Base generation prompt

```text
You are a MySQL 8 SQL expert. Generate SQL strictly based on the given schema and do not invent fields.
Output requirement: return JSON only, in the format {"sql": "..."}.
Schema: <paste the Telco_Customer_Churn CREATE TABLE statement>
Question: <natural-language question>
```

2. Constraint-enhanced prompt (to avoid missing conditions)

```text
Before generating SQL, first list the recognized filters, grouping dimensions, and metric definition (numerator/denominator),
then output the final SQL. The database is MySQL 8.
```

3. Dialect-correction prompt

```text
Check whether your SQL uses functions unsupported by MySQL 8. If so, rewrite it to a MySQL 8-executable version.
Output SQL only.
```

4. Result-validation prompt

```text
Self-check against business semantics:
1) Is the denominator consistent with the required scope?
2) Are any WHERE conditions missing?
3) Could NULL values affect correctness?
Then provide the corrected SQL.
```

### 8.6 Safety Notes

- No Python scripts were run in this section.
- This section only covers static SQL design and error-case analysis, with no destructive database operations (such as DROP/TRUNCATE/DELETE).
