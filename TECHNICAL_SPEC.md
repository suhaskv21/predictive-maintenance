# Technical Specification: Predictive Maintenance ML System

## 1. Objective

Design, implement, and deploy an end-to-end machine-learning application that helps industrial maintenance engineers assess the failure risk of a machine from a sensor reading and prioritize maintenance work.

The emphasis is on sound system design, reproducible engineering, operationally appropriate evaluation, and clear communication—not on maximizing a model score in isolation. The result must be a modular Python project and a live web application accessible at submission time.

## 2. Operational context

The organization operates its own production machines. Maintenance engineers need a practical way to use machine readings to inform maintenance prioritization. A false negative (a failure risk assessed as safe) and a false positive (a safe machine assessed as at risk) have different operational costs. The solution must discuss those trade-offs and explain how its output should be interpreted; it must not imply that a model prediction is a guarantee.

## 3. Dataset

- **Dataset:** AI4I 2020 Predictive Maintenance Dataset.
- **Available local file:** `data/ai4i2020.csv` (10,000 data rows in the provided copy).
- **Observed fields:** `UDI`, `Product ID`, `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Machine failure`, and failure-mode indicators `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
- `Type` identifies the product quality/type levels (`L`, `M`, `H`).

Use `Machine failure` as the default primary target unless a different target is explicitly justified. Treat the failure-mode indicators as outcome fields, not input features, when predicting `Machine failure`, to avoid target leakage. `UDI` and `Product ID` are identifiers; justify any use of them as predictors. Confirm the exact feature and target selection in the README. If the dataset is downloaded rather than using the supplied file, document the source and any required setup.

## 4. Scope and requirements

### 4.1 Problem framing and system design

The solution must:

1. State the prediction problem and intended users in one concise paragraph.
2. Explain how predictions reach maintenance engineers and what a useful outcome looks like.
3. Describe the operational consequences of false negatives and false positives.
4. Include a system design diagram covering data ingestion, model training, model serving, and monitoring.
5. Identify production components or controls that are not implemented in this challenge.
6. Explain the model-selection approach and trade-offs between interpretability and predictive performance.

### 4.2 Machine-learning pipeline

The solution must:

1. Provide a reproducible, modular preprocessing, training, and evaluation pipeline in a Python package; a single notebook is not an acceptable implementation.
2. Split data into training and held-out test sets before fitting preprocessing or model-selection steps. Fit preprocessing only on training data and apply the fitted transformations to validation/test data.
3. Train and evaluate a model for the selected target.
4. Explain the evaluation metric(s), accounting for class imbalance and the operational costs of errors. Report results on data not used to fit the evaluated model.
5. Serialize the fitted model and its preprocessing pipeline together, or serialize them in a way that reliably applies the same transformations at inference time.
6. Make the training/evaluation process repeatable; document relevant seeds, commands, and data expectations.

### 4.3 Web application and deployment

The solution must:

1. Provide a small web application through which a maintenance engineer can enter or select a machine reading and assess its failure risk.
2. Use the serialized training artifacts for inference; do not independently reimplement preprocessing in the UI.
3. Choose and clearly label an output, such as a failure probability, a risk flag, or a likely failure type. Explain the meaning and limitations of that output.
4. Deploy the application to a free-tier platform (Streamlit Community Cloud or an equivalent service).
5. Include a live, publicly accessible application URL in `README.md` and ensure it works at submission time.

### 4.4 Production-readiness critique

The README must describe:

1. How the deployed system would be monitored.
2. At least two production metrics, with a clear definition and purpose. Suitable examples include data drift, prediction-distribution shift, and post-outcome model performance.
3. A retraining strategy, including how a candidate model would be validated before replacing the live model.
4. Risks of full production deployment and practical mitigations.

## 5. Expected repository deliverables

The submitted GitHub repository must contain:

- Modular Python source code for preprocessing, training, evaluation, and inference (not a single notebook).
- A working web application and the configuration/code needed to deploy it.
- A system-design diagram in any readable format.
- `README.md` with setup instructions, the live application URL, key technical decisions, and the production-readiness critique.
- `requirements.txt` or `pyproject.toml`, sufficient for the project to run from a clean install.
- The serialized model and preprocessing artifacts, or documented reproducible instructions to generate them before running the app.

## 6. Acceptance criteria

The challenge is complete when all of the following are true:

- [ ] The repository installs from a clean environment using its documented dependency instructions.
- [ ] Training and evaluation can be run from documented commands without relying on notebook-only state.
- [ ] The chosen target, feature set, split strategy, and evaluation metrics are documented.
- [ ] Preprocessing is fitted only on training data; held-out test data is used only for final evaluation.
- [ ] Evaluation results and the metric rationale are reported, including the implications of class imbalance and false-negative/false-positive costs.
- [ ] A saved model plus its fitted preprocessing can be loaded and used by the web app to make a prediction from a machine reading.
- [ ] The UI explains its output sufficiently for a maintenance engineer to interpret it and does not overstate certainty.
- [ ] A live application URL is present in the README and accessible at submission time.
- [ ] The system-design diagram includes ingestion, training, serving, monitoring, and the production additions absent from the challenge implementation.
- [ ] The README includes at least two defined monitoring metrics, a retraining and model-validation approach, and deployment risks with mitigations.

No minimum model-accuracy score or specific model, metric, UI framework, deployment provider, or risk threshold is prescribed. The implementer must select and justify these choices in the repository documentation.

## 7. Constraints and non-goals

- **Time estimate:** 6–10 hours.
- The application need not be production-grade, but it must be live and usable at submission time.
- Production monitoring, retraining automation, and enterprise infrastructure are to be addressed in the design/critique; the challenge does not require building those full production systems.
- Model performance alone is not the success criterion. Reproducibility, leakage-safe evaluation, engineering quality, and operational clarity are central.

## 8. Decisions to make explicit

The implementer should record these decisions and their rationale in `README.md`:

- Whether to use the supplied dataset file or retrieve another copy.
- The target definition and the input features used to predict it.
- Any excluded columns and how categorical or missing values are handled.
- The train/test strategy and any validation or cross-validation strategy.
- The primary evaluation metric and any supporting metrics or operating threshold.
- The model choice and its interpretability/performance trade-off.
- The prediction shown in the application and how users should act on it.
- The deployment provider and the limitations of the deployed demonstration.
