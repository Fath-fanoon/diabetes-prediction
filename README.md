# Diabetes Prediction from Patient Health Measurements

A small end-to-end machine learning project that predicts whether a patient has diabetes from eight clinical measurements, with a focus on honest evaluation of an imbalanced healthcare dataset.

**Notebook:** `diabetes-prediction.ipynb` (open it in Google Colab and run all cells)

## Problem

Given measurements such as glucose, BMI, and age, can we classify a patient as diabetic or not? Because the data is imbalanced, accuracy alone is misleading, so I evaluate with precision, recall, F1, and ROC-AUC.

## Data

- **Source:** Pima Indians Diabetes dataset, loaded from OpenML (originally from the UCI Machine Learning Repository / NIDDK).
- **Size:** 768 patients, 8 features, binary target (268 diabetic, about 35%; 500 non-diabetic).
- **Features:** pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age.

## Method

1. **Cleaning:** Several columns (glucose, blood pressure, skin thickness, insulin, BMI) contained zeros that are physiologically impossible. I treated these as missing values. Insulin was missing for 374 patients and skin thickness for 227.
2. **Imputation:** Missing values were filled with the median, learned from training data only (inside a scikit-learn `Pipeline`) to avoid data leakage.
3. **Split:** 80/20 stratified train/test split (test set: 154 patients).
4. **Models:** A majority-class baseline, logistic regression (with feature scaling), and a random forest (200 trees).
5. **Evaluation:** Held-out test set, plus 5-fold stratified cross-validation.

## Results

**Held-out test set (154 patients):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (always "no") | 0.649 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 |
| Random Forest | 0.740 | 0.652 | 0.556 | 0.600 | 0.817 |

**5-fold cross-validation (mean ± std):**

| Model | ROC-AUC | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.837 ± 0.020 | 0.571 ± 0.047 | 0.636 ± 0.021 |
| Random Forest | 0.826 ± 0.021 | 0.612 ± 0.035 | 0.645 ± 0.035 |

### Key findings

- Both models clearly beat the baseline. The baseline reaches 65% accuracy but finds none of the diabetic patients, which shows why accuracy is a poor metric here.
- The two models perform comparably. Differences between them are within one standard deviation across folds, so I do not claim either is better.
- **Glucose is the strongest predictor.** It has the highest correlation with the outcome (r = 0.49) and the highest random forest importance (0.268), followed by BMI (0.158), diabetes pedigree (0.124), and age (0.120).
- Diabetes pedigree ranks third in the forest despite a modest linear correlation (r = 0.17), suggesting a non-linear relationship that a correlation alone does not capture.

## Limitations

- **Recall is moderate.** The models detect only about 57-61% of diabetic patients in cross-validation, so they would miss many cases. This would matter in a real screening setting.
- **Small dataset.** 768 patients, with a 154-patient test set, so estimates are noisy.
- **Limited population.** All patients are women of Pima Indian heritage, aged 21 or older, so results may not generalize to other populations.
- **Heavy missingness.** Nearly half of the insulin values were missing, and median imputation is a simple choice that may distort the data.
- **Feature importance caveat.** Random forest impurity-based importance can favour features with many distinct values, so rankings should be read as indicative only.
- This is an educational project and not a clinical tool.

## Possible next steps

- Tune hyperparameters and try gradient boosting.
- Adjust the decision threshold or use class weighting to improve recall.
- Compare imputation methods (for example, KNN or iterative imputation).
- Add permutation importance or SHAP for a more reliable interpretation.

## Tools

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Google Colab
