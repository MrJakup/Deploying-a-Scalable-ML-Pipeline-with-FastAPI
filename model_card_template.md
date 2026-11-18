# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a supervised binary classification pipeline developed in November 2026 (Version 1.0) for the Udacity Machine Learning DevOps Engineer nanodegree project. It uses a `RandomForestClassifier` implemented in `scikit-learn` to predict whether an individual's annual income exceeds $50,000 based on demographic and employment attributes. Hyperparameters were tuned using `RandomizedSearchCV` with 3-fold cross-validation optimizing for the F1 score across tree depth, number of estimators, split criteria, and class weighting. Categorical input features are encoded using a fitted `OneHotEncoder`, and target labels are encoded using a fitted `LabelBinarizer`.

## Intended Use
The primary intended use of this model is educational and demonstrative, specifically to showcase an end-to-end machine learning pipeline deployed via a FastAPI REST service with automated CI/CD testing and data version control. Its primary intended users are machine learning engineers, developers, and reviewers evaluating MLOps workflows. This model is strictly out-of-scope for high-stakes automated decision-making, such as credit underwriting, loan approvals, hiring, or determining eligibility for financial or government assistance.

## Training Data
The model was trained on an 80% random split (approximately 26,048 rows) of the UCI Census Income dataset (`census.csv`), which contains 32,561 total records derived from the 1994 U.S. Census database. Each record includes 14 demographic and employment features, such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours worked per week, and native country. During preprocessing, the eight categorical columns were one-hot encoded and the binary target variable (`salary`) was transformed into `0` (`<=50K`) and `1` (`>50K`). The dataset exhibits notable class imbalance, with approximately 76% of instances belonging to the `<=50K` class and 24% belonging to the `>50K` class.

## Evaluation Data
The model was evaluated on the remaining 20% held-out test split (approximately 6,513 rows) generated using `train_test_split` with `random_state=42`. To prevent data leakage, the test set was transformed using the `OneHotEncoder` and `LabelBinarizer` instances fitted exclusively on the training data (`training=False`). In addition to aggregate evaluation across the entire test set, disaggregated slice evaluations were performed across every unique value of the categorical features to check for performance disparities across demographic and socioeconomic groups.

## Metrics
Model performance was evaluated using **Precision**, **Recall**, and the **F1 Score** (`F-beta` with `beta=1`) at the standard `0.5` classification threshold. These metrics were selected over raw accuracy because the dataset is imbalanced toward the `<=50K` class, and precision and recall better capture false positives and false negatives when predicting the minority `>50K` class.

* **Overall Test Set Performance:** On the held-out test set, the tuned model achieved a Precision of `0.5777`, a Recall of `0.8587`, and an F1 Score of `0.6907`.
* **Categorical Slice Performance (`sex`):** On the `Male` slice, the model achieved a Precision of `0.5756`, a Recall of `0.8849`, and an F1 Score of `0.6975`, whereas on the `Female` slice, it achieved a Precision of `0.5935`, a Recall of `0.7082`, and an F1 Score of `0.6458`.
* **Categorical Slice Performance (`race`):** Across racial subgroups, performance varied with sample size, achieving an F1 Score of `0.6918` on `White` (`n = 5,595`) and `0.6706` on `Black` (`n = 599`). Full slice metrics across all categorical columns are recorded in `slice_output.txt`.
  
## Ethical Considerations
The training data includes sensitive protected attributes such as `race`, `sex`, `age`, and `native-country`, alongside historical socioeconomic disparities from 1994. Because historical income distributions reflect systemic inequities—such as wage gaps across gender and racial groups, a model trained on this data can learn and perpetuate biased correlations. Using such predictions in real-world financial or employment workflows could result in unfair discrimination against historically marginalized groups. To surface these disparities transparently, disaggregated slice metrics were computed across all demographic features during evaluation.

## Caveats and Recommendations
This dataset is over three decades old and reflects 1994 U.S. income thresholds and labor demographics, meaning its predictions do not generalize to modern economic conditions. Furthermore, some demographic slices in the test set (such as smaller `native-country` or `race` categories) contain very few positive `>50K` examples, leading to higher variance in slice-level precision and recall scores. Before any real-world analytical use, the model should be retrained on current, representative data, evaluated across intersectional subgroups (such as `race` combined with `sex`), and paired with fairness constraints or threshold adjustments to mitigate disparate error rates.