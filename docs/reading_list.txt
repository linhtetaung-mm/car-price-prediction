Your practical reading order
If you want to prepare efficiently, I'd do it in this order:
1. ISLP Chapter 2 — statistical learning, training/test error, bias/variance, regression basics.
2. ISLP Chapter 3 — understand linear regression well enough to explain coefficients, residuals and limitations.
3. Scikit-learn Getting Started https://scikit-learn.org/stable/getting_started.html — focus especially on transformers, ColumnTransformer, Pipeline, fitting/predicting and evaluation. GitHub
4. ISLP Chapter 5 — train/test splitting and cross-validation.
5. Scikit-learn Cross-validation guide https://scikit-learn.org/stable/modules/cross_validation.html — understand why the test set stays untouched during model selection. Scikit-learn
6. ISLP Chapter 8 — Decision Trees, bagging, Random Forest and boosting.
7. Return to your assignment and implement the complete ML experiment.
8. Only then learn Dash + Docker for Task 3.
If you understand roughly 70–80% of those concepts, start the assignment. Don't wait until you feel like you know everything.
One useful test is whether you can explain, without code: why we split the data, what leakage is, why preprocessing belongs inside a pipeline, why we need a baseline, why we use cross-validation, what MAE/RMSE/R² mean, why a tree can overfit, why Random Forest helps, why the professor log-transforms price, and why the test set is touched only at the end. If you can do that, you're ready to build the assignment and learn the remaining details while doing it.