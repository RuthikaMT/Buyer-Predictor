# Buyer Predictor — Predicting Online Purchases with 5 Classifiers

**M.Tech (AIML) · Machine Learning · Assignment 2**
Student: `Ruthika M T` 

- **GitHub repository:** https://github.com/<your-username>/<repo-name>
- **Live Streamlit app:** https://<your-app-name>.streamlit.app

---

## a. Problem statement

An e-commerce site sees thousands of browsing sessions a day, but only a small share end in a purchase.
The goal is to **predict, from a session's browsing behaviour, whether it will end in a purchase
(`Revenue = True`) or not** — a **binary classification** problem. Five classical classifiers are trained
on the same data, compared on six metrics, and served through an interactive Streamlit app that
evaluates the saved models on an uploaded test file without retraining.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository (ID 468) — https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Instances | 12,330 sessions (each from a different user over a one-year period) |
| Features | 17 input features (10 numerical, 7 categorical) + 1 target |
| Target | `Revenue` — True if the session ended in a purchase |
| Class balance | 10,422 no-purchase (84.5%) vs 1,908 purchase (15.5%) — imbalanced |
| Missing values | None |

**Features**

| Group | Columns | Meaning |
|---|---|---|
| Page counts & time | `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration` | Pages of each type visited and seconds spent on them |
| Google Analytics | `BounceRates`, `ExitRates`, `PageValues` | Average bounce rate, exit rate and page value of the pages visited |
| Calendar | `SpecialDay`, `Month`, `Weekend` | Closeness to a special day (e.g. Valentine's Day), month, weekend flag |
| Technical / traffic | `OperatingSystems`, `Browser`, `Region`, `TrafficType` | Integer-coded categories |
| Visitor | `VisitorType` | Returning, New or Other visitor |

**Preprocessing (inside each saved pipeline)**
- `log1p` + standard scaling for the right-skewed page counts, durations and `PageValues`
- Standard scaling for `BounceRates`, `ExitRates`, `SpecialDay`
- One-hot encoding for `Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`
  (integer codes are labels, not quantities)
- `Weekend` and `Revenue` converted to 0/1
- Stratified 80/20 split (`random_state=42`): 9,864 training rows, **2,466 test rows → `test_data.csv`**
- Hyper-parameters tuned with 5-fold stratified cross-validation on the training split only (scoring = F1)

## c. GitHub repository

https://github.com/<your-username>/<repo-name>

```
ml-assignment-2/
├── app.py                       # Streamlit app (loads saved models, no retraining)
├── train_models.py              # Trains, tunes, evaluates and saves all 5 models
├── requirements.txt
├── README.md
├── test_data.csv                # Held-out test split (features + Revenue) to upload in the app
├── data/
│   └── online_shoppers_intention.csv   # Full UCI dataset
├── src/
│   ├── __init__.py
│   └── utils.py                 # Column lists, input validation, metric functions
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics.csv              # Comparison table below
    └── training_info.json       # Split sizes + best hyper-parameters
```

**Run locally**
```bash
pip install -r requirements.txt
python train_models.py     # re-creates model/*.joblib, metrics.csv and test_data.csv
streamlit run app.py
```

## d. Models used

| Model | Configuration (best from CV) |
|---|---|
| Logistic Regression | `C=0.1`, `class_weight="balanced"` |
| Decision Tree | `max_depth=6`, `min_samples_leaf=30`, `class_weight="balanced"` |
| K-Nearest Neighbor | `n_neighbors=25`, distance-weighted |
| Naive Bayes | Gaussian, `var_smoothing=1e-3` |
| Support Vector Machine | RBF kernel, `C=0.5`, `class_weight="balanced"`, Platt-scaled probabilities |

### Comparison table (test set, 2,466 sessions, threshold 0.5)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8646 | 0.9160 | 0.5433 | 0.7880 | **0.6432** | **0.5779** |
| Decision Tree | 0.8431 | **0.9181** | 0.4961 | 0.8298 | 0.6210 | 0.5577 |
| KNN | **0.8905** | 0.9015 | **0.6918** | 0.5288 | 0.5994 | 0.5438 |
| Naive Bayes | 0.6103 | 0.7867 | 0.2625 | **0.8377** | 0.3998 | 0.2940 |
| Support Vector Machine | 0.8832 | 0.9142 | 0.6199 | 0.6361 | 0.6279 | 0.5587 |

> Note: predicting "no purchase" for everyone already gives 84.5% accuracy, so **F1 and MCC** are the
> fairer yardsticks on this imbalanced data.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Gave the best overall balance: top F1 (0.6432) and MCC (0.5779), with AUC 0.9160. It found about 79% of real buyers (recall 0.7880), but roughly half the visitors it flagged as buyers did not buy (precision 0.5433). PageValues and ExitRates change the chance of buying in a fairly straight-line way, so a simple linear model works well here. |
| Decision Tree | Best at ranking buyers above non-buyers (highest AUC, 0.9181) and caught 83% of buyers (recall 0.8298). Its precision was the lowest of the good models (0.4961), so it raised more false alarms. Limiting depth to 6 stopped overfitting, and its first split is on PageValues, which makes it easy to explain. |
| KNN | Highest accuracy (0.8905) and precision (0.6918): when it says "buyer", it is usually right. But it missed almost half the buyers (recall 0.5288), because most neighbours of any visitor are non-buyers. It is also the slowest at prediction, since it compares each visitor with every training row. |
| Naive Bayes | Weakest model: F1 0.3998, MCC 0.2940 and accuracy only 0.6103. It assumes all features are independent, but page counts and time spent are strongly linked (correlation up to 0.86). So it labels too many visitors as buyers: it catches 84% of them, but only 26% of its "buyer" calls are correct. |
| Support Vector Machine | Strong and steady: accuracy 0.8832 and AUC 0.9142, with precision (0.6199) and recall (0.6361) almost equal. The RBF kernel picks up curved patterns, but after probability calibration it is a little more careful than Logistic Regression, so its F1 (0.6279) and MCC (0.5587) are slightly lower. |
| **Overall winner** | **Logistic Regression.** It has the highest F1 and MCC, which matter most here because only 15.5% of visitors buy. Accuracy alone misleads: always guessing "no purchase" already scores 84.5%. SVM is a close second. |

## e. Streamlit app features

- **CSV upload** of test data (with a one-click download of the sample `test_data.csv`)
- **Model selection dropdown** for all 5 saved models
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix**, **classification report** and **ROC curve**
- **Compare-all-models tab** with a highlighted metrics table and bar chart
- Adjustable decision threshold and downloadable predictions
- Models are loaded from `model/`; the uploaded data is **never used for training**

## f. Deployment

Deployed on Streamlit Community Cloud: New app → this repository → branch `main` → main file `app.py` → Deploy.
