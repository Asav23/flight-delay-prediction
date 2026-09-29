# ✈️ Flight Delay Prediction — UK & Ireland

Predicting whether a flight will depart 15+ minutes late using schedule, route and weather data, and identifying which operational factors drive delays.

The project covers the full workflow: merging raw datasets, feature engineering, exploratory analysis, training and comparing three classifiers under class imbalance, and feature-importance analysis to explain the drivers of delay.

---

## Dataset

Two CSV files covering **3,000 flights across 2024** on four carriers (Aer Lingus, British Airways, Ryanair, easyJet), operating between UK and Irish airports including Dublin, Shannon, Belfast, Glasgow, London Gatwick and London City.

| File | Contents |
|---|---|
| `flights_schedule.csv` | Flight ID, date, scheduled departure/arrival, carrier, origin, destination, distance, aircraft type |
| `flight_conditions_outcomes.csv` | Flight ID, precipitation, wind conditions, actual departure delay (minutes) |

The files are joined on `flight_id`. The target variable `delayed` is 1 when the departure delay is 15 minutes or more, which is the standard industry threshold for on-time performance. About **20% of flights are delayed**, so the classes are imbalanced.

---

## Approach

**1. Data preparation and feature engineering**
- Merged the schedule and conditions datasets, and converted mixed-type fields to numeric.
- Parsed `HHMM` departure times into hour, minute and fractional-hour features.
- Derived weekday, month and an `is_international` flag.
- Reduced high-cardinality airport fields by keeping the top 20 origins and destinations and grouping the rest as `OTHER`.
- Built a scikit-learn `ColumnTransformer` (standard scaling plus one-hot encoding) inside a `Pipeline`, so preprocessing is fitted only on training data and does not leak test information.

**2. Exploratory analysis.** Delay rates were broken down by weekday, precipitation, carrier and aircraft type (see findings below).

**3. Classification and evaluation**
- Models: **Logistic Regression** (class-weighted baseline), **Gradient Boosting** and **XGBoost**.
- 70/30 stratified train-test split, plus **5-fold stratified cross-validation**.
- Metrics: accuracy, precision, recall, F1 and ROC-AUC. Because of the class imbalance, F1 and ROC-AUC are the primary metrics; accuracy alone is misleading.

**4. Predictive feature analysis.** Features were ranked using **mutual information**, alongside Random Forest and permutation importance, all computed on training data only.

---

## Results

5-fold cross-validation, mean scores:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.652 | 0.323 | **0.629** | **0.426** | **0.691** |
| Gradient Boosting | **0.795** | **0.501** | 0.160 | 0.243 | 0.673 |
| XGBoost | 0.753 | 0.338 | 0.208 | 0.257 | 0.605 |

**Takeaway:** the class-weighted Logistic Regression was the most useful model. It catches about 63% of delayed flights and has the best F1 and ROC-AUC. The boosted models reach higher accuracy mostly by predicting "on time", and they miss 80%+ of actual delays. This is a clear example of why accuracy is the wrong metric for imbalanced operational problems: a model that never predicts a delay would still score about 80% accuracy.

---

## Key findings

- **Weather:** the delay rate rises from 17% in dry conditions to 31% in heavy rain.
- **Day of week:** Monday is the worst day (25% delayed) and Thursday the most reliable (16%).
- **Carrier:** in this dataset, Ryanair (31%) and easyJet (23%) show higher delay rates than Aer Lingus and British Airways (14% each).
- **Aircraft type:** makes almost no difference; regional, narrowbody and widebody aircraft all sit around 20%.
- **Top predictive features (mutual information):** specific destinations (Glasgow, Dublin) and origins (Belfast, Gatwick, Shannon), scheduled departure minute, distance, carrier, and wind conditions. Airport and operational factors outweigh aircraft characteristics.

---

## Limitations and next steps

- The overall predictive signal is modest (best ROC-AUC ≈ 0.69), which is expected with 3,000 flights and no historical or real-time operational features.
- Boosted models were used with default settings. Class weighting (`scale_pos_weight`), hyperparameter tuning and decision-threshold adjustment would likely improve their recall.
- Useful additional features would include previous-leg delays (knock-on delays), airport congestion at the scheduled slot, and more detailed weather data.
- SHAP values would give per-prediction explanations that are more reliable than global importance scores.

---

## Repository structure

```
├── flight_delay_analysis.ipynb        # Full analysis notebook
├── flights_schedule.csv               # Schedule and route data
├── flight_conditions_outcomes.csv     # Weather and delay outcomes
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/Asav23/flight-delay-prediction.git
cd flight-delay-prediction
pip install -r requirements.txt
jupyter notebook flight_delay_analysis.ipynb
```

**requirements.txt:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `jupyter`

## Tech stack

Python · Pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · Jupyter


