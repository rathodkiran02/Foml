# 🌾 Farm Land Classification using Machine Learning

## 📌 Overview

This project focuses on predicting agricultural land classification into **low**, **medium**, and **high** categories using machine learning techniques.

The solution involves:

* Extensive **data preprocessing & feature engineering**
* Handling **missing values and imbalanced data**
* Training and evaluating models like **Random Forest** and **XGBoost**
* Improving performance using **threshold tuning**

---

## 👥 Team

* **R Kiran Kumar (ES22BTECH11030)**
* **Challa Srikrishna Reddy (ES22BTECH11006)**

---

## 📂 Dataset

* **Train Data:** 112,569 rows × 58 columns
* **Test Data:** 15,921 rows × 57 columns

Target classes:

* `low` → 0
* `medium` → 1
* `high` → 2

⚠️ Dataset is **imbalanced**:

* Medium: ~60%
* Low & High: ~20% each

---

## ⚙️ Workflow

### 1. Data Preprocessing

* Dropped irrelevant columns (IDs, redundant features)
* Removed features with **>80% missing values**
* Handled missing values:

  * Median for high-unique features
  * Mode for low-unique features
* Feature analysis using:

  * NaN percentage
  * Unique values
  * Statistical summaries

---

### 2. Exploratory Analysis

* Checked label distribution
* Visualized missing data & feature distributions
* Observed class imbalance → required model adjustment

---

### 3. Model Building

#### 🔹 Random Forest

* Initial model showed bias toward **medium class**
* Applied **custom threshold tuning**
* Improved Macro F1 Score:

  * From ~0.35 → ~0.41

---

#### 🔹 XGBoost (Final Model 🚀)

* Reduced overfitting compared to Gradient Boosting
* Used **Stratified 5-Fold Cross Validation**
* Applied **probability threshold tuning**

✅ Final Performance:

* **Macro F1 Score:** ~0.44
* Balanced performance across all classes

---

### 4. Threshold Optimization

* Adjusted class probabilities instead of raw predictions
* Found optimal thresholds:

  * Low: `0.23`
  * High: `0.27`

This significantly improved classification of minority classes.

---

### 5. Final Prediction

* Predictions generated on test dataset
* Output stored as:

```bash
submission.csv
```

Format:

```
UID, Target
```

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib
* Scikit-learn
* XGBoost

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib xgboost
```

---

### 2. Run the notebook / script

Make sure you have:

* `train.csv`
* `test.csv`

---

### 3. Generate predictions

```bash
python your_script.py --test-file test.csv --predictions-file submission.csv
```

---

## 📊 Key Insights

* Dataset is highly **imbalanced**, requiring threshold tuning
* Removing high-missing-value features improves performance
* **XGBoost + threshold tuning** outperforms Random Forest
* Cross-validation ensures model stability

---

## 📈 Future Improvements

* Hyperparameter tuning (Grid Search / Optuna)
* Try deep learning models
* Feature importance analysis

---

## 📜 License

This project is for academic and learning purposes.

---

## ⭐ Acknowledgements

* Hackathon Team: **ByteBots**
* Dataset provided as part of ML challenge
