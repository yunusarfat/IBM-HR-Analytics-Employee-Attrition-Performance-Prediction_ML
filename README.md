# 🧠 Employee Attrition Prediction (IBM HR Analytics)

## 📌 Project Overview

This project predicts whether an employee is likely to leave a company using the **IBM HR Analytics Employee Attrition dataset**.

The goal is to help organizations identify at-risk employees and take proactive measures to improve retention.


## 🎯 Problem Statement

Employee attrition is a major challenge for companies. This project builds a **machine learning model** that classifies employees into:

* ✅ **0 → Will Stay**
* ❌ **1 → Will Leave**

---

## 📊 Dataset Information

* Dataset: IBM HR Analytics Employee Attrition & Performance
* Total Records: 1470
* Features: 35 (before preprocessing)

### Target Variable:

* **Attrition**

  * No → 0
  * Yes → 1

### Class Distribution:

* No: 1233
* Yes: 237
  👉 Imbalanced dataset

---

## ⚙️ Preprocessing Steps

The following preprocessing steps were applied:

1. **Removed irrelevant columns**

   * EmployeeCount, StandardHours, Over18, EmployeeNumber

2. **Target Encoding**

   * Attrition mapped to 0 and 1

3. **Handling Missing Values**

   * No missing values found

4. **Categorical Encoding**

   * OneHotEncoder used for categorical variables

5. **Feature Scaling**

   * StandardScaler applied to numerical features

6. **Feature Engineering**

   * IncomePerYear
   * YearsAtCompanyRatio
   * DistanceIncome

---

## 🤖 Model Used

### Random Forest Classifier

### Why Random Forest?

* Handles mixed data types well
* Works well with non-linear relationships
* Robust to overfitting
* Performs well on imbalanced datasets (with class_weight)

---

## 🔁 Model Pipeline

A complete pipeline was built using:

* **ColumnTransformer**
* **StandardScaler**
* **OneHotEncoder**
* **RandomForestClassifier**

👉 This ensures preprocessing + model training happen together.

---

## 📈 Model Evaluation

### Cross-Validation Results:

* Mean F1 Score: **0.147**
* Standard Deviation: **0.058**

### After Hyperparameter Tuning:

* Best F1 Score: **0.45**

### Test Set Performance:

| Metric    | Class 0 | Class 1 |
| --------- | ------- | ------- |
| Precision | 0.86    | 0.36    |
| Recall    | 0.94    | 0.19    |
| F1-score  | 0.90    | 0.25    |

### Accuracy:

* **82%**

## 🔧 Hyperparameter Tuning

Used **GridSearchCV** with parameters:

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf

### Best Parameters:

```
n_estimators = 300  
max_depth = 10  
min_samples_split = 10  
min_samples_leaf = 4
```

---

## 🌐 Web Application (Gradio)

A user-friendly interface was built using **Gradio**.

### Features:

* Input employee details
* Predict attrition instantly
* Uses trained pipeline (no manual preprocessing needed)

---

## 🚀 How to Run the Project

### 1. Clone the repository

```
git remote add origin https://github.com/yunusarfat/IBM-HR-Analytics-Employee-Attrition-Performance.git
cd project-folder
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:7860/
```

---

## 📁 Project Structure

```
├── app.py
├── modelRF.pkl
├── notebook.ipynb
├── requirements.txt
└── README.md
```

---

## 📌 Key Learnings

* Importance of using pipelines in ML
* Handling imbalanced datasets
* Feature engineering impact
* Model evaluation using F1-score
* Deploying ML models with Gradio

---

## 🔮 Future Improvements

* Use SMOTE for balancing data
* Try advanced models (XGBoost, LightGBM)
* Improve recall for minority class
* Add probability/confidence output in UI

---

## 🙌 Conclusion

This project demonstrates a complete machine learning workflow:

➡️ Data preprocessing
➡️ Model training
➡️ Evaluation
➡️ Hyperparameter tuning
➡️ Deployment with Gradio

---

## 👨‍💻 Author

**Arfat**

---
