# 🚗 Accident Severity Prediction

### End-to-End Machine Learning Pipeline | Datathon Project

---

## 📌 Overview

This project presents a complete machine learning pipeline to predict **accident severity** using structured tabular data. The goal was to accurately classify accidents into:

* **Slight Injury**
* **Serious Injury**
* **Fatal Injury**

Despite severe class imbalance (~1% fatal cases), the model is optimized for **Macro F1-score**, ensuring fair performance across all classes.

---

## 🧠 Problem Statement

Traditional models maximize accuracy, which leads to poor detection of rare but critical events (like fatal accidents).

This project focuses on:

> 🎯 **Maximizing Macro F1-score**
> Ensuring minority classes are not ignored

---

## 📊 Dataset

* ~12,000 rows
* 30+ features
* Mostly categorical data
* Highly imbalanced target distribution

---

## 🔍 Key Challenges

* ⚠️ Extreme class imbalance
* ⚠️ High-dimensional sparse features (OHE)
* ⚠️ Noisy categorical combinations
* ⚠️ Risk of data leakage during resampling

---

## ⚙️ Pipeline Overview

```text
Raw Data
→ Data Cleaning
→ Feature Encoding (OHE)
→ Model Training (LightGBM)
→ Probability Prediction
→ Threshold Optimization
→ Rule-Based Overrides
→ Final Prediction
```

---

## 🧪 Experiments Conducted

### Models Tested

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM ✅ (Best)
* CatBoost
* Neural Networks

---

### Imbalance Strategies Tried

| Method        | Result                          |
| ------------- | ------------------------------- |
| SMOTE         | ❌ Poor (invalid synthetic data) |
| Class Weights | ⚠️ Unstable                     |
| Undersampling | ⚠️ Loss of information          |
| No Resampling | ✅ Best                          |

---

## 🧠 Key Insight

> ❗ The bottleneck was NOT the model
> ✅ It was the **decision strategy**

---

## 🔥 Breakthrough: Threshold Optimization

Instead of relying on:

```python
argmax(probabilities)
```

We used:

```python
prediction = argmax(probabilities * weights)
```

Example:

```python
weights = [10, 3.5, 1]
```

---

### 🧠 Why it works

* Boosts minority class importance
* Improves recall without retraining
* Maintains strong precision

---

## ⚡ Rule-Based Overrides

Further improvements using domain-aware rules:

```python
if fatal_probability > threshold:
    predict Fatal
```

---

## 🏆 Final Results

| Metric                              | Score      |
| ----------------------------------- | ---------- |
| **Macro F1**                        | **~0.50+** |
| Accuracy                            | ~80%       |
| Balanced performance across classes |            |

---

## 📊 Model Comparison (Top Results)

| Model    | Strategy | Macro F1  |
| -------- | -------- | --------- |
| LightGBM | None     | **0.506** |
| XGBoost  | None     | 0.4728    |
| CatBoost | None     | 0.4711    |

---

## 💾 Saved Artifacts

| File                    | Description                |
| ----------------------- | -------------------------- |
| `lgbm_model.pkl`        | Trained LightGBM model     |
| `label_encoder.pkl`     | Target encoding            |
| `threshold_weights.pkl` | Optimized decision weights |

---

## 🌐 Streamlit Deployment

Interactive UI built using **Streamlit**:

### Features:

* User-friendly input form
* Real-time predictions
* Clean probability display
* Visualization of class probabilities

---

### Run Locally

```bash
streamlit run scripts/app.py
```

---

## 📦 Project Structure

```bash
data/
  ├── raw/
  └── processed/

models/
  ├── lgbm_model.pkl
  ├── label_encoder.pkl
  └── threshold_weights.pkl

notebooks/
scripts/
  ├── app.py
  ├── compare_models.py
  ├── hypertuning.py

requirements.txt
README.md
```

---

## 🧠 Key Learnings

* ✅ **Decision strategy > model complexity**
* ✅ SMOTE fails on high-cardinality categorical data
* ✅ Threshold tuning is critical for imbalanced problems
* ✅ Feature interactions matter more than raw features

---

## 🚀 Future Improvements

* Target Encoding instead of OHE
* SHAP-based explainability
* Ensemble stacking
* Bayesian hyperparameter tuning
* Advanced feature engineering

---

## 🏁 Conclusion

This project demonstrates how to move beyond standard ML workflows and build a **robust, competition-grade solution**.

> 🎯 Real success came from understanding *how the model makes decisions*, not just training it.

---

## 👤 Author

Built as part of a Datathon challenge focused on real-world ML problem solving.

---

## ⭐ If you found this useful

Give the repo a ⭐ and feel free to fork or contribute!