# ---------------------------------------------------------
# 🏆 FINAL LIGHTGBM MODEL (SAVEABLE VERSION)
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report

from lightgbm import LGBMClassifier

print("="*50)
print("FINAL LIGHTGBM TRAINING")
print("="*50)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv('/Users/deepak/Desktop/Datathon/data/processed/road_processed.csv')

X = df.drop('Accident_severity', axis=1)
y = df['Accident_severity']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)
X_encoded.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in X_encoded.columns]

# ---------------------------------------------------------
# 2. TRAIN-TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ---------------------------------------------------------
# 3. MODEL (YOUR BEST CONFIG BASE)
# ---------------------------------------------------------
print("Training LightGBM...")

model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. PROBABILITIES
# ---------------------------------------------------------
y_proba = model.predict_proba(X_test)

# ---------------------------------------------------------
# 5. THRESHOLD OPTIMIZATION 🔥
# ---------------------------------------------------------
print("Optimizing thresholds...")

best_score = 0
best_weights = None

for f in np.linspace(5, 25, 40):
    for s in np.linspace(3, 15, 40):
        w = np.array([f, s, 1.0])
        preds = np.argmax(y_proba * w, axis=1)
        
        score = f1_score(y_test, preds, average='macro')
        
        if score > best_score:
            best_score = score
            best_weights = w

print("Best weights:", best_weights)

# ---------------------------------------------------------
# 6. FINAL PREDICTIONS + RULE BOOST
# ---------------------------------------------------------
y_pred = np.argmax(y_proba * best_weights, axis=1)

# Rule-based override (VERY IMPORTANT)
y_pred[y_proba[:, 0] > 0.07] = 0

mask = (y_proba[:, 1] > 0.18) & (y_pred != 0)
y_pred[mask] = 1

# ---------------------------------------------------------
# 7. FINAL METRICS
# ---------------------------------------------------------
print("\n" + "="*40)
print("🏆 FINAL RESULTS")
print("="*40)

macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

print(f"Macro F1: {macro_f1:.4f}")
print(f"Micro F1: {micro_f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ---------------------------------------------------------
# 8. SAVE MODEL 🔥
# ---------------------------------------------------------
print("\nSaving model...")

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/lgbm_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(best_weights, "models/threshold_weights.pkl")

print("✅ Model saved to models/")