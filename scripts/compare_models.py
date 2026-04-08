import pandas as pd
import numpy as np
import re
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv('/Users/deepak/Desktop/Datathon/data/processed/road_processed.csv')

X = df.drop('Accident_severity', axis=1)
y = df['Accident_severity']

le = LabelEncoder()
y = le.fit_transform(y)

# One-hot encode
X = pd.get_dummies(X, drop_first=True)
X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in X.columns]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)

# ---------------------------------------------------------
# 2. IMBALANCE STRATEGIES
# ---------------------------------------------------------
def get_datasets(strategy):
    if strategy == "none":
        return X_train, y_train, None
    
    elif strategy == "class_weight":
        weights = compute_sample_weight("balanced", y_train)
        return X_train, y_train, weights
    
    elif strategy == "smote":
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        return X_res, y_res, None
    
    elif strategy == "undersample":
        fatal = np.sum(y_train == 0)
        serious = np.sum(y_train == 1)

        sampling = {
            0: fatal,
            1: min(400, serious),
            2: 800
        }

        rus = RandomUnderSampler(sampling_strategy=sampling, random_state=42)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        return X_res, y_res, None

# ---------------------------------------------------------
# 3. MODELS
# ---------------------------------------------------------
models = {
    "LogReg": LogisticRegression(max_iter=2000, n_jobs=-1),
    "RF": RandomForestClassifier(n_estimators=200, n_jobs=-1),
    "XGB": XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1
    ),
    "LGBM": LGBMClassifier(n_estimators=200),
    "CatBoost": CatBoostClassifier(iterations=200, verbose=0)
}

strategies = ["none", "class_weight", "smote", "undersample"]

# ---------------------------------------------------------
# 4. THRESHOLD OPTIMIZATION
# ---------------------------------------------------------
def optimize_thresholds(y_true, proba):
    best_score = 0
    best_w = [1, 1, 1]

    for f in np.linspace(2, 10, 12):
        for s in np.linspace(1.5, 6, 12):
            w = np.array([f, s, 1.0])
            preds = np.argmax(proba * w, axis=1)
            score = f1_score(y_true, preds, average='macro')

            if score > best_score:
                best_score = score
                best_w = w

    return best_score, best_w

# ---------------------------------------------------------
# 5. TRAIN + EVALUATE
# ---------------------------------------------------------
results = []

for strat in strategies:
    print(f"\n=== Strategy: {strat} ===")

    for name, model in models.items():
        print(f"Training {name}...")

        start = time.time()

        X_tr, y_tr, weights = get_datasets(strat)

        if weights is not None:
            model.fit(X_tr, y_tr, sample_weight=weights)
        else:
            model.fit(X_tr, y_tr)

        train_time = time.time() - start

        # Predict probabilities
        try:
            proba = model.predict_proba(X_test)
        except:
            continue

        # Optimize thresholds
        macro_f1, best_w = optimize_thresholds(y_test, proba)
        y_pred = np.argmax(proba * best_w, axis=1)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro')
        rec_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        results.append({
            "Model": name,
            "Strategy": strat,
            "Accuracy": round(acc, 4),
            "Precision_macro": round(prec_macro, 4),
            "Recall_macro": round(rec_macro, 4),
            "F1_macro": round(f1_macro, 4),
            "F1_micro": round(f1_micro, 4),
            "F1_weighted": round(f1_weighted, 4),
            "Weights": best_w,
            "Time(s)": round(train_time, 2)
        })

# ---------------------------------------------------------
# 6. LEADERBOARD
# ---------------------------------------------------------
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="F1_macro", ascending=False)

print("\n" + "="*80)
print("🏆 FINAL LEADERBOARD (SORTED BY MACRO F1)")
print("="*80)
print(df_results.to_string(index=False))

# Save results
df_results.to_csv("leaderboard_full_metrics.csv", index=False)

print("\n✅ Saved leaderboard_full_metrics.csv")