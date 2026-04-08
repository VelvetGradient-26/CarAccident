# ---------------------------------------------------------
# 🏆 LIGHTGBM GRID SEARCH (MACRO F1 OPTIMIZATION)
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score

from lightgbm import LGBMClassifier

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("Loading data...")
df = pd.read_csv('/Users/deepak/Desktop/Datathon/data/processed/road_processed.csv')

X = df.drop('Accident_severity', axis=1)
y = df['Accident_severity']

le = LabelEncoder()
y = le.fit_transform(y)

# OHE
X = pd.get_dummies(X, drop_first=True)
X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in X.columns]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 2. DEFINE MODEL
# ---------------------------------------------------------
lgbm = LGBMClassifier(random_state=42, n_jobs=-1)

# ---------------------------------------------------------
# 3. PARAM GRID (SMART RANGE)
# ---------------------------------------------------------
param_grid = {
    'n_estimators': [300, 500],
    'learning_rate': [0.03, 0.05],
    'num_leaves': [31, 63],
    'min_child_samples': [20, 40],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# ---------------------------------------------------------
# 4. MACRO F1 SCORER
# ---------------------------------------------------------
scorer = make_scorer(f1_score, average='macro')

# ---------------------------------------------------------
# 5. GRID SEARCH
# ---------------------------------------------------------
print("\nRunning GridSearchCV... (this will take time)")

grid = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring=scorer,
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. BEST RESULTS
# ---------------------------------------------------------
print("\n" + "="*50)
print("🏆 BEST PARAMETERS")
print("="*50)
print(grid.best_params_)

print("\nBest CV Macro F1:", round(grid.best_score_, 4))

# ---------------------------------------------------------
# 7. EVALUATE ON TEST SET
# ---------------------------------------------------------
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

test_f1 = f1_score(y_test, y_pred, average='macro')

print("\nTest Macro F1 (no thresholding):", round(test_f1, 4))