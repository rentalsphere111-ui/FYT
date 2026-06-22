import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
import joblib

print("--- Starting Hyperparameter Tuning for Free-Kick Model ---")

# 1. Load & Prep Data
df = pd.read_csv("Premier_League_Shots_14_24_Direct.csv")

# Filter for Direct Free Kicks
df_dfk = df[df['situation'] == 'DirectFreekick'].copy()
df_dfk['is_goal'] = df_dfk['result'].apply(lambda x: 1 if x == 'Goal' else 0)

# Calculate standard coordinates
df_dfk['x_m'] = df_dfk['X'] * 105
df_dfk['y_m'] = df_dfk['Y'] * 68
df_dfk['distance'] = np.sqrt((105 - df_dfk['x_m'])**2 + (34 - df_dfk['y_m'])**2)

def calc_angle(x, y):
    a = np.sqrt((105 - x)**2 + (30.34 - y)**2)
    b = np.sqrt((105 - x)**2 + (37.66 - y)**2)
    c = 7.32
    if a * b == 0: return 0
    return np.degrees(np.arccos(np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)))

df_dfk['angle'] = df_dfk.apply(lambda row: calc_angle(row['x_m'], row['y_m']), axis=1)

# Synthetic features
np.random.seed(42)

# 1. wall_count (1 to 6)
base_wall = 8 - (df_dfk['distance'] / 4.5)
wall_noise = np.random.normal(0, 0.4, size=len(df_dfk))
df_dfk['wall_count'] = np.clip(np.round(base_wall + wall_noise), 1, 6).astype(int)

# 2. wall_jump (0 or 1)
jump_prob = np.where((df_dfk['distance'] >= 18) & (df_dfk['distance'] <= 30), 0.75, 0.25)
df_dfk['wall_jump'] = (np.random.random(size=len(df_dfk)) < jump_prob).astype(int)

# 3. croc_present (0 or 1)
season_year = df_dfk['season'].apply(lambda s: int(s.split('/')[0]))
croc_prob = np.where(season_year >= 2019, 0.60, 0.01)
df_dfk['croc_present'] = (np.random.random(size=len(df_dfk)) < croc_prob).astype(int)

# 4. gk_position (Standard, DefensiveError, OverCovering)
gk_random = np.random.random(size=len(df_dfk))
df_dfk['gk_position'] = np.where(gk_random < 0.70, 'Standard', 
                                 np.where(gk_random < 0.88, 'DefensiveError', 'OverCovering'))

# Target alignment
error_mask = (df_dfk['gk_position'] == 'DefensiveError') & (df_dfk['distance'] < 28) & (df_dfk['is_goal'] == 0)
df_dfk.loc[df_dfk[error_mask].sample(frac=0.25, random_state=42).index, 'is_goal'] = 1

small_wall_mask = (df_dfk['wall_count'] <= 3) & (df_dfk['distance'] < 24) & (df_dfk['is_goal'] == 0)
df_dfk.loc[df_dfk[small_wall_mask].sample(frac=0.30, random_state=42).index, 'is_goal'] = 1

block_mask = (df_dfk['wall_jump'] == 1) & (df_dfk['croc_present'] == 1) & (df_dfk['is_goal'] == 1)
df_dfk.loc[df_dfk[block_mask].sample(frac=0.15, random_state=42).index, 'is_goal'] = 0

# Prepare dataset
features_to_use = ['distance', 'angle', 'minute', 'wall_count', 'wall_jump', 'croc_present', 'gk_position', 'shotType']
X = df_dfk[features_to_use].copy()
y = df_dfk['is_goal']

# One-hot encoding
X = pd.get_dummies(X, columns=['gk_position', 'shotType'], drop_first=True)

# Ensure consistent column naming/presence
expected_cols = [
    'distance', 'angle', 'minute', 'wall_count', 'wall_jump', 'croc_present',
    'gk_position_DefensiveError', 'gk_position_OverCovering', 'shotType_RightFoot'
]
for col in expected_cols:
    if col not in X.columns:
        X[col] = 0
X = X[expected_cols]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Define the Search Space
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5],
    'scale_pos_weight': [5, 10, 15, 20, 25]  # Weight for goal class (imbalance is ~1:18)
}

# 3. Setup the Tuner
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=30,              # Try 30 combinations
    scoring='roc_auc',      # Optimize for AUC specifically
    cv=3,                   # 3-fold cross validation
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# 4. Run the Search
print("Tuning in progress... please wait...")
search.fit(X_train, y_train)

# 5. Results
best_model = search.best_estimator_
y_prob = best_model.predict_proba(X_test)[:, 1]
new_auc = roc_auc_score(y_test, y_prob)

print("\n--- TUNING COMPLETE ---")
print(f"Best CV AUC Score: {search.best_score_:.4f}")
print(f"Test Set AUC Score: {new_auc:.4f}")
print("Best Parameters Found:")
print(search.best_params_)

# Save the best model
joblib.dump(best_model, 'xgboost_tuned_best.pkl')
print("\nSaved best model as 'xgboost_tuned_best.pkl'")