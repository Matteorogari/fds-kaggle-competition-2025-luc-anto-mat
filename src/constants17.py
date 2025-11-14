import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier as RFC, HistGradientBoostingClassifier as HGBC

# Safely handle optional xgboost dependency
try:
    from xgboost import XGBClassifier as XGBC
    HAS_XGB = True
except Exception:
    XGBC = None
    HAS_XGB = False
    print("[WARN] xgboost not available: XGBoost-based models will be skipped.")

# --- Configuration and Paths ---
PROJECT_IDENTIFIER = 'fds-pokemon-battles-prediction-2025'
RESOURCE_PATH = os.path.join('../input', PROJECT_IDENTIFIER)
TRAIN_SOURCE = os.path.join(RESOURCE_PATH, 'train.jsonl')
TEST_SOURCE = os.path.join(RESOURCE_PATH, 'test.jsonl')

# --- Modeling Parameters ---
NUM_FOLDS = 5
C_VECTOR = np.logspace(-3, 3, 15)
RANDOM_SEARCH_COUNT = 20

# Weights for the power index (Statistics)
W_SPEED_F, W_ATT_SPATT_F, W_DEF_SPDEF_F, W_HP_F = 1.5, 1.0, 0.5, 0.4

# State Score Map
CONDITION_MALUS = dict(
    nostatus=0.0, brn=-1.0, psn=-1.0, tox=-2.0, par=-1.5, slp=-2.5, frz=-2.5
)

# Type Effectiveness Factors
TYPE_INTERACTIONS = {
    'Normal': {'Rock': 0.5, 'Ghost': 0.0},
    'Fire': {'Grass': 2.0, 'Ice': 2.0, 'Water': 0.5, 'Rock': 0.5, 'Bug': 2.0, 'Steel': 2.0, 'Fairy': 1.0},
    'Water': {'Fire': 2.0, 'Ground': 2.0, 'Rock': 2.0, 'Grass': 0.5, 'Dragon': 0.5},
    'Grass': {'Water': 2.0, 'Ground': 2.0, 'Rock': 2.0, 'Fire': 0.5, 'Flying': 0.5, 'Bug': 0.5, 'Poison': 0.5, 'Dragon': 0.5, 'Steel': 0.5},
    'Electric': {'Water': 2.0, 'Flying': 2.0, 'Ground': 0.0, 'Grass': 0.5, 'Electric': 0.5, 'Dragon': 0.5},
    'Ice': {'Grass': 2.0, 'Ground': 2.0, 'Flying': 2.0, 'Dragon': 2.0, 'Fire': 0.5, 'Water': 0.5, 'Ice': 0.5, 'Steel': 0.5},
    'Fighting': {'Normal': 2.0, 'Rock': 2.0, 'Steel': 2.0, 'Ice': 2.0, 'Dark': 2.0, 'Flying': 0.5, 'Poison': 0.5, 'Bug': 0.5, 'Psychic': 0.5, 'Fairy': 0.5, 'Ghost': 0.0},
    'Poison': {'Grass': 2.0, 'Fairy': 2.0, 'Ground': 0.5, 'Rock': 0.5, 'Ghost': 0.5, 'Steel': 0.0},
    'Ground': {'Fire': 2.0, 'Electric': 2.0, 'Poison': 2.0, 'Rock': 2.0, 'Steel': 2.0, 'Grass': 0.5, 'Bug': 0.5, 'Flying': 0.0},
    'Flying': {'Grass': 2.0, 'Fighting': 2.0, 'Bug': 2.0, 'Rock': 0.5, 'Steel': 0.5, 'Electric': 0.5},
    'Psychic': {'Fighting': 2.0, 'Poison': 2.0, 'Steel': 0.5, 'Psychic': 0.5, 'Dark': 0.0},
    'Bug': {'Grass': 2.0, 'Psychic': 2.0, 'Dark': 2.0, 'Fire': 0.5, 'Fighting': 0.5, 'Flying': 0.5, 'Ghost': 0.5, 'Steel': 0.5, 'Fairy': 0.5},
    'Rock': {'Fire': 2.0, 'Ice': 2.0, 'Flying': 2.0, 'Bug': 2.0, 'Fighting': 0.5, 'Ground': 0.5, 'Steel': 0.5},
    'Ghost': {'Psychic': 2.0, 'Ghost': 2.0, 'Normal': 0.0, 'Dark': 0.5},
    'Dragon': {'Dragon': 2.0, 'Steel': 0.5, 'Fairy': 0.0},
    'Steel': {'Ice': 2.0, 'Rock': 2.0, 'Fairy': 2.0, 'Fire': 0.5, 'Water': 0.5, 'Electric': 0.5, 'Steel': 0.5},
    'Dark': {'Psychic': 2.0, 'Ghost': 2.0, 'Fighting': 0.5, 'Dark': 0.5, 'Fairy': 0.5},
    'Fairy': {'Fighting': 2.0, 'Dragon': 2.0, 'Dark': 2.0, 'Fire': 0.5, 'Poison': 0.5, 'Steel': 0.5}
}

# Hyperparameter Search Settings
RF_SEARCH_SPACE = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [1, 3],
    'max_features': [0.3, 0.5, 0.7]
}

XGB_SEARCH_SPACE = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'reg_alpha': [0.1, 0.5, 1.0],
    'gamma': [0, 0.1, 0.5, 1.0]
}

HGBT_SEARCH_SPACE = {
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [10, 20, 30]
}

# Level 1 Models (Stacking)
if HAS_XGB:
    TIER1_MODELS_DEF = {
        'XGBoost_Meta': XGBC(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            reg_alpha=1.0,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
    }
else:
    # No XGBoost available: start with an empty dict, will add Logistic_Meta later in main
    TIER1_MODELS_DEF = {}
