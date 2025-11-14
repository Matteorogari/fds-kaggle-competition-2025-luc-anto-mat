#---Features engineering---
import json 
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
#---Feature matrix construction and preview---
print("Processing training data...")
train_df = compute_final_features(train_raw)

print("\nProcessing test data...")
test_df = compute_final_features(test_raw)

print("\nTraining features preview:")
display(train_df.head())

#---ML parameter setup and feature selection---
RANDOM_STATE = 42
NUM_FOLDS = 5
K_LIST = [7, 15, 20, 25, 30]

ALL_FEATURES = [c for c in train_df.columns if c not in ['battle_id', 'player_won']]
X_full = train_df[ALL_FEATURES].values
y_target = train_df['player_won'].values.astype(int)
X_test_full = test_df[ALL_FEATURES].values

SUBSET_FEATURES = [
    'p1_mean_pc_hp', 'p2_mean_pc_hp',
    'p1_surviving_pokemon', 'p2_surviving_pokemon',
    'p1_status_score', 'p2_status_score',
    'hp_sum_diff'
]
SUBSET_FEATURES = [f for f in SUBSET_FEATURES if f in train_df.columns]

print("\nNumero feature totali:", len(ALL_FEATURES))
print("Feature LR_LITE (SUBSET_FEATURES):", SUBSET_FEATURES)

#---Base model definitions (Tier 0)---
TIER0_MODELS = {
    "LR_Full": {
        "features": ALL_FEATURES,
        "estimator": Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                solver='lbfgs',
                random_state=RANDOM_STATE
            ))
        ]),
        "param_grid": {
            "clf__C": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        }
    },
    "LR_Lite": {
        "features": SUBSET_FEATURES,
        "estimator": Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                solver='lbfgs',
                random_state=RANDOM_STATE
            ))
        ]),
        "param_grid": {
            "clf__C": [0.1, 0.5, 1.0, 2.0]
        }
    },
    "RF_Model": {
        "features": ALL_FEATURES,
        "estimator": RandomForestClassifier(
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "param_grid": {
            "n_estimators": [200, 400],
            "max_depth": [5, 8],
            "min_samples_leaf": [1, 3]
        }
    },
    "HGBT_Model": {
        "features": ALL_FEATURES,
        "estimator": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "param_grid": {
            "learning_rate": [0.03, 0.06],
            "max_depth": [3, 4],
            "max_leaf_nodes": [31, 63],
            "min_samples_leaf": [20, 40]
        }
    },
    "KNN_Model": {
        "features": ALL_FEATURES,
        "estimator": Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ]),
        "param_grid": {
            "clf__n_neighbors": K_LIST,
            "clf__weights": ["uniform", "distance"]
        }
    }
}

#---Optional: XGBoost base model---
if HAS_XGB:
    TIER0_MODELS["XGB_Model"] = {
        "features": ALL_FEATURES,
        "estimator": XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist',
            random_state=RANDOM_STATE
        ),
        "param_grid": {
            "n_estimators": [200, 400],
            "max_depth": [3, 4],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.7, 0.9],
            "reg_lambda": [1.0, 3.0],
            "reg_alpha": [0.0, 0.5]
        }
    }

#---Cross-validation and base model training---
skf_splitter = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

OOF_PROB_LIST = []
TEST_PROB_LIST = []
BASE_MODEL_TAGS = []

print("\n=== TRAINING BASE MODELS (con GridSearchCV + OOF) ===")

for model_tag, cfg in TIER0_MODELS.items():
    feat_cols = cfg["features"]
    if len(feat_cols) == 0:
        print(f"[SKIP] {model_tag}: nessuna feature valida.")
        continue

    X_model = train_df[feat_cols].values
    X_test_model = test_df[feat_cols].values

    print(f"\n--- {model_tag} ---")
    print(f"Numero feature usate: {len(feat_cols)}")

    grid = GridSearchCV(
        estimator=cfg["estimator"],
        param_grid=cfg["param_grid"],
        cv=skf_splitter,
        scoring='neg_log_loss',
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_model, y_target)

    best_estimator = grid.best_estimator_
    print("Best params:", grid.best_params_)
    print("CV log_loss (GridSearch best):", -grid.best_score_)

    oof_proba = cross_val_predict(
        best_estimator,
        X_model,
        y_target,
        cv=skf_splitter,
        method='predict_proba',
        n_jobs=-1
    )[:, 1]

    oof_ll = log_loss(y_target, oof_proba)
    oof_acc_05 = accuracy_score(y_target, (oof_proba >= 0.5).astype(int))

    print(f"OOF log_loss {model_tag}: {oof_ll:.6f}")
    print(f"OOF accuracy (soglia 0.5) {model_tag}: {oof_acc_05:.6f}")

    OOF_PROB_LIST.append(oof_proba)
    TEST_PROB_LIST.append(best_estimator.predict_proba(X_test_model)[:, 1])
    BASE_MODEL_TAGS.append(model_tag)

#---Meta-model construction (STACKING, Tier 1)---
print("\n=== COSTRUZIONE META-MODEL (STACKING) ===")

X_meta = np.vstack(OOF_PROB_LIST).T
X_meta_test = np.vstack(TEST_PROB_LIST).T

print("Base models usati nello stacking:", BASE_MODEL_TAGS)
print("Forma X_meta:", X_meta.shape)

TIER1_MODELS = {}

TIER1_MODELS["Logistic_Meta"] = LogisticRegression(
    C=1.0,
    max_iter=2000,
    solver='lbfgs',
    random_state=RANDOM_STATE
)

if HAS_XGB:
    TIER1_MODELS["XGBoost_Meta"] = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        tree_method='hist',
        n_estimators=300,
        max_depth=3,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        random_state=RANDOM_STATE
    )

#---Threshold search and meta-model evaluation---
META_RESULTS = {}

threshold_grid = np.linspace(0.30, 0.70, 81)

for meta_tag, meta_model in TIER1_MODELS.items():
    print(f"\n--- META MODEL: {meta_tag} ---")

    oof_meta_proba = cross_val_predict(
        meta_model,
        X_meta,
        y_target,
        cv=skf_splitter,
        method='predict_proba',
        n_jobs=-1
    )[:, 1]

    meta_ll = log_loss(y_target, oof_meta_proba)
    acc_05 = accuracy_score(y_target, (oof_meta_proba >= 0.5).astype(int))
    print(f"OOF log_loss (meta, soglia libera): {meta_ll:.6f}")
    print(f"OOF accuracy (meta, soglia 0.5):    {acc_05:.6f}")

    nested_preds = np.zeros_like(y_target, dtype=int)

    for tr_idx, va_idx in skf_splitter.split(oof_meta_proba, y_target):
        y_tr = y_target[tr_idx]
        p_tr = oof_meta_proba[tr_idx]
        p_va = oof_meta_proba[va_idx]

        best_t = 0.5
        best_acc = 0.0

        for t in threshold_grid:
            acc = accuracy_score(y_tr, (p_tr >= t).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_t = t

        nested_preds[va_idx] = (p_va >= best_t).astype(int)

    acc_nested = accuracy_score(y_target, nested_preds)
    print(f"ACC_EFFETTIVA (nested threshold) {meta_tag}: {acc_nested:.6f}")

    accs_per_t = [
        accuracy_score(y_target, (oof_meta_proba >= t).astype(int)) for t in threshold_grid
    ]
    best_t_global = threshold_grid[int(np.argmax(accs_per_t))]
    print(f"Miglior soglia globale stimata: {best_t_global:.3f}")

    META_RESULTS[meta_tag] = {
        "estimator": meta_model,
        "oof_proba": oof_meta_proba,
        "logloss": meta_ll,
        "acc_05": acc_05,
        "acc_nested": acc_nested,
        "best_threshold": float(best_t_global)
    }

best_meta_tag = max(META_RESULTS, key=lambda k: META_RESULTS[k]["acc_nested"])
best_meta = META_RESULTS[best_meta_tag]

print("\n==============================")
print("MIGLIOR META-MODEL:", best_meta_tag)
print(f"LOG_LOSS FINALE (meta):      {best_meta['logloss']:.6f}")
print(f"ACC_EFFETTIVA FINALE (CV):   {best_meta['acc_nested']:.6f}")
print(f"Soglia finale usata (global): {best_meta['best_threshold']:.3f}")
print("==============================\n")
::contentReference[oaicite:0]{index=0}
