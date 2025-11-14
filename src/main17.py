# ==========================================================
# 1. SETUP and IMPORT
# ==========================================================
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from constants17 import (
    TRAIN_SOURCE,
    TEST_SOURCE,
    NUM_FOLDS,
    RF_SEARCH_SPACE,
    XGB_SEARCH_SPACE,
    HGBT_SEARCH_SPACE,
    TIER1_MODELS_DEF,
    HAS_XGB,
)
from feature_engineering17 import (
    data_ingestion,
    compute_final_features,
    display_frame_preview,
)
from modelling_utils17 import (
    define_linear_grid_search,
    define_non_linear_random_search,
    estimate_optimal_cutoff,
    analyze_meta_model_weights,
)
from sklearn.ensemble import RandomForestClassifier as RFC, HistGradientBoostingClassifier as HGBC
from xgboost import XGBClassifier as XGBC  # safe because HAS_XGB checked before use


def generate_submission(output_path: str):

    # ==========================================================
    # 2. DATA LOADING AND FEATURE ENGINEERING
    # ==========================================================

    # Names of the subset features
    SUBSET_FEATURES = [
        'hp_avg_delta', 'survivor_count_delta', 'team_status_net_delta',
        'total_weighted_power_delta', 'switch_activity_log_ratio',
        'stat_dom_diff_metric', 'lead_hp_frac_diff', 'lead_status_value_diff',
        'lead_stat_power_diff'
    ]

    train_raw = data_ingestion(TRAIN_SOURCE, file_context="training")
    test_raw = data_ingestion(TEST_SOURCE, file_context="test_set")

    train_features_df = compute_final_features(train_raw)
    test_features_df = compute_final_features(test_raw)

    # Data preparation for the model
    ALL_FEATURES = [
        col for col in train_features_df.columns
        if col not in ['battle_id', 'player_won']
    ]
    X_full = train_features_df[ALL_FEATURES]
    X_lite = train_features_df[SUBSET_FEATURES]
    y_target = train_features_df['player_won']
    N_TRAIN_SAMPLES = X_full.shape[0]

    X_test_full = test_features_df[ALL_FEATURES].copy()
    X_test_lite = test_features_df[SUBSET_FEATURES].copy()

    # Definition of Level 0 models
    TIER0_MODELS = {
        'LR_Full': (
            X_full,
            X_test_full,
            define_linear_grid_search(X_full),
        ),
        'LR_Lite': (
            X_lite,
            X_test_lite,
            define_linear_grid_search(X_lite),
        ),
        'RF_Model': (
            X_full,
            X_test_full,
            define_non_linear_random_search(
                RFC(random_state=42),
                RF_SEARCH_SPACE,
            ),
        ),
        'HGBT_Model': (
            X_full,
            X_test_full,
            define_non_linear_random_search(
                HGBC(random_state=42),
                HGBT_SEARCH_SPACE,
            ),
        ),
    }

    # Add XGB base model only if xgboost is available
    if HAS_XGB:
        TIER0_MODELS['XGB_Model'] = (
            X_full,
            X_test_full,
            define_non_linear_random_search(
                XGBC(
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                ),
                XGB_SEARCH_SPACE,
            ),
        )

    # ==========================================================
    # 3. CROSS-VALIDATION OOF (Out-of-Fold)
    # ==========================================================
    OOF_PROBABILITIES = pd.DataFrame(index=X_full.index)
    skf_splitter = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=42,
    )
    test_set_oof_preds = {}

    print("\n" + "=" * 50)
    print("=== CALIBRAZIONE TIER-0 MODELLI (Ricerca + OOF) ===")
    print("=" * 50)

    for model_tag, (X_source, X_test_source, model_pipe) in TIER0_MODELS.items():
        print(f"\n--- Modello: {model_tag} ---")
        oof_array = np.zeros(N_TRAIN_SAMPLES)
        test_preds_sum = np.zeros(X_test_source.shape[0])

        for fold_id, (train_indices, val_indices) in enumerate(
            skf_splitter.split(X_source, y_target)
        ):
            X_train_fold = X_source.iloc[train_indices]
            X_val_fold = X_source.iloc[val_indices]
            y_train_fold = y_target.iloc[train_indices]

            model_pipe.fit(X_train_fold, y_train_fold)
            oof_array[val_indices] = model_pipe.predict_proba(X_val_fold)[:, 1]
            test_preds_sum += (
                model_pipe.predict_proba(X_test_source)[:, 1] / NUM_FOLDS
            )

        OOF_PROBABILITIES[model_tag] = oof_array
        test_set_oof_preds[model_tag] = test_preds_sum

        oof_logloss_val = log_loss(y_target, oof_array)
        print(f"Log_loss OOF {model_tag}: {oof_logloss_val:.6f}")

    # ==========================================================
    # 4. STACKING (META-MODEL) AND FINAL RESULT
    # ==========================================================
    # Always add a Logistic meta-model
    TIER1_MODELS_DEF['Logistic_Meta'] = LogisticRegression(
        C=0.5,
        solver='liblinear',
        random_state=42,
    )

    X_tier1_test = pd.DataFrame(test_set_oof_preds)

    best_tier1_tag = ""
    best_tier1_acc = 0.0
    final_optimal_cutoff = 0.0
    final_tier1_test_probs = None

    print("\n" + "=" * 50)
    print("=== FASE DI STACKING: COSTRUZIONE META-MODELLO ===")
    print("=" * 50)

    for tier1_tag, tier1_model in TIER1_MODELS_DEF.items():
        tier1_model.fit(OOF_PROBABILITIES, y_target)
        tier1_oof_probs = tier1_model.predict_proba(OOF_PROBABILITIES)[:, 1]
        opt_cutoff, opt_acc = estimate_optimal_cutoff(
            y_target,
            tier1_oof_probs,
        )

        if opt_acc > best_tier1_acc:
            best_tier1_acc = opt_acc
            best_tier1_tag = tier1_tag
            final_optimal_cutoff = opt_cutoff
            final_tier1_test_probs = tier1_model.predict_proba(
                X_tier1_test
            )[:, 1]

    # Weight Analysis
    winning_model = TIER1_MODELS_DEF[best_tier1_tag]
    analyze_meta_model_weights(
        best_tier1_tag,
        winning_model,
        X_tier1_test,
    )

    # Final Result and Submission
    final_test_predictions = (
        final_tier1_test_probs >= final_optimal_cutoff
    ).astype(int)

    submission_df = pd.DataFrame({
        'battle_id': test_features_df['battle_id'],
        'player_won': final_test_predictions,
    })

    submission_df.to_csv(output_path, index=False)

    print("\n" + "=" * 30)
    print("==============================")
    print(f"META-MODELLO PIÙ EFFICACE: {best_tier1_tag}")
    print(f"ACCURATEZZA MAX OTTENUTA (CV): {best_tier1_acc:.6f}")
    print(f"Soglia di classificazione finale: {final_optimal_cutoff:.3f}")
    print("==============================")
    print(f"\nFile di submission generato: {output_path}")
    display_frame_preview(submission_df)

    return submission_df


if __name__ == "__main__":
    generate_submission("submission.csv")
