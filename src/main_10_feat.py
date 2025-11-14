###########################################################################################
# ---Main---

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from config_10_feat import init_project_paths, train_source, test_source
from features_10_feat import build_feature_tables_and_spearman
from model_stacking_10_feat import train_stacking_pipeline


def generate_submission(output_path: str):
    # Initialize training and test file paths
    # *** opzionale ma coerente: usiamo la funzione di init per ottenere i path ***
    train_src, test_src = init_project_paths()
    # in alternativa, avresti potuto lasciare:
    # train_src, test_src = train_source, test_source

    # Build feature tables and analysis outputs (Spearman and partial Spearman)
    (
        train_df,
        test_df,
        BASE_FEATURES_10,
        spearman_matrix,
        partial_df,
    ) = build_feature_tables_and_spearman(train_src, test_src)

    # Train Tier-0 base models and Tier-1 meta-models (stacking)
    (
        TIER0_MODELS,
        META_RESULTS,
        best_meta_tag,
        best_meta,
        X_meta,
        X_meta_test,
        y_target,
        cv_splitter,
    ) = train_stacking_pipeline(train_df, test_df, BASE_FEATURES_10)

    # Final refit of base models and prediction on the test set
    FINAL_TEST_PROB_LIST = []

    for model_tag, cfg in TIER0_MODELS.items():
        feat_cols = cfg["features"]
        if len(feat_cols) == 0:
            continue

        X_model = train_df[feat_cols].values
        X_test_model = test_df[feat_cols].values

        print(f"Refit finale base model: {model_tag}")

        grid = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["param_grid"],
            cv=cv_splitter,
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_model, y_target)
        best_estimator = grid.best_estimator_
        FINAL_TEST_PROB_LIST.append(best_estimator.predict_proba(X_test_model)[:, 1])

    # Build the final meta-feature matrix on the test set
    X_meta_test_final = np.vstack(FINAL_TEST_PROB_LIST).T

    # Final refit of the best meta-model and prediction on the test set
    final_meta_model = META_RESULTS[best_meta_tag]["estimator"]
    final_meta_model.fit(X_meta, y_target)

    test_meta_proba = final_meta_model.predict_proba(X_meta_test_final)[:, 1]
    t_final = best_meta["best_threshold"]
    test_predictions = (test_meta_proba >= t_final).astype(int)

    # Build and save the submission file
    submission_df = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': test_predictions
    })

    submission_df.to_csv(output_path, index=False)

    print(f"\nSubmission file created successfully at: {output_path}")
    print("First rows of the submission:")
    print(submission_df.head())

    return submission_df


if __name__ == "__main__":
    generate_submission("submission.csv")
