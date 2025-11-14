def generate_submission(output_path: str):
    
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

    X_meta_test_final = np.vstack(FINAL_TEST_PROB_LIST).T

    final_meta_model = META_RESULTS[best_meta_tag]["estimator"]
    final_meta_model.fit(X_meta, y_target)

    test_meta_proba = final_meta_model.predict_proba(X_meta_test_final)[:, 1]
    t_final = best_meta["best_threshold"]
    test_predictions = (test_meta_proba >= t_final).astype(int)

    submission_df = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': test_predictions
    })

    
    submission_df.to_csv(output_path, index=False)

    print("\n'submission.csv' file created successfully!")
    print("Prime righe della submission:")
    display(submission_df.head())



if __name__ == "__main__":
    generate_submission("submission.csv")
