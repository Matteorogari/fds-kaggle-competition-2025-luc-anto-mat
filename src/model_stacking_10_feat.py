def train_stacking_pipeline(train_df, test_df, BASE_FEATURES_10):
    #---CV setup, feature lists and base model definitions---
    RANDOM_STATE = 42
    NUM_FOLDS = 5
    K_LIST = [7, 15, 20, 25, 30]

    # Builds the full feature matrix and target vector
    ALL_FEATURES = BASE_FEATURES_10.copy()
    X_full = train_df[ALL_FEATURES].values
    y_target = train_df['player_won'].values.astype(int)
    X_test_full = test_df[ALL_FEATURES].values

    # Defines a compact subset of features for the LR_Lite model
    SUBSET_FEATURES = [
        'p1_mean_pc_hp', 'p2_mean_pc_hp',
        'p1_surviving_pokemon',
        'p1_status_score', 'p2_status_score',
        'hp_sum_diff'
    ]
    SUBSET_FEATURES = [f for f in SUBSET_FEATURES if f in train_df.columns]

    print("\nNumero feature totali:", len(ALL_FEATURES))
    print("Feature LR_LITE (SUBSET_FEATURES):", SUBSET_FEATURES)

    # Defines the Tier-0 model space with estimators and hyperparameter grids
    TIER0_MODELS = {
        "LR_Full": {
            "features": ALL_FEATURES,
            "estimator": Pipeline(steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000,
                    solver='lbfgs',
                    penalty='l2',
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
                    penalty='l2',
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

    # Adds the XGBoost base model if the library is available
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

    # Sets up the stratified cross-validation strategy
    cv_splitter = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Lists to store OOF and test predictions for base models
    OOF_PROB_LIST = []
    TEST_PROB_LIST = []
    BASE_MODEL_TAGS = []

    print("\n=== TRAINING BASE MODELS (con GridSearchCV + OOF + Permutation Importance) ===")

    # Trains, tunes and evaluates OOF performance for each base model
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
            cv=cv_splitter,
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_model, y_target)

        best_estimator = grid.best_estimator_
        print("Best params:", grid.best_params_)
        print("CV log_loss (GridSearch best):", -grid.best_score_)

        # Computes OOF probabilities via cross_val_predict
        oof_proba = cross_val_predict(
            best_estimator,
            X_model,
            y_target,
            cv=cv_splitter,
            method='predict_proba',
            n_jobs=-1
        )[:, 1]

        oof_ll = log_loss(y_target, oof_proba)
        oof_acc_05 = accuracy_score(y_target, (oof_proba >= 0.5).astype(int))

        print(f"OOF log_loss {model_tag}: {oof_ll:.6f}")
        print(f"OOF accuracy (soglia 0.5) {model_tag}: {oof_acc_05:.6f}")

        # Computes permutation feature importance where applicable
        try:
            if model_tag != "KNN_Model":
                pi_res = permutation_importance(
                    best_estimator, X_model, y_target,
                    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
                )
                pi_df = pd.DataFrame({
                    "feature": feat_cols,
                    "importance": pi_res.importances_mean
                }).sort_values(by="importance", ascending=False)
                print(f"\nPermutation Importance per {model_tag}:")
                display(pi_df)
            else:
                print("\n[SKIP] Permutation importance saltata per KNN_Model.")
        except Exception as e:
            print(f"[WARN] permutation_importance fallita per {model_tag}: {e}")

        # Stores OOF probabilities and test predictions for the current model
        OOF_PROB_LIST.append(oof_proba)
        TEST_PROB_LIST.append(best_estimator.predict_proba(X_test_model)[:, 1])
        BASE_MODEL_TAGS.append(model_tag)

    print("\n=== COSTRUZIONE META-MODEL (STACKING) ===")

    # Builds meta matrices using base model predictions
    X_meta = np.vstack(OOF_PROB_LIST).T
    X_meta_test = np.vstack(TEST_PROB_LIST).T

    print("Base models usati nello stacking:", BASE_MODEL_TAGS)
    print("Forma X_meta:", X_meta.shape)

    # Defines the dictionary of level-1 meta models
    TIER1_MODELS = {}

    TIER1_MODELS["Logistic_Meta"] = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver='lbfgs',
        penalty='l2',
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

    # Initializes the structure to collect metrics and optimal thresholds
    META_RESULTS = {}
    threshold_grid = np.linspace(0.30, 0.70, 81)

    # Evaluates each meta model on OOF data and optimizes the classification threshold
    for meta_tag, meta_estimator in TIER1_MODELS.items():
        print(f"\n--- META MODEL: {meta_tag} ---")

        oof_meta_proba = cross_val_predict(
            meta_estimator,
            X_meta,
            y_target,
            cv=cv_splitter,
            method='predict_proba',
            n_jobs=-1
        )[:, 1]

        meta_ll = log_loss(y_target, oof_meta_proba)
        acc_05 = accuracy_score(y_target, (oof_meta_proba >= 0.5).astype(int))
        print(f"OOF log_loss (meta, soglia libera): {meta_ll:.6f}")
        print(f"OOF accuracy (meta, soglia 0.5):    {acc_05:.6f}")

        # Applies nested validation to estimate an optimal threshold for each fold
        nested_preds = np.zeros_like(y_target, dtype=int)
        for tr_idx, va_idx in cv_splitter.split(oof_meta_proba, y_target):
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

        # Estimates a global optimal threshold on the full OOF distribution
        accs_per_t = [
            accuracy_score(y_target, (oof_meta_proba >= t).astype(int)) for t in threshold_grid
        ]
        best_t_global = threshold_grid[int(np.argmax(accs_per_t))]
        print(f"Miglior soglia globale stimata: {best_t_global:.3f}")

        META_RESULTS[meta_tag] = {
            "estimator": meta_estimator,
            "oof_proba": oof_meta_proba,
            "logloss": meta_ll,
            "acc_05": acc_05,
            "acc_nested": acc_nested,
            "best_threshold": float(best_t_global)
        }

    # Identifies the best meta model based on nested accuracy
    best_meta_tag = max(META_RESULTS, key=lambda k: META_RESULTS[k]["acc_nested"])
    best_meta = META_RESULTS[best_meta_tag]

    print("\n==============================")
    print("MIGLIOR META-MODEL:", best_meta_tag)
    print(f"LOG_LOSS FINALE (meta):      {best_meta['logloss']:.6f}")
    print(f"ACC_EFFETTIVA FINALE (CV):   {best_meta['acc_nested']:.6f}")
    print(f"Soglia finale usata (global): {best_meta['best_threshold']:.3f}")
    print("==============================\n")

    return (
        TIER0_MODELS,
        META_RESULTS,
        best_meta_tag,
        best_meta,
        X_meta,
        X_meta_test,
        y_target,
        cv_splitter
    )
