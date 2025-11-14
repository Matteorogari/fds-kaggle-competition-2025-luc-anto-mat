import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from .constants import C_VECTOR, RANDOM_SEARCH_COUNT # Import the constants

def define_linear_grid_search(X_subset):
    #LR pipeline with standardization and GridSearchCV (we use C_VECTOR)
    pipeline = Pipeline([
        ('feature_scaler', StandardScaler()),
        ('base_classifier', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
    ])
    param_set = {'base_classifier__C': C_VECTOR}
    return GridSearchCV(pipeline, param_set, cv=3, scoring='neg_log_loss', refit=True, n_jobs=-1)

def define_non_linear_random_search(model_inst, param_distribution):
    #RandomizedSearchCV for nonlinear models
    return RandomizedSearchCV(
        model_inst,
        param_distributions=param_distribution,
        n_iter=RANDOM_SEARCH_COUNT,
        cv=3,
        scoring='neg_log_loss',
        random_state=42,
        refit=True,
        n_jobs=-1
    )

def estimate_optimal_cutoff(y_true, y_pred_proba):
    #the probability threshold that maximizes accuracy
    best_acc, opt_thresh = 0, 0.5
    for cutoff_value in np.linspace(0.4, 0.6, 101):
        acc = accuracy_score(y_true, (y_pred_proba >= cutoff_value).astype(int))
        if acc > best_acc: best_acc, opt_thresh = acc, cutoff_value
    return opt_thresh, best_acc

def analyze_meta_model_weights(best_tier1_tag, winning_model, X_tier1_test):
    #Printing the importance/weights of features (Tier-0 predictions) in Stacking
    print("\n" + "="*50)
    print("=== ANALISI DEI PESI DEL META-MODELLO VINCITORE ===")
    print("===================================================")
    if best_tier1_tag == 'XGBoost_Meta':
        importance_df = pd.DataFrame({
            'Modello Base (Feature)': X_tier1_test.columns,
            'Importanza (Peso)': winning_model.feature_importances_
        }).sort_values(by='Importanza (Peso)', ascending=False)
        print("Importanza delle Predizioni (Features) Nello Stacking:\n")
        print(importance_df.to_string(index=False, float_format='%.4f'))
    else: # Logistic_Meta
        coefs = winning_model.coef_[0]
        importance_df = pd.DataFrame({
            'Modello Base (Feature)': X_tier1_test.columns,
            'Peso Lineare (Coef)': coefs
        }).sort_values(by='Peso Lineare (Coef)', key=np.abs, ascending=False)
        print(f"Pesi Lineari delle Predizioni (Features) Nello Stacking:\n")
        print(importance_df.to_string(index=False, float_format='%.4f'))
