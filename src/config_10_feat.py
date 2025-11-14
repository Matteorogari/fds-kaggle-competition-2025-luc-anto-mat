#---Base imports and configuration---
import os
import json
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

# Safely handle optional xgboost dependency
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBClassifier = None
    print("[WARN] xgboost non disponibile: il modello XGB verrà saltato.")

#---Project initialization---
PROJECT_IDENTIFIER = 'fds-pokemon-battles-prediction-2025'

# *** FIX: gestisci correttamente la cartella input su Kaggle ***
if os.path.exists("/kaggle/input"):
    BASE_INPUT_DIR = "/kaggle/input"
else:
    # fallback per esecuzioni locali
    BASE_INPUT_DIR = "../input"

RESOURCE_PATH = os.path.join(BASE_INPUT_DIR, PROJECT_IDENTIFIER)
train_source = os.path.join(RESOURCE_PATH, 'train.jsonl')
test_source = os.path.join(RESOURCE_PATH, 'test.jsonl')


# *** FIX: funzione mancante per compatibilità con main_10_feat ***
def init_project_paths():
    """
    Restituisce i path completi di train e test.
    Funzione di utilità mantenuta per compatibilità col codice esistente.
    """
    return train_source, test_source
