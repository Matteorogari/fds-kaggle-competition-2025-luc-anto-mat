#---Import e configurazione di base---
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

# Gestisce la disponibilità del modello XGBoost in modo robusto
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBClassifier = None
    print("[WARN] xgboost non disponibile: il modello XGB verrà saltato.")


def init_project_paths():
    """
    Inizializza i percorsi principali del progetto e i file JSONL di train e test.

    Restituisce:
        train_source (str): percorso del file di training.
        test_source (str): percorso del file di test.
    """
    #---Inizializzazione del progetto---
    PROJECT_IDENTIFIER = 'fds-pokemon-battles-prediction-2025'
    RESOURCE_PATH = os.path.join('../input', PROJECT_IDENTIFIER)
    train_source = os.path.join(RESOURCE_PATH, 'train.jsonl')
    test_source = os.path.join(RESOURCE_PATH, 'test.jsonl')
    return train_source, test_source
