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
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBClassifier = None
    print("[WARN] xgboost non disponibile: il modello XGB verrà saltato.")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

PROJECT_IDENTIFIER = 'fds-pokemon-battles-prediction-2025'
RESOURCE_PATH = os.path.join('../input', PROJECT_IDENTIFIER)
train_source = os.path.join(RESOURCE_PATH, 'train.jsonl')
test_source = os.path.join(RESOURCE_PATH, 'test.jsonl')
