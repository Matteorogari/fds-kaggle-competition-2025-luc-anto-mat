
# --- Base imports and configuration ---
import os

# Safely handle optional xgboost dependency
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBClassifier = None
    print("[WARN] xgboost non disponibile: il modello XGB verrà saltato.")

# --- Project initialization ---
PROJECT_IDENTIFIER = 'fds-pokemon-battles-prediction-2025'

# Handle Kaggle vs local paths safely
if os.path.exists("/kaggle/input"):
    BASE_INPUT_DIR = "/kaggle/input"
else:
    BASE_INPUT_DIR = "../input"

RESOURCE_PATH = os.path.join(BASE_INPUT_DIR, PROJECT_IDENTIFIER)
train_source = os.path.join(RESOURCE_PATH, 'train.jsonl')
test_source = os.path.join(RESOURCE_PATH, 'test.jsonl')


def init_project_paths():
       return train_source, test_source
