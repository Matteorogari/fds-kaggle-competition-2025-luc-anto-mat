#---Feature engineering functions---
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
def build_pokemon_stat_registry(battle_records: list[dict]):
    # Builds a registry of base stats for Pokémon seen in the data
    stat_registry = {}
    for battle in battle_records:
        pkmn_list = battle.get('p1_team_details', [])
        for pokemon in pkmn_list:
            mon_name = pokemon.get('name')
            mon_level = pokemon.get('level')
            if mon_name and mon_name not in stat_registry and mon_level == 100:
                base_stats = {k: v for k, v in pokemon.items() if k.startswith('base_')}
                base_stats['types'] = pokemon.get('types', ['notype', 'notype'])
                if base_stats:
                    stat_registry[mon_name] = base_stats
    return stat_registry


def extract_battle_summary(battle):
    # Summarizes the dynamic battle state (HP, status and switches)
    p1_team_state = {
        pokemon.get('name', f'p1_unknown_{i}'): {
            'hp': 1.00,
            'status': 'nostatus'
        } for i, pokemon in enumerate(battle.get('p1_team_details', []))
    }
    p2_team_state = {}
    p2_team_state[battle.get('p2_lead_details', {}).get('name')] = {
        'hp': 1.00,
        'status': 'nostatus'
    }

    for turn in battle.get('battle_timeline', []):
        p1_team_state[turn.get('p1_pokemon_state', {}).get('name')] = {
            'hp': turn.get('p1_pokemon_state', {}).get('hp_pct'),
            'status': turn.get('p1_pokemon_state', {}).get('status')
        }
        p2_team_state[turn.get('p2_pokemon_state', {}).get('name')] = {
            'hp': turn.get('p2_pokemon_state', {}).get('hp_pct'),
            'status': turn.get('p2_pokemon_state', {}).get('status')
        }

    p1_swap_count = 0
    p2_swap_count = 0
    for turn in battle.get('battle_timeline', []):
        p1_current_pk = turn.get('p1_pokemon_state', {}).get('name')
        p2_current_pk = turn.get('p2_pokemon_state', {}).get('name')
        if turn != battle.get('battle_timeline', [])[0]:
            if p1_prev_pk != p1_current_pk:
                p1_swap_count += 1
            if p2_prev_pk != p2_current_pk:
                p2_swap_count += 1
        else:
            p1_prev_pk = p1_current_pk
            p2_prev_pk = p2_current_pk

    p1_effects = len(battle.get('battle_timeline', [])[-1].get('p1_pokemon_state', {}).get('effects', [])) * 0.4
    p2_effects = len(battle.get('battle_timeline', [])[-1].get('p2_pokemon_state', {}).get('effects', [])) * 0.4

    for i in range(len(p2_team_state), 6):
        p2_team_state[f'p2_unknown_{i}'] = {
            'hp': 1.00,
            'status': 'nostatus'
        }

    return p1_swap_count, p1_effects, p1_team_state, p2_swap_count, p2_effects, p2_team_state


def compute_base_stat_differences(p1_team_state, p2_team_state, stat_registry):
    # Computes aggregated differences of base stats between the two teams
    p1_total_speed = 0
    p2_total_speed = 0
    p1_total_attack = 0
    p2_total_attack = 0
    p1_total_sp_attack = 0
    p2_total_sp_attack = 0
    p1_total_sp_defense = 0
    p2_total_sp_defense = 0
    p1_total_hp = 0
    p2_total_hp = 0

    for pokemon in p1_team_state.keys():
        if pokemon in stat_registry:
            p1_total_speed += stat_registry[pokemon]['base_spe']
            p1_total_attack += stat_registry[pokemon]['base_atk']
            p1_total_sp_attack += stat_registry[pokemon]['base_spa']
            p1_total_sp_defense += stat_registry[pokemon]['base_spd']
            p1_total_hp += stat_registry[pokemon]['base_hp']

    for pokemon in p2_team_state.keys():
        if pokemon in stat_registry:
            p2_total_speed += stat_registry[pokemon]['base_spe']
            p2_total_attack += stat_registry[pokemon]['base_atk']
            p2_total_sp_attack += stat_registry[pokemon]['base_spa']
            p2_total_sp_defense += stat_registry[pokemon]['base_spd']
            p2_total_hp += stat_registry[pokemon]['base_hp']

    speed_sum_diff = p1_total_speed - p2_total_speed
    attack_sum_diff = p1_total_attack - p2_total_attack
    sp_attack_sum_diff = p1_total_sp_attack - p2_total_sp_attack
    sp_defense_sum_diff = p1_total_sp_defense - p2_total_sp_defense
    hp_sum_diff = p1_total_hp - p2_total_hp

    return speed_sum_diff, attack_sum_diff, sp_attack_sum_diff, sp_defense_sum_diff, hp_sum_diff


def compute_final_features(source_data: list[dict]) -> pd.DataFrame:
    # Converts raw battle records into a numerical feature matrix
    feature_matrix = []
    stat_registry = build_pokemon_stat_registry(source_data)

    for battle in tqdm(source_data, desc="Extracting features"):
        current_features = {}

        p1_swap_count, p1_effects, p1_team_state, p2_swap_count, p2_effects, p2_team_state = extract_battle_summary(battle)

        p1_mean_pc_hp = np.mean([info['hp'] for info in p1_team_state.values()])
        p2_mean_pc_hp = np.mean([info['hp'] for info in p2_team_state.values()])
        current_features['p1_mean_pc_hp'] = p1_mean_pc_hp
        current_features['p2_mean_pc_hp'] = p2_mean_pc_hp

        p1_surviving_pokemon = sum(1 for info in p1_team_state.values() if info["hp"] > 0)
        p2_surviving_pokemon = sum(1 for info in p2_team_state.values() if info["hp"] > 0)
        current_features['p1_surviving_pokemon'] = p1_surviving_pokemon

        p1_status_score = sum(1 for i in p1_team_state.values() if i['hp'] > 0 and i['status'] != 'nostatus') + p1_effects
        p2_status_score = sum(1 for i in p2_team_state.values() if i['hp'] > 0 and i['status'] != 'nostatus') + p2_effects
        current_features['p1_status_score'] = p1_status_score
        current_features['p2_status_score'] = p2_status_score

        speed_sum_diff, attack_sum_diff, sp_attack_sum_diff, sp_defense_sum_diff, hp_sum_diff = compute_base_stat_differences(
            p1_team_state, p2_team_state, stat_registry
        )
        current_features['speed_sum_diff'] = speed_sum_diff
        current_features['attack_sum_diff'] = attack_sum_diff
        current_features['sp_attack_sum_diff'] = sp_attack_sum_diff
        current_features['sp_defense_sum_diff'] = sp_defense_sum_diff
        current_features['hp_sum_diff'] = hp_sum_diff

        current_features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            current_features['player_won'] = int(battle['player_won'])

        feature_matrix.append(current_features)

    return pd.DataFrame(feature_matrix).fillna(0)


def build_feature_tables_and_spearman(train_source, test_source):
     #---Raw data loading---
    train_raw = []
    print(f"Loading data from '{train_source}'...")
    try:
        with open(train_source, 'r') as f:
            for line in f:
                train_raw.append(json.loads(line))
        print(f"Successfully loaded {len(train_raw)} battles.")
    except FileNotFoundError:
        print(f"ERROR: Could not find the training file at '{train_source}'.")

    # Removes battle 4877 from training for consistency with the original logic
    train_raw = [battle for battle in train_raw if battle.get("battle_id") != 4877]

    test_raw = []
    print(f"Loading data from '{test_source}'...")
    try:
        with open(test_source, 'r') as f:
            for line in f:
                test_raw.append(json.loads(line))
        print(f"Successfully loaded {len(test_raw)} battles.")
    except FileNotFoundError:
        print(f"ERROR: Could not find the test file at '{test_source}'.")

    #---Feature table construction and Spearman analysis---
    print("Processing training data...")
    train_df = compute_final_features(train_raw)

    print("\nProcessing test data...")
    test_df = compute_final_features(test_raw)

    print("\nTraining features preview:")
    display(train_df.head())

    BASE_FEATURES_10 = [
        'p1_mean_pc_hp', 'p2_mean_pc_hp',
        'p1_surviving_pokemon',
        'p1_status_score', 'p2_status_score',
        'speed_sum_diff', 'attack_sum_diff',
        'sp_attack_sum_diff',
        'sp_defense_sum_diff', 'hp_sum_diff'
    ]
    BASE_FEATURES_10 = [f for f in BASE_FEATURES_10 if f in train_df.columns]

    print("\n=== Spearman correlation tra le 10 feature ===")
    spearman_matrix = train_df[BASE_FEATURES_10].corr(method='spearman')
    display(spearman_matrix)

    print("\n=== Spearman & Partial Spearman vs target (player_won) ===")
    df_pc = train_df[BASE_FEATURES_10 + ['player_won']].copy()

    partial_rows = []
    for feat in BASE_FEATURES_10:
        rho_simple, _ = spearmanr(df_pc[feat], df_pc['player_won'])
        others = [f for f in BASE_FEATURES_10 if f != feat]
        X_others = df_pc[others].values
        n = X_others.shape[0]
        X_design = np.hstack([np.ones((n, 1)), X_others])
        beta_f, *_ = np.linalg.lstsq(X_design, df_pc[feat].values, rcond=None)
        resid_f = df_pc[feat].values - X_design @ beta_f
        beta_t, *_ = np.linalg.lstsq(X_design, df_pc['player_won'].values, rcond=None)
        resid_t = df_pc['player_won'].values - X_design @ beta_t
        rho_partial, _ = spearmanr(resid_f, resid_t)
        partial_rows.append({
            "feature": feat,
            "spearman_vs_target": rho_simple,
            "partial_spearman_vs_target": rho_partial
        })

    partial_df = pd.DataFrame(partial_rows)
    partial_df = partial_df.sort_values(by="partial_spearman_vs_target", key=lambda c: np.abs(c), ascending=False)
    display(partial_df)

    return train_df, test_df, BASE_FEATURES_10, spearman_matrix, partial_df
