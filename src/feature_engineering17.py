import pandas as pd
import numpy as np
import json
from tqdm import tqdm
# Importa costanti e mappe da .constants
from .constants import CONDITION_MALUS, TYPE_INTERACTIONS, W_SPEED_F, W_ATT_SPATT_F, W_DEF_SPDEF_F, W_HP_F

def display_frame_preview(df_output):
    print(df_output.head())

def get_type_modifier_val(attack_type, defense_type):
    #Moltiplicatore di danno per tipo (usa TYPE_INTERACTIONS)
    return TYPE_INTERACTIONS.get(attack_type, {}).get(defense_type, 1.0)

def calculate_meta_type_scores(p1_cond_map, p2_cond_map, stats_data):
    #score di vantaggio/vulnerabilità di tipo del team.
    p1_off_score, p2_off_score = 0.0, 0.0
    p1_type_set, p2_type_set = [], []
    for pk_name in p1_cond_map.keys():
        pk_info = stats_data.get(pk_name)
        if pk_info: p1_type_set.extend(pk_info['types'])
    for pk_name in p2_cond_map.keys():
        pk_info = stats_data.get(pk_name)
        if pk_info: p2_type_set.extend(pk_info['types'])
    p1_types_unique = list(set([t for t in p1_type_set if t != 'notype']))
    p2_types_unique = list(set([t for t in p2_type_set if t != 'notype']))
    if p1_types_unique and p2_types_unique:
        for p1_att_type in p1_types_unique:
            for p2_def_type in p2_types_unique: p1_off_score += get_type_modifier_val(p1_att_type, p2_def_type)
        for p2_att_type in p2_types_unique:
            for p1_def_type in p1_types_unique: p2_off_score += get_type_modifier_val(p2_att_type, p1_def_type)
    p1_net_off_score = p1_off_score - p2_off_score
    p1_def_vulnerability_ratio = p2_off_score / (p1_off_score + 1e-6)
    return p1_net_off_score, p1_def_vulnerability_ratio

def data_ingestion(filepath, file_context="training"):
    raw_entries = []
    print(f"Acquisizione dati da '{filepath}'...")
    try:
        with open(filepath, 'r') as f:
            for line in f: raw_entries.append(json.loads(line))
        initial_count = len(raw_entries)
        if file_context == "training":
            raw_entries = [battle for battle in raw_entries if battle.get("battle_id") != 4877]
            if len(raw_entries) < initial_count: print("Rimosso record battaglia 4877 dai dati di training.")
        print(f"Caricati {len(raw_entries)} record di battaglia con successo.")
        return raw_entries
    except FileNotFoundError:
        print(f"ERRORE: Impossibile trovare il file sorgente a '{filepath}'.")
        return []

def build_pokemon_stat_registry(battle_records: list[dict]):
    #registro statico delle statistiche base dei Pokémon (livello 100)
    stat_registry = {}
    for entry in battle_records:
        pkmn_list = entry.get('p1_team_details', [])
        p2_lead_data = entry.get('p2_lead_details', {})
        if p2_lead_data: pkmn_list.append(p2_lead_data)
        for pkmn in pkmn_list:
            mon_name = pkmn.get('name')
            mon_level = pkmn.get('level')
            if mon_name and mon_name not in stat_registry and mon_level == 100:
                base_stats = {k: v for k, v in pkmn.items() if k.startswith('base_')}
                base_stats['types'] = pkmn.get('types', ['notype', 'notype'])
                if base_stats: stat_registry[mon_name] = base_stats
    return stat_registry

def extract_battle_summary(battle_data):
    # dati dinamici (HP finali, status, switch counts, leader attivo)
    p1_team_state = {p.get('name', f'p1_unknown_{i}'): {'hp': 1.00, 'status': 'nostatus'} for i, p in enumerate(battle_data.get('p1_team_details', []))}
    p2_team_state = {battle_data.get('p2_lead_details', {}).get('name'): {'hp': 1.00, 'status': 'nostatus'}}
    timeline = battle_data.get('battle_timeline', [])
    last_turn_snapshot = timeline[-1] if timeline else None
    final_p1_active, final_p2_active = None, None
    for turn_entry in timeline:
        p1_snapshot, p2_snapshot = turn_entry.get('p1_pokemon_state', {}), turn_entry.get('p2_pokemon_state', {})
        p1_name, p2_name = p1_snapshot.get('name'), p2_snapshot.get('name')
        if p1_name: p1_team_state[p1_name] = {'hp': p1_snapshot.get('hp_pct'), 'status': p1_snapshot.get('status')}
        if p2_name: p2_team_state[p2_name] = {'hp': p2_snapshot.get('hp_pct'), 'status': p2_snapshot.get('status')}
        final_p1_active = p1_name
        final_p2_active = p2_name
    p1_swap_count, p2_swap_count = 0, 0
    p1_prev_pk, p2_prev_pk = None, None
    for turn_entry in timeline:
        p1_current_pk = turn_entry.get('p1_pokemon_state', {}).get('name')
        p2_current_pk = turn_entry.get('p2_pokemon_state', {}).get('name')
        if p1_prev_pk is not None:
            if p1_prev_pk != p1_current_pk: p1_swap_count += 1
            if p2_prev_pk != p2_current_pk: p2_swap_count += 1
        p1_prev_pk, p2_prev_pk = p1_current_pk, p2_current_pk
    p1_field_effect_score = len(last_turn_snapshot.get('p1_pokemon_state', {}).get('effects', [])) * 0.4 if last_turn_snapshot else 0
    p2_field_effect_score = len(last_turn_snapshot.get('p2_pokemon_state', {}).get('effects', [])) * 0.4 if last_turn_snapshot else 0
    for i in range(len(p2_team_state), 6): p2_team_state[f'p2_unknown_{i}'] = {'hp': 1.00, 'status': 'nostatus'}
    p1_lead_status = last_turn_snapshot.get('p1_pokemon_state', {}) if last_turn_snapshot else {}
    p2_lead_status = last_turn_snapshot.get('p2_pokemon_state', {}) if last_turn_snapshot else {}
    active_leader_summary = {
        'p1_hp_fraction': p1_lead_status.get('hp_pct', 1.0), 'p2_hp_fraction': p2_lead_status.get('hp_pct', 1.0),
        'p1_status_value': CONDITION_MALUS.get(p1_lead_status.get('status', 'nostatus'), 0),
        'p2_status_value': CONDITION_MALUS.get(p2_lead_status.get('status', 'nostatus'), 0),
        'p1_active_pk_name': final_p1_active, 'p2_active_pk_name': final_p2_active
    }
    return (p1_swap_count, p1_field_effect_score, p1_team_state, p2_swap_count, p2_field_effect_score, p2_team_state, active_leader_summary)

def compute_final_features(source_data: list[dict]) -> pd.DataFrame:
    #Trasformazione dei record di battaglia in matrice di feature
    feature_matrix = []
    if not source_data: return pd.DataFrame()
    pkmn_stat_map = build_pokemon_stat_registry(source_data)
    for battle_record in tqdm(source_data, desc="Costruzione Matrice di Feature"):
        current_features = {}
        (p1_swaps, p1_effects, p1_conds_map, p2_swaps, p2_effects, p2_conds_map, lead_data) = extract_battle_summary(battle_record)
        # Metriche Aggregate di Stato del Team
        #Calcoli intermedi per lo status del Team
        p1_avg_hp = np.mean(list(info['hp'] for info in p1_conds_map.values()))
        p2_avg_hp = np.mean(list(info['hp'] for info in p2_conds_map.values()))
        p1_survivors = sum(1 for info in p1_conds_map.values() if info["hp"] > 0)
        p2_survivors = sum(1 for info in p2_conds_map.values() if info["hp"] > 0)
        p1_total_status_score = sum(1 for i in p1_conds_map.values() if i['hp'] > 0 and i['status'] != 'nostatus') + p1_effects
        p2_total_status_score = sum(1 for i in p2_conds_map.values() if i['hp'] > 0 and i['status'] != 'nostatus') + p2_effects
        #metriche aggregate di stato del Team (Dinamiche)
        current_features['hp_avg_delta'] = p1_avg_hp - p2_avg_hp
        current_features['survivor_count_delta'] = p1_survivors - p2_survivors
        current_features['team_status_net_delta'] = p1_total_status_score - p2_total_status_score
        current_features['switch_activity_log_ratio'] = np.log((p1_swaps + 1) / (p2_swaps + 1))
        # Metriche di Potenza Statistica (Team Totale)
        p1_sum_stats = {'spe': 0, 'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'hp': 0}
        p2_sum_stats = {'spe': 0, 'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'hp': 0}
        for name in p1_conds_map.keys():
            stats = pkmn_stat_map.get(name, {})
            for key in p1_sum_stats.keys(): p1_sum_stats[key] += stats.get(f'base_{key}', 0)
        for name in p2_conds_map.keys():
            stats = pkmn_stat_map.get(name, {})
            for key in p2_sum_stats.keys(): p2_sum_stats[key] += stats.get(f'base_{key}', 0)
        p1_off_total = p1_sum_stats['atk'] + p1_sum_stats['spa']
        p1_def_total = p1_sum_stats['def'] + p1_sum_stats['spd']
        p2_off_total = p2_sum_stats['atk'] + p2_sum_stats['spa']
        p2_def_total = p2_sum_stats['def'] + p2_sum_stats['spd']
        #differenza nel bilanciamento Offensiva-Difesa totale
        current_features['stat_dom_diff_metric'] = (p1_off_total - p1_def_total) - (p2_off_total - p2_def_total)
        #differenza nella somma totale di velocità
        current_features['speed_sum_diff'] = p1_sum_stats['spe'] - p2_sum_stats['spe']
        # Differenza nella somma totale di Attacco Fisico
        current_features['attack_sum_diff'] = p1_sum_stats['atk'] - p2_sum_stats['atk']
        # Differenza nella somma totale di Difesa Fisica
        current_features['defense_sum_diff'] = p1_sum_stats['def'] - p2_sum_stats['def']
        # Differenza nella somma totale di Attacco Speciale
        current_features['sp_attack_sum_diff'] = p1_sum_stats['spa'] - p2_sum_stats['spa']
        # Differenza nella somma totale di Difesa Speciale
        current_features['sp_defense_sum_diff'] = p1_sum_stats['spd'] - p2_sum_stats['spd']
        # Differenza nella somma totale di HP
        current_features['hp_sum_diff'] = p1_sum_stats['hp'] - p2_sum_stats['hp']
        p1_weighted_idx = (p1_sum_stats['spe'] * W_SPEED_F) + (p1_sum_stats['atk'] * W_ATT_SPATT_F) + (p1_sum_stats['spa'] * W_ATT_SPATT_F) + (p1_sum_stats['def'] * W_DEF_SPDEF_F) + (p1_sum_stats['spd'] * W_DEF_SPDEF_F) + (p1_sum_stats['hp'] * W_HP_F)
        p2_weighted_idx = (p2_sum_stats['spe'] * W_SPEED_F) + (p2_sum_stats['atk'] * W_ATT_SPATT_F) + (p2_sum_stats['spa'] * W_ATT_SPATT_F) + (p2_sum_stats['def'] * W_DEF_SPDEF_F) + (p2_sum_stats['spd'] * W_DEF_SPDEF_F) + (p2_sum_stats['hp'] * W_HP_F)
        #Differenza nell'Indice di Potenza Ponderata Totale
        current_features['total_weighted_power_delta'] = p1_weighted_idx - p2_weighted_idx
        # Metriche di Vantaggio di Tipo
        p1_type_adv, p1_type_vuln = calculate_meta_type_scores(p1_conds_map, p2_conds_map, pkmn_stat_map)
        # Score di vantaggio offensivo netto di P1 vs P2. Positivo = P1 ha vantaggio di tipo offensivo.
        current_features['p1_off_type_advantage_score'] = p1_type_adv
        # Rapporto di vulnerabilità difensiva di P1 (P2 Off Score / P1 Off Score). Alto = P1 è vulnerabile.
        current_features['p1_def_type_vulnerability_ratio'] = p1_type_vuln
        # FEATURE DEL LEAD ATTIVO FINALE
        p1_lead_name = lead_data['p1_active_pk_name']
        p2_lead_name = lead_data['p2_active_pk_name']
        p1_lead_stats = pkmn_stat_map.get(p1_lead_name, {'base_spe': 0, 'base_atk': 0, 'base_spa': 0, 'base_def': 0, 'base_spd': 0, 'base_hp': 0})
        p2_lead_stats = pkmn_stat_map.get(p2_lead_name, {'base_spe': 0, 'base_atk': 0, 'base_spa': 0, 'base_def': 0, 'base_spd': 0, 'base_hp': 0})
        p1_lead_power_idx = (p1_lead_stats.get('base_spe', 0) * W_SPEED_F) + (p1_lead_stats.get('base_atk', 0) * W_ATT_SPATT_F) + (p1_lead_stats.get('base_spa', 0) * W_ATT_SPATT_F) + (p1_lead_stats.get('base_def', 0) * W_DEF_SPDEF_F) + (p1_lead_stats.get('base_spd', 0) * W_DEF_SPDEF_F) + (p1_lead_stats.get('base_hp', 0) * W_HP_F)
        p2_lead_power_idx = (p2_lead_stats.get('base_spe', 0) * W_SPEED_F) + (p2_lead_stats.get('base_atk', 0) * W_ATT_SPATT_F) + (p2_lead_stats.get('base_spa', 0) * W_ATT_SPATT_F) + (p2_lead_stats.get('base_def', 0) * W_DEF_SPDEF_F) + (p2_lead_stats.get('base_spd', 0) * W_DEF_SPDEF_F) + (p2_lead_stats.get('base_hp', 0) * W_HP_F)
        # Differenza nell'Indice di Potenza Ponderata tra i Leader finali
        current_features['lead_stat_power_diff'] = p1_lead_power_idx - p2_lead_power_idx
        # Differenza nella frazione HP tra i Leader finali
        current_features['lead_hp_frac_diff'] = lead_data['p1_hp_fraction'] - lead_data['p2_hp_fraction']
        # Differenza nel valore di afflizione di stato tra i Leader finali
        current_features['lead_status_value_diff'] = lead_data['p1_status_value'] - lead_data['p2_status_value']
        # --- Identificatori e Target ---
        current_features['battle_unique_id'] = battle_record.get('battle_id')
        if 'player_won' in battle_record: current_features['p1_won_flag'] = int(battle_record['player_won'])
        feature_matrix.append(current_features)
    df_result = pd.DataFrame(feature_matrix).fillna(0)
    df_result = df_result.rename(columns={'battle_unique_id': 'battle_id', 'p1_won_flag': 'player_won'}, errors='ignore')
    return df_result
