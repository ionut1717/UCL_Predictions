import pandas as pd
import numpy as np
import csv

# Lista de coduri pentru parsarea datelor
COUNTRY_CODES = {
    'it', 'nl', 'ch', 'eng', 'pt', 'fr', 'de', 'hr', 'es', 'cz', 
    'at', 'ua', 'sct', 'sk', 'be', 'rs', 'tr', 'kz', 'az', 'no', 'cy'
}

def clean_team_name(name):
    if not isinstance(name, str): return name
    parts = name.strip().split()
    if not parts: return name
    if parts[0].lower() in COUNTRY_CODES: parts = parts[1:]
    if not parts: return ""
    if parts[-1].lower() in COUNTRY_CODES: parts = parts[:-1]
    return " ".join(parts).strip()

def parse_and_calculate_ratings(matches_file, priors_file):    
    try:
        # 1. Citire și Curățare
        df_m = pd.read_csv(matches_file, encoding='utf-8-sig', sep=None, engine='python')
        df_m = df_m.iloc[:, :5]
        df_m.columns = ['Home', 'xG_Home', 'Score', 'xG_Away', 'Away']
        df_m = df_m.dropna(subset=['Home', 'Away'])
        df_m['Home'] = df_m['Home'].apply(clean_team_name)
        df_m['Away'] = df_m['Away'].apply(clean_team_name)

        df_p = pd.read_csv(priors_file, encoding='utf-8-sig', sep=None, engine='python')
        df_p.columns = ['Team', 'Value', 'Coeff']
        df_p['Team'] = df_p['Team'].apply(clean_team_name)
        priors_dict = df_p.set_index('Team').to_dict('index')
        
    except Exception as e:
        print(f"Eroare la citirea fisierelor: {e}"); return None

    df_m['xG_Home'] = df_m['xG_Home'].astype(str).str.replace(',', '.').astype(float)
    df_m['xG_Away'] = df_m['xG_Away'].astype(str).str.replace(',', '.').astype(float)

    # 2. Acumulare Statistici
    stats = {}
    for _, row in df_m.iterrows():
        h, a = row['Home'], row['Away']
        try:
            score_str = str(row['Score']).replace('–', '-').replace('—', '-')
            gh, ga = map(int, score_str.split('-'))
        except: gh, ga = 0, 0

        for t in [h, a]:
            if t not in stats:
                stats[t] = {'pts': 0, 'gd': 0, 'gs': 0, 'ga': 0, 'wins': 0, 'xg_list': [], 'xga_list': [], 'opp': []}
        
        stats[h]['gs'] += gh; stats[h]['ga'] += ga; stats[h]['gd'] += (gh - ga)
        stats[a]['gs'] += ga; stats[a]['ga'] += gh; stats[a]['gd'] += (ga - gh)
        stats[h]['xg_list'].append(row['xG_Home']); stats[h]['xga_list'].append(row['xG_Away']); stats[h]['opp'].append(a)
        stats[a]['xg_list'].append(row['xG_Away']); stats[a]['xga_list'].append(row['xG_Home']); stats[a]['opp'].append(h)
        
        if gh > ga: stats[h]['pts'] += 3; stats[h]['wins'] += 1
        elif ga > gh: stats[a]['pts'] += 3; stats[a]['wins'] += 1
        else: stats[h]['pts'] += 1; stats[a]['pts'] += 1

    # 3. Calcul variabile de bază
    league_avg_xg = (df_m['xG_Home'].mean() + df_m['xG_Away'].mean()) / 2
    max_val, max_coeff = df_p['Value'].max(), df_p['Coeff'].max()
    smoothing = 0

    classic, hybrid, full = {}, {}, {}

    # --- PASUL A: MODEL 1 (Classic) ---
    for t in stats:
        att_brut = (sum(stats[t]['xg_list']) + (smoothing * league_avg_xg)) / (len(stats[t]['xg_list']) + smoothing)
        def_brut = (sum(stats[t]['xga_list']) + (smoothing * league_avg_xg)) / (len(stats[t]['xga_list']) + smoothing)
        classic[t] = {'att': att_brut / league_avg_xg, 'def': def_brut / league_avg_xg}

    # --- PASUL B: POWER FACTORS (Valoare Lot + UEFA) ---
    power_factors = {}
    for t in stats:
        p = priors_dict.get(t, {'Value': df_p['Value'].median(), 'Coeff': df_p['Coeff'].median()})
        #Cat de mult conteaza banii si coeficientul uefa
        power_factors[t] = 0.7 + (p['Value'] / max_val * 0.4) + (p['Coeff'] / max_coeff * 0.4)

    # --- PASUL C: MODELELE 2 ȘI 3 (Meticulos și Chaos) ---
    for t in stats:
        match_perf_att = []
        match_perf_def = []
        adj_xg_sum = 0 # Suma xG-ului personalizat pentru Clinical Factor
        
        for i in range(len(stats[t]['xg_list'])):
            raw_xg = stats[t]['xg_list'][i]
            raw_xga = stats[t]['xga_list'][i]
            
            opp = stats[t]['opp'][i]
            
            # Puterea adversarului (Classic xG + Power Factor)
            opp_def_str = (classic[opp]['def'] * 0.6) + ((1 / power_factors[opp]) * 0.4)
            opp_att_str = (classic[opp]['att'] * 0.6) + (power_factors[opp] * 0.4)
            
            # Ajustare meci cu meci (xG Personalizat)
            match_xg_adj = raw_xg / (opp_def_str * league_avg_xg)
            match_xga_adj = raw_xga / (opp_att_str * league_avg_xg)
            
            match_perf_att.append(match_xg_adj)
            match_perf_def.append(match_xga_adj)

        # Model 2: Hybrid (Meticulos)
        m2_att = np.mean(match_perf_att)
        m2_def = np.mean(match_perf_def)
        hybrid[t] = {'attack_rating': m2_att, 'defense_rating': m2_def}

        # Model 3: Full Chaos (+ Clinical Factor bazat pe xG Personalizat)
        # Clinical Factor = Goluri Reale / xG Personalizat
        # Folosim smoothing (+1) pentru a evita împărțirea la zero sau rezultate extreme
        raw_xg_sum = sum(stats[t]['xg_list'])
        clinical_factor = (stats[t]['gs'] + 1) / (raw_xg_sum + 1)

        raw_xga_sum = sum(stats[t]['xga_list'])
        clinical_factor_def = (stats[t]['ga'] + 1) / (raw_xga_sum + 1)
        
        full[t] = {
            'attack_rating': (m2_att * clinical_factor * 0.2) + m2_att * 0.8, # Recompensă/Penalizare eficiență
            'defense_rating': (m2_def * clinical_factor_def * 0.2) + m2_def * 0.8  # Apărarea rămâne cea din modelul meticulos
        }

    standings = sorted(stats.keys(), key=lambda x: (stats[x]['pts'], stats[x]['gd'], stats[x]['gs']), reverse=True)
    return {
        'standings': standings, 
        'classic': classic, 
        'hybrid': hybrid, 
        'full': full, 
        'league_avg': league_avg_xg, 
        'home_adv': 1.1
    }