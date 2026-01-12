import pandas as pd
import numpy as np
import csv

# Lista de coduri de țară pentru curățarea numelor
COUNTRY_CODES = {
    'it', 'nl', 'ch', 'eng', 'pt', 'fr', 'de', 'hr', 'es', 'cz', 
    'at', 'ua', 'sct', 'sk', 'be', 'rs', 'tr', 'kz', 'az', 'no', 'cy'
}

def clean_team_name(name):
    """Normalizează numele echipei eliminând codul de țară (prefix sau sufix)."""
    if not isinstance(name, str): return name
    parts = name.strip().split()
    if not parts: return name
    
    if parts[0].lower() in COUNTRY_CODES: parts = parts[1:]
    if not parts: return ""
    if parts[-1].lower() in COUNTRY_CODES: parts = parts[:-1]
    
    return " ".join(parts).strip()

def parse_and_calculate_ratings(matches_file, priors_file):
    print(f"Încărcăm datele și calculăm cele 3 modele de rating...")
    
    try:
        # 1. Citire Meciuri
        df_m = pd.read_csv(matches_file, encoding='utf-8-sig', sep=None, engine='python')
        df_m = df_m.iloc[:, :5]
        df_m.columns = ['Home', 'xG_Home', 'Score', 'xG_Away', 'Away']
        df_m = df_m.dropna(subset=['Home', 'Away'])
        df_m['Home'] = df_m['Home'].apply(clean_team_name)
        df_m['Away'] = df_m['Away'].apply(clean_team_name)

        # 2. Citire Date Live (Priors)
        df_p = pd.read_csv(priors_file, encoding='utf-8-sig', sep=None, engine='python')
        df_p.columns = ['Team', 'Value', 'Coeff']
        df_p['Team'] = df_p['Team'].apply(clean_team_name)
        priors_dict = df_p.set_index('Team').to_dict('index')
        
    except Exception as e:
        print(f"Eroare la citirea fișierelor: {e}"); return None

    df_m['xG_Home'] = df_m['xG_Home'].astype(str).str.replace(',', '.').astype(float)
    df_m['xG_Away'] = df_m['xG_Away'].astype(str).str.replace(',', '.').astype(float)

    # 3. Acumulare Statistici
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

    # 4. Calcul Modele
    league_avg_xg = (df_m['xG_Home'].mean() + df_m['xG_Away'].mean()) / 2
    max_val, max_coeff = df_p['Value'].max(), df_p['Coeff'].max()
    smoothing = 2

    classic, hybrid, full = {}, {}, {}

    for t in stats:
        # Model 1: Classic (xG Brut cu smoothing)
        att_brut = (sum(stats[t]['xg_list']) + (smoothing * league_avg_xg)) / (len(stats[t]['xg_list']) + smoothing)
        def_brut = (sum(stats[t]['xga_list']) + (smoothing * league_avg_xg)) / (len(stats[t]['xga_list']) + smoothing)
        classic[t] = {'att': att_brut / league_avg_xg, 'def': def_brut / league_avg_xg}

    for t in stats:
        # Pregătire variabile pentru Adjusted
        opp_def = np.mean([classic[opp]['def'] for opp in stats[t]['opp']])
        opp_att = np.mean([classic[opp]['att'] for opp in stats[t]['opp']])
        xg_att_adj = classic[t]['att'] / opp_def
        xg_def_adj = classic[t]['def'] / opp_att
        
        p = priors_dict.get(t, {'Value': df_p['Value'].median(), 'Coeff': df_p['Coeff'].median()})
        power_f = 0.7 + (p['Value'] / max_val * 0.4) + (p['Coeff'] / max_coeff * 0.4)
        clinical_att = (stats[t]['gs'] + 1) / (sum(stats[t]['xg_list']) + 1)
        clinical_def = (stats[t]['ga'] + 1) / (sum(stats[t]['xga_list']) + 1)

        # Model 2: Hybrid (60% xG Ajustat + 40% Power)
        hybrid[t] = {
            'attack_rating': (xg_att_adj * 0.6) + (power_f * 0.4),
            'defense_rating': (xg_def_adj * 0.6) + ((1/power_f) * 0.4)
        }

        # Model 3: Full Chaos (50% xG Ajustat + 30% Power + 20% Clinicality)
        full[t] = {
            'attack_rating': (xg_att_adj * 0.5) + (power_f * 0.3) + (clinical_att * 0.2),
            'defense_rating': (xg_def_adj * 0.5) + ((1/power_f) * 0.3) + (clinical_def * 0.2)
        }

    standings = sorted(stats.keys(), key=lambda x: (stats[x]['pts'], stats[x]['gd'], stats[x]['gs']), reverse=True)
    return {'standings': standings, 'classic': classic, 'hybrid': hybrid, 'full': full, 'league_avg': league_avg_xg, 'home_adv': 1.1}