import pandas as pd
import numpy as np
import csv

# Lista extinsă de coduri de țară care apar în datele tale
COUNTRY_CODES = {
    'it', 'nl', 'ch', 'eng', 'pt', 'fr', 'de', 'hr', 'es', 'cz', 
    'at', 'ua', 'sct', 'sk', 'be', 'rs', 'rs', 'sk', 'ua', 'cz'
}

def clean_team_name(name):
    """
    Normalizează numele echipei prin eliminarea codului de țară 
    de la începutul sau de la finalul numelui.
    """
    if not isinstance(name, str):
        return name
    
    parts = name.strip().split()
    if not parts:
        return name
    
    # Eliminăm codul dacă este la început (ex: 'es Real Madrid')
    if parts[0].lower() in COUNTRY_CODES:
        parts = parts[1:]
    
    if not parts: return ""

    # Eliminăm codul dacă este la final (ex: 'Real Madrid es')
    if parts[-1].lower() in COUNTRY_CODES:
        parts = parts[:-1]
        
    return " ".join(parts).strip()

def parse_and_calculate_ratings(matches_file, priors_file):
    print(f"Procesăm datele și normalizăm numele echipelor...")
    
    try:
        # 1. Citirea și curățarea meciurilor
        df_m = pd.read_csv(matches_file, encoding='utf-8-sig', sep=None, engine='python')
        df_m = df_m.iloc[:, :5]
        df_m.columns = ['Home', 'xG_Home', 'Score', 'xG_Away', 'Away']
        df_m = df_m.dropna(subset=['Home', 'Away'])

        # Aplicăm normalizarea numelor pentru Home și Away
        df_m['Home'] = df_m['Home'].apply(clean_team_name)
        df_m['Away'] = df_m['Away'].apply(clean_team_name)

        # 2. Citirea și normalizarea datelor de tip Priors
        df_p = pd.read_csv(priors_file, encoding='utf-8-sig', sep=None, engine='python')
        df_p.columns = ['Team', 'Value', 'Coeff']
        
        # Aplicăm normalizarea și pe coloana Team din Priors
        df_p['Team'] = df_p['Team'].apply(clean_team_name)
        priors_dict = df_p.set_index('Team').to_dict('index')
        
    except Exception as e:
        print(f"Eroare critică: {e}")
        return None

    # Conversie xG
    df_m['xG_Home'] = df_m['xG_Home'].astype(str).str.replace(',', '.').astype(float)
    df_m['xG_Away'] = df_m['xG_Away'].astype(str).str.replace(',', '.').astype(float)

    max_val = df_p['Value'].max()
    max_coeff = df_p['Coeff'].max()

    # 3. Calcul Statistici
    stats = {}
    for _, row in df_m.iterrows():
        h, a = row['Home'], row['Away']
        xg_h, xg_a = row['xG_Home'], row['xG_Away']
        
        try:
            score_str = str(row['Score']).replace('–', '-').replace('—', '-')
            gh, ga = map(int, score_str.split('-'))
        except: 
            gh, ga = 0, 0

        for t in [h, a]:
            if t not in stats:
                stats[t] = {'pts': 0, 'gd': 0, 'gs': 0, 'wins': 0, 'xg_list': [], 'xga_list': [], 'opp': []}
        
        stats[h]['gs'] += gh; stats[a]['gs'] += ga
        stats[h]['gd'] += (gh - ga); stats[a]['gd'] += (ga - gh)
        stats[h]['xg_list'].append(xg_h); stats[h]['xga_list'].append(xg_a); stats[h]['opp'].append(a)
        stats[a]['xg_list'].append(xg_a); stats[a]['xga_list'].append(xg_h); stats[a]['opp'].append(h)
        
        if gh > ga: stats[h]['pts'] += 3; stats[h]['wins'] += 1
        elif ga > gh: stats[a]['pts'] += 3; stats[a]['wins'] += 1
        else: stats[h]['pts'] += 1; stats[a]['pts'] += 1

    standings = sorted(stats.keys(), key=lambda x: (stats[x]['pts'], stats[x]['gd'], stats[x]['gs'], stats[x]['wins']), reverse=True)
    
    # 4. Calcul Ratings (60/40)
    league_avg_xg = (df_m['xG_Home'].mean() + df_m['xG_Away'].mean()) / 2
    smoothing_mp = 2

    classic = {}
    for t in stats:
        adj_xg_avg = (sum(stats[t]['xg_list']) + (smoothing_mp * league_avg_xg)) / (len(stats[t]['xg_list']) + smoothing_mp)
        adj_xga_avg = (sum(stats[t]['xga_list']) + (smoothing_mp * league_avg_xg)) / (len(stats[t]['xga_list']) + smoothing_mp)
        classic[t] = {'att': adj_xg_avg / league_avg_xg, 'def': adj_xga_avg / league_avg_xg}
    
    adjusted = {}
    for t in stats:
        opp_def = [classic[opp]['def'] for opp in stats[t]['opp']]
        opp_att = [classic[opp]['att'] for opp in stats[t]['opp']]
        xg_att = classic[t]['att'] / np.mean(opp_def)
        xg_def = classic[t]['def'] / np.mean(opp_att)
        
        # Power Factor normalizat (0.7 baza + bonusuri)
        p = priors_dict.get(t, {'Value': df_p['Value'].median(), 'Coeff': df_p['Coeff'].median()})
        power_factor = 0.7 + (p['Value'] / max_val * 0.4) + (p['Coeff'] / max_coeff * 0.4)
        
        adjusted[t] = {
            'attack_rating': (xg_att * 0.6) + (power_factor * 0.4),
            'defense_rating': (xg_def * 0.6) + ((1/power_factor) * 0.4)
        }

    return {
        'standings': standings, 
        'classic': classic, 
        'adjusted': adjusted, 
        'league_avg': league_avg_xg, 
        'home_adv': df_m['xG_Home'].mean() / df_m['xG_Away'].mean()
    }