import pandas as pd
import numpy as np
import csv

def parse_and_calculate_ratings(file_path):
    print(f"Procesăm datele din: {file_path}...")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig', sep=None, engine='python', quoting=csv.QUOTE_MINIMAL)
        
        df = df.iloc[:, :5]
        df.columns = ['Home', 'xG_Home', 'Score', 'xG_Away', 'Away']
    except Exception as e:
        print(f"Eroare la citire: {e}")
        return None

    # Curățare date: eliminăm rândurile goale și convertim xG
    df = df.dropna(subset=['Home', 'Away'])
    df['xG_Home'] = df['xG_Home'].astype(str).str.replace(',', '.').astype(float)
    df['xG_Away'] = df['xG_Away'].astype(str).str.replace(',', '.').astype(float)
    for col in ['Home', 'Away']: df[col] = df[col].astype(str).str.strip()

    # 1. CALCULUL CLASAMENTULUI (Criterii FIFA)
    stats = {}
    for _, row in df.iterrows():
        h, a = row['Home'], row['Away']
        xg_h, xg_a = row['xG_Home'], row['xG_Away']
        try:
            gh, ga = map(int, str(row['Score']).split('-'))
        except: gh, ga = 0, 0

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

    # Sortăm clasamentul
    standings = sorted(stats.keys(), key=lambda x: (stats[x]['pts'], stats[x]['gd'], stats[x]['gs'], stats[x]['wins']), reverse=True)
    
    # 2. CALCUL RATINGS (Classic vs Adjusted)
    league_avg_xg = (df['xG_Home'].mean() + df['xG_Away'].mean()) / 2
    classic = {t: {'att': np.mean(stats[t]['xg_list']) / league_avg_xg, 
                   'def': np.mean(stats[t]['xga_list']) / league_avg_xg} for t in stats}
    
    adjusted = {}
    for t in stats:
        avg_opp_def = np.mean([classic[opp]['def'] for opp in stats[t]['opp']])
        avg_opp_att = np.mean([classic[opp]['att'] for opp in stats[t]['opp']])
        adjusted[t] = {'attack_rating': classic[t]['att'] / avg_opp_def, 
                       'defense_rating': classic[t]['def'] / avg_opp_att}

    return {'standings': standings, 'classic': classic, 'adjusted': adjusted, 
            'league_avg': league_avg_xg, 'home_adv': df['xG_Home'].mean() / df['xG_Away'].mean()}