import numpy as np
import random
import plotly.graph_objects as go
from collections import Counter
from Parsare_Meciuri import parse_and_calculate_ratings

# Configurare fișiere
FILE_MECIURI = "Meciuri_UCL_2025-2026.csv"
FILE_PRIORS = "Echipe_2025-2026.csv"

data = parse_and_calculate_ratings(FILE_MECIURI, FILE_PRIORS)
if not data: exit()

def get_winner(h, a, ratings, mode, neutral=False):
    h_adv = data['home_adv'] if not neutral else 1.0
    
    if mode == 'classic':
        att_h, def_h = ratings[h]['att'], ratings[h]['def']
        att_a, def_a = ratings[a]['att'], ratings[a]['def']
    else:
        att_h, def_h = ratings[h]['attack_rating'], ratings[h]['defense_rating']
        att_a, def_a = ratings[a]['attack_rating'], ratings[a]['defense_rating']

    l_h = att_h * def_a * data['league_avg'] * h_adv
    l_a = att_a * def_h * data['league_avg'] * (1/h_adv if not neutral else 1.0)
    
    # Adăugăm factorul de Randomness (Chaos) doar pentru modelul 'full'
    if mode == 'full':
        l_h *= np.random.uniform(0.9, 1.1)
        l_a *= np.random.uniform(0.9, 1.1)

    gh, ga = np.random.poisson(max(0.05, l_h)), np.random.poisson(max(0.05, l_a))
    
    if gh > ga: return h
    if ga > gh: return a
    return random.choice([h, a])

def run_tournament(mode):
    # Faza de baraj (Locurile 9-24)
    p_teams = data['standings'][8:24]
    winners_po = [get_winner(p_teams[i], p_teams[15-i], data[mode], mode) for i in range(8)]
    # Optimi
    pool = data['standings'][:8] + winners_po
    random.shuffle(pool)
    while len(pool) > 1:
        next_round = []
        for i in range(0, len(pool), 2):
            next_round.append(get_winner(pool[i], pool[i+1], data[mode], mode, neutral=(len(pool)==2)))
        pool = next_round
    return pool[0]

# Simulări
N = 50000 
print(f"Rulăm {N} simulări pentru fiecare din cele 3 modele...")
res_classic = Counter([run_tournament('classic') for _ in range(N)])
res_hybrid = Counter([run_tournament('hybrid') for _ in range(N)])
res_full = Counter([run_tournament('full') for _ in range(N)])

# Pregătire date grafic
top_teams = [t for t, _ in res_full.most_common(20)]
probs_c = [round(res_classic[t]/N * 100, 2) for t in top_teams]
probs_h = [round(res_hybrid[t]/N * 100, 2) for t in top_teams]
probs_f = [round(res_full[t]/N * 100, 2) for t in top_teams]

# Vizualizare Plotly
fig = go.Figure()
fig.add_trace(go.Bar(x=top_teams, y=probs_c, name='1. xG Clasic (Brut)', marker_color='silver'))
fig.add_trace(go.Bar(x=top_teams, y=probs_h, name='2. Hibrid (xG + Valoare)', marker_color='skyblue'))
fig.add_trace(go.Bar(x=top_teams, y=probs_f, name='3. Chaos (Full Adjusted)', marker_color='royalblue'))

fig.update_layout(
    title=f"Analiză Monte Carlo UCL (N={N}): Evoluția Probabilităților",
    xaxis_title="Echipe", yaxis_title="Probabilitate Câștig (%)",
    barmode='group', template='plotly_white', hovermode='x unified'
)
fig.show()