import numpy as np
import random
import plotly.graph_objects as go
from collections import Counter
from Parsare_Meciuri import parse_and_calculate_ratings
from Z_scoreGetter import get_z_score

# Alege nivelul dorit de incredere al datelor
ERROR_MARGIN = 0.01
CONFIDENCE_LEVEL = 0.70 
# CONFIDENCE_LEVEL = 0.90 

Z_SCORE = get_z_score(CONFIDENCE_LEVEL) 
print(Z_SCORE)

# 1. Încărcăm datele pentru anul respectiv
FILE_MECIURI = "Meciuri_UCL_2025-2026.csv"
FILE_PRIORS = "Echipe_2025-2026.csv"

data = parse_and_calculate_ratings(FILE_MECIURI, FILE_PRIORS)
if not data: 
    print("Eroare la încărcarea datelor.")
    exit()

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
    
    if mode == 'full':
        l_h *= np.random.uniform(0.85, 1.15)
        l_a *= np.random.uniform(0.85, 1.15)

    gh = np.random.poisson(max(0.01, l_h))
    ga = np.random.poisson(max(0.01, l_a))
    
    if gh > ga: return h
    if ga > gh: return a
    return random.choice([h, a])

def run_tournament(mode):
    p_teams = data['standings'][8:24]
    winners_po = [get_winner(p_teams[i], p_teams[15-i], data[mode], mode) for i in range(8)]
    pool = data['standings'][:8] + winners_po
    random.shuffle(pool) 
    while len(pool) > 1:
        next_round = []
        for i in range(0, len(pool), 2):
            is_final = len(pool) == 2
            next_round.append(get_winner(pool[i], pool[i+1], data[mode], mode, neutral=is_final))
        pool = next_round
    return pool[0]

def cal_N(marja_eroare, z_score):  
    # Folosim p = 0.5 pentru varianța maximă
    p = 0.5

    n_necesar = (z_score**2 * p * (1 - p)) / (marja_eroare**2)
    
    return int(np.ceil(n_necesar))


# --- EXECUȚIE SIMULARE MONTE CARLO ---
N = 50000 
N = cal_N(ERROR_MARGIN, Z_SCORE)



print(f"Pornire simulare: {N} iterații pentru fiecare din cele 3 modele...")

res_classic = Counter([run_tournament('classic') for _ in range(N)])
res_hybrid = Counter([run_tournament('hybrid') for _ in range(N)])
res_full = Counter([run_tournament('full') for _ in range(N)])

top_teams = [t for t, _ in res_full.most_common(16)]

def calculate_margin(count, n_total):
    p = count / n_total
    if p == 0: return 0
    se = np.sqrt(p * (1 - p) / n_total)
    return se * Z_SCORE * 100 

probs_c = [round(res_classic[t]/N * 100, 2) for t in top_teams]
probs_h = [round(res_hybrid[t]/N * 100, 2) for t in top_teams]
probs_f = [round(res_full[t]/N * 100, 2) for t in top_teams]

# Calculăm erorile
errors_c = [calculate_margin(res_classic[t], N) for t in top_teams]
errors_h = [calculate_margin(res_hybrid[t], N) for t in top_teams]
errors_f = [calculate_margin(res_full[t], N) for t in top_teams]

# --- VIZUALIZARE INTERACTIVĂ (PLOTLY) ---
fig = go.Figure()

fig.add_trace(go.Bar(
    x=top_teams, y=probs_c, name='1. xG Clasic', marker_color='silver',
    error_y=dict(type='data', array=errors_c, visible=True)
))

fig.add_trace(go.Bar(
    x=top_teams, y=probs_h, name='2. xG Ajustat + Valoare', marker_color='skyblue',
    error_y=dict(type='data', array=errors_h, visible=True)
))

fig.add_trace(go.Bar(
    x=top_teams, y=probs_f, name='3. Chaos + Clinical', marker_color='royalblue',
    error_y=dict(type='data', array=errors_f, visible=True)
))


max_theoretical_error = (np.sqrt(0.25/N) * Z_SCORE * 100)

fig.update_layout(
    title=f"Evoluția Probabilităților UCL (N={N})<br><sup>Marja de eroare statistică (interval {CONFIDENCE_LEVEL*100:.0f}%): ±{max_theoretical_error:.2f}%</sup>",
    xaxis_title="Echipe",
    yaxis_title="Probabilitate Câștig (%)",
    barmode='group',
    template='plotly_white',
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.show()