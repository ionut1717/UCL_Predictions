import numpy as np
import random
import plotly.graph_objects as go
from collections import Counter
from Parsare_Meciuri import parse_and_calculate_ratings

# 1. Încărcăm datele (Asigură-te că numele fișierelor sunt corecte)
FILE_MECIURI = "Meciuri_UCL_2025-2026.csv"
FILE_PRIORS = "Echipe_2025-2026.csv"

data = parse_and_calculate_ratings(FILE_MECIURI, FILE_PRIORS)
if not data: 
    print("Eroare la încărcarea datelor.")
    exit()

def get_winner(h, a, ratings, mode, neutral=False):
    """
    Simulează un meci unic. 
    Modelele Hybrid și Full folosesc rating-uri ajustate meci-cu-meci.
    """
    h_adv = data['home_adv'] if not neutral else 1.0
    
    # Extragere parametri
    if mode == 'classic':
        att_h, def_h = ratings[h]['att'], ratings[h]['def']
        att_a, def_a = ratings[a]['att'], ratings[a]['def']
    else:
        # Pentru Hybrid și Full, cheile sunt 'attack_rating' și 'defense_rating'
        att_h, def_h = ratings[h]['attack_rating'], ratings[h]['defense_rating']
        att_a, def_a = ratings[a]['attack_rating'], ratings[a]['defense_rating']

    # Calcul Lambda (Media de goluri)
    l_h = att_h * def_a * data['league_avg'] * h_adv
    l_a = att_a * def_h * data['league_avg'] * (1/h_adv if not neutral else 1.0)
    
    # --- FACTORUL DE CHAOS (Doar pentru modelul Full) ---
    # Simulăm evenimente neprevăzute (cartonașe, erori) +/- 15% varianță
    if mode == 'full':
        l_h *= np.random.uniform(0.85, 1.15)
        l_a *= np.random.uniform(0.85, 1.15)

    # Generare goluri (Poisson) - Adăugăm max(0.01) pentru a evita erori matematice
    gh = np.random.poisson(max(0.01, l_h))
    ga = np.random.poisson(max(0.01, l_a))
    
    if gh > ga: return h
    if ga > gh: return a
    
    # Departajare la penalty-uri (50/50)
    return random.choice([h, a])

def run_tournament(mode):

    p_teams = data['standings'][8:24]
    winners_po = [get_winner(p_teams[i], p_teams[15-i], data[mode], mode) for i in range(8)]
    
    # Faza optimilor (Top 8 + Câștigători Baraj)
    pool = data['standings'][:8] + winners_po
    random.shuffle(pool) 
    
    # Eliminări succesive până la finală
    while len(pool) > 1:
        next_round = []
        for i in range(0, len(pool), 2):
            # Finala este ultimul meci din listă când len(pool) == 2
            is_final = len(pool) == 2
            next_round.append(get_winner(pool[i], pool[i+1], data[mode], mode, neutral=is_final))
        pool = next_round
    return pool[0]

# --- EXECUȚIE SIMULARE MONTE CARLO ---
N = 50000 
print(f"Pornire simulare: {N} iterații pentru fiecare din cele 3 modele...")

res_classic = Counter([run_tournament('classic') for _ in range(N)])
res_hybrid = Counter([run_tournament('hybrid') for _ in range(N)])
res_full = Counter([run_tournament('full') for _ in range(N)])


top_teams = [t for t, _ in res_full.most_common(16)]
probs_c = [round(res_classic[t]/N * 100, 2) for t in top_teams]
probs_h = [round(res_hybrid[t]/N * 100, 2) for t in top_teams]
probs_f = [round(res_full[t]/N * 100, 2) for t in top_teams]

# --- VIZUALIZARE INTERACTIVĂ (PLOTLY) ---
fig = go.Figure()

fig.add_trace(go.Bar(x=top_teams, y=probs_c, name='1. xG Clasic', marker_color='silver'))
fig.add_trace(go.Bar(x=top_teams, y=probs_h, name='2. xG Ajustat + Valoare Echipa', marker_color='skyblue'))
fig.add_trace(go.Bar(x=top_teams, y=probs_f, name='3. Adaugare clinical factor', marker_color='royalblue'))

fig.update_layout(
    title=f"Evoluția Probabilităților UCL (N={N})<br><sup>Impactul Ajustării Meci-cu-Meci și al Eficienței</sup>",
    xaxis_title="Echipe",
    yaxis_title="Probabilitate Câștig (%)",
    barmode='group',
    template='plotly_white',
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.show()