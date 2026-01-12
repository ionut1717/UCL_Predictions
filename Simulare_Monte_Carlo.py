import numpy as np
import random
import plotly.graph_objects as go
from collections import Counter
from Parsare_Meciuri import parse_and_calculate_ratings

# 1. Încărcăm datele din scriptul de parsare
data = parse_and_calculate_ratings("Meciuri_UCL_2025-2026.csv")
if not data: 
    print("Eroare: Nu s-au putut încărca datele.")
    exit()

def get_winner(h, a, ratings, mode, neutral=False):
    h_adv = data['home_adv'] if not neutral else 1.0
    
    if mode == 'classic':
        att_h, def_h = ratings[h]['att'], ratings[h]['def']
        att_a, def_a = ratings[a]['att'], ratings[a]['def']
    else:
        att_h, def_h = ratings[h]['attack_rating'], ratings[h]['defense_rating']
        att_a, def_a = ratings[a]['attack_rating'], ratings[a]['defense_rating']

    l_h = min(3.5, att_h * def_a * data['league_avg'] * h_adv)
    l_a = min(3.5, att_a * def_h * data['league_avg'] * (1/h_adv if not neutral else 1.0))
    
    gh = np.random.poisson(l_h)
    ga = np.random.poisson(l_a)
    
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

# --- Simulare Monte Carlo ---
N = 50000 
print(f"Se execută {N} simulări pentru ambele modele...")
results_c = Counter([run_tournament('classic') for _ in range(N)])
results_a = Counter([run_tournament('adjusted') for _ in range(N)])

# --- Pregătire Date pentru Plotly ---
# Luăm top 15 echipe bazat pe modelul ajustat
# 'len(results_a)' va lua automat toate echipele care au câștigat cel puțin o dată
top_teams = [t for t, _ in results_a.most_common(len(results_a))]
c_probs = [round(results_c[t]/N * 100, 2) for t in top_teams]
a_probs = [round(results_a[t]/N * 100, 2) for t in top_teams]

# --- Creare Grafic Interactiv cu Bare Verticale ---
fig = go.Figure()

# Adăugăm barele pentru xG Clasic
fig.add_trace(go.Bar(
    x=top_teams,
    y=c_probs,
    name='xG Clasic',
    marker_color='lightgrey',
    hovertemplate='Echipa: %{x}<br>Probabilitate: %{y}%<extra></extra>'
))

# Adăugăm barele pentru xG Ajustat
fig.add_trace(go.Bar(
    x=top_teams,
    y=a_probs,
    name='xG Ajustat',
    marker_color='skyblue',
    hovertemplate='Echipa: %{x}<br>Probabilitate: %{y}%<extra></extra>'
))

# Personalizare aspect (Layout)
fig.update_layout(
    title=f'Probabilități Câștig UCL - Simulare Monte Carlo (N={N})',
    xaxis_title='Echipe',
    yaxis_title='Probabilitate Câștig (%)',
    barmode='group', # Barele stau una lângă alta
    hovermode='x unified', # Arată ambele valori când mouse-ul e pe coloană
    template='plotly_white',
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
)

# Afișare grafic
fig.show()

# Opțional: Salvare ca fișier HTML pentru prezentare
# fig.write_html("Simulare_UCL_Interactiv.html")