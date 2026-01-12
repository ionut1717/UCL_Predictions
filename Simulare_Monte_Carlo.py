import numpy as np
import random
import plotly.graph_objects as go
from collections import Counter
from Parsare_Meciuri import parse_and_calculate_ratings

# 1. Încărcăm datele folosind ambele fișiere
# Asigură-te că fișierele au aceste nume în folderul tău
FILE_MECIURI = "Meciuri_UCL_2024-2025.csv"
FILE_PRIORS = "Echipe_2024-2025.csv"

data = parse_and_calculate_ratings(FILE_MECIURI, FILE_PRIORS)

if not data: 
    print("Eroare: Nu s-au putut încărca datele. Verifică numele fișierelor CSV.")
    exit()

def get_winner(h, a, ratings, mode, neutral=False):
    """
    Simulează un singur meci bazat pe distribuția Poisson.
    Modelele (Classic/Adjusted) sunt deja pre-calculate în Parsare_Meciuri.
    """
    h_adv = data['home_adv'] if not neutral else 1.0
    
    # Extragere parametri în funcție de modelul ales
    if mode == 'classic':
        att_h, def_h = ratings[h]['att'], ratings[h]['def']
        att_a, def_a = ratings[a]['att'], ratings[a]['def']
    else:
        # Modelul Adjusted conține acum și Power Factor (Market Value + Coeff)
        att_h, def_h = ratings[h]['attack_rating'], ratings[h]['defense_rating']
        att_a, def_a = ratings[a]['attack_rating'], ratings[a]['defense_rating']

    # Calcul Lambda (media de goluri așteptate)
    # Am setat un prag de siguranță (min/max) pentru a evita rezultate aberante
    l_h = max(0.05, min(3.5, att_h * def_a * data['league_avg'] * h_adv))
    l_a = max(0.05, min(3.5, att_a * def_h * data['league_avg'] * (1/h_adv if not neutral else 1.0)))
    
    # Generare goluri (Eveniment unic)
    gh = np.random.poisson(l_h)
    ga = np.random.poisson(l_a)
    
    if gh > ga: return h
    if ga > gh: return a
    return random.choice([h, a]) # Departajare la penalty-uri

def run_tournament(mode):
    """Simulează un întreg parcurs de turneu de la Baraj la Finală."""
    # Etapa 1: Baraj (Locurile 9-24 conform clasamentului calculat)
    p_teams = data['standings'][8:24]
    winners_po = [get_winner(p_teams[i], p_teams[15-i], data[mode], mode) for i in range(8)]
    
    # Etapa 2: Optimi (Top 8 + Câștigători Baraj)
    pool = data['standings'][:8] + winners_po
    random.shuffle(pool) # Randomizarea traseului pentru a testa capabilitatea pură
    
    # Etapele eliminatorii (Optimi -> Sferturi -> Semifinale -> Finală)
    while len(pool) > 1:
        next_round = []
        for i in range(0, len(pool), 2):
            is_final = len(pool) == 2
            # Finala se joacă pe teren neutru (neutral=True)
            next_round.append(get_winner(pool[i], pool[i+1], data[mode], mode, neutral=is_final))
        pool = next_round
    return pool[0]

# --- Simulare Monte Carlo ---
N = 50000 
print(f"Pornire simulare: {N} iterații per model...")

# Rulăm simulările pentru ambele perspective
results_c = Counter([run_tournament('classic') for _ in range(N)])
results_a = Counter([run_tournament('adjusted') for _ in range(N)])

# --- Pregătire Vizualizare ---
# Sortăm toate echipele care au câștigat cel puțin o dată, descrescător după modelul Adjusted
all_winners = [t for t, _ in results_a.most_common()]

# Limităm la top 20 pentru ca graficul să rămână lizibil, dar cuprinzător
top_n = min(20, len(all_winners))
top_teams = all_winners[:top_n]

c_probs = [round(results_c[t]/N * 100, 2) for t in top_teams]
a_probs = [round(results_a[t]/N * 100, 2) for t in top_teams]

# --- Creare Grafic Plotly ---
fig = go.Figure()

fig.add_trace(go.Bar(
    x=top_teams,
    y=c_probs,
    name='Model xG Clasic',
    marker_color='lightgrey',
    hovertemplate='<b>%{x}</b><br>Probabilitate: %{y}%<extra></extra>'
))

fig.add_trace(go.Bar(
    x=top_teams,
    y=a_probs,
    name='Model Hibrid (xG + Valoare Lot)',
    marker_color='skyblue',
    hovertemplate='<b>%{x}</b><br>Probabilitate: %{y}%<extra></extra>'
))

fig.update_layout(
    title=f'Analiză Monte Carlo UCL: Șanse de Câștig (N={N})<br><sup>Comparație între performanța brută și valoarea structurală a lotului</sup>',
    xaxis_title='Echipe',
    yaxis_title='Probabilitate de Câștig (%)',
    barmode='group',
    hovermode='x unified',
    template='plotly_white',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.show()