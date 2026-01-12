import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
from Parsare_Meciuri import parse_and_calculate_ratings

data = parse_and_calculate_ratings("Meciuri_UCL.csv")
if not data: exit()

def get_winner(h, a, ratings, mode, neutral=False):
    h_adv = data['home_adv'] if not neutral else 1.0
    # Mapăm cheile în funcție de model
    att_h = ratings[h]['att'] if mode == 'classic' else ratings[h]['attack_rating']
    def_h = ratings[h]['def'] if mode == 'classic' else ratings[h]['defense_rating']
    att_a = ratings[a]['att'] if mode == 'classic' else ratings[a]['attack_rating']
    def_a = ratings[a]['def'] if mode == 'classic' else ratings[a]['defense_rating']

    l_h = att_h * def_a * data['league_avg'] * h_adv
    l_a = att_a * def_h * data['league_avg'] * (1/h_adv if not neutral else 1.0)
    
    # Sub-simulare pentru "cel mai probabil" scor
    wins = []
    for _ in range(100):
        gh, ga = np.random.poisson(max(0.1, l_h)), np.random.poisson(max(0.1, l_a))
        wins.append(h if gh > ga else (a if ga > gh else random.choice([h, a])))
    return Counter(wins).most_common(1)[0][0]

def run_tournament(mode):
    # Baraj
    p_teams = data['standings'][8:24]
    winners_po = [get_winner(p_teams[i], p_teams[15-i], data[mode], mode) for i in range(8)]
    
    # Optimi (Randomizate)
    pool = data['standings'][:8] + winners_po
    random.shuffle(pool)
    
    # Fazele eliminatorii
    while len(pool) > 1:
        next_round = []
        for i in range(0, len(pool), 2):
            is_final = len(pool) == 2
            next_round.append(get_winner(pool[i], pool[i+1], data[mode], mode, neutral=is_final))
        pool = next_round
    return pool[0]

# --- Simulare Monte Carlo ---
N = 50000
results_c = Counter([run_tournament('classic') for _ in range(N)])
results_a = Counter([run_tournament('adjusted') for _ in range(N)])

# --- Vizualizare ---
top_teams = [t for t, _ in results_a.most_common(15)]
c_probs = [results_c[t]/N * 100 for t in top_teams]
a_probs = [results_a[t]/N * 100 for t in top_teams]

y = np.arange(len(top_teams))
plt.figure(figsize=(12, 8))
plt.barh(y - 0.2, c_probs, 0.4, label='xG Clasic', color='silver')
plt.barh(y + 0.2, a_probs, 0.4, label='xG Ajustat', color='skyblue')
plt.yticks(y, top_teams)
plt.xlabel('Probabilitate Câștig (%)')
plt.title(f'Simulare Monte Carlo UCL (N={N}): Impactul Dificultății Adversarilor')
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()