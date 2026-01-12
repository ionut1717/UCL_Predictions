def get_z_score(confidence_level):
    """
    Calculează Z-Score pentru un nivel de încredere dat.
    
    Formula:
    1. Alpha (eroarea) = 1 - confidence_level
    2. Deoarece distribuția e simetrică (două cozi), ne interesează 1 - (alpha / 2)
    3. Folosim funcția PPF (Percent Point Function) pentru a găsi Z.
    
    Args:
        confidence_level (float): Valoare între 0 și 1 (ex: 0.95 pentru 95%)
        
    Returns:
        float: Valoarea Z-Score (ex: 1.96)
    """
    
    # 1. Verificare rapidă pentru valorile standard (fără a importa biblioteci grele)
    # Asta face codul foarte rapid pentru cazurile uzuale.
    standard_map = {
        0.90: 1.645,
        0.95: 1.960,
        0.98: 2.326,
        0.99: 2.576,
        0.999: 3.291
    }
    
    if confidence_level in standard_map:
        return standard_map[confidence_level]

    # 2. Calcul exact pentru valori atipice (ex: 0.92 sau 0.995)
    try:
        from scipy.stats import norm
    except ImportError:
        print("\n[!] Eroare: Pentru valori atipice (non-standard), ai nevoie de biblioteca 'scipy'.")
        print("Instalează-o rulând: pip install scipy\n")
        return None

    alpha = 1 - confidence_level
    z_score = norm.ppf(1 - (alpha / 2))
    
    return z_score