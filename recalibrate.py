"""
recalibrate.py — Recalibration automatique de rho/alpha par championnat.

Reprend la méthodologie de calibrate_per_league.py (grid search rho x alpha,
évaluation par Brier score sur plus/moins 2.5, comparaison vs prédiction
naïve) mais en continu : à lancer périodiquement (GitHub Action mensuelle)
pour ré-ajuster league_settings.json à partir des matchs les plus récents.

Ne remplace les valeurs en place que si la nouvelle calibration fait
strictement mieux (Brier score plus bas) — sinon on garde l'ancien réglage,
pour éviter qu'un petit échantillon récent ne dérègle le modèle.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import requests

API_BASE_URL = "https://api.football-data.org/v4"
REQUEST_DELAY_SECONDS = 6.5
MAX_GOALS = 6
SETTINGS_PATH = Path("league_settings.json")

COMPETITIONS = ["PL", "PD", "BL1", "SA", "FL1", "DED", "PPL", "ELC", "BSA"]

# Même grille que calibrate_per_league.py (16 combinaisons par championnat).
RHO_GRID = [-0.18, -0.13, -0.05, 0.00]
ALPHA_GRID = [0.4, 0.6, 0.8, 1.0]

MIN_MATCHES_MIXED = 6
MIN_MATCHES_VENUE = 4


def get_api_key() -> str:
    key = os.environ.get("FOOTBALL_DATA_KEY")
    if not key:
        sys.exit("Erreur : FOOTBALL_DATA_KEY manquant dans l'environnement.")
    return key


def api_get(endpoint: str, params: dict | None = None) -> dict:
    headers = {"X-Auth-Token": get_api_key()}
    for attempt in range(4):
        response = requests.get(f"{API_BASE_URL}/{endpoint}", headers=headers, params=params or {}, timeout=15)
        if response.status_code == 429:
            time.sleep(30)
            continue
        if response.status_code != 200:
            raise RuntimeError(f"Erreur API ({response.status_code}) sur {endpoint} : {response.text}")
        time.sleep(REQUEST_DELAY_SECONDS)
        return response.json()
    raise RuntimeError(f"Échec après plusieurs tentatives sur {endpoint}.")


def fetch_season_matches(competition_code: str) -> list:
    data = api_get(f"competitions/{competition_code}/matches", {"status": "FINISHED"})
    matches = data.get("matches", [])
    matches.sort(key=lambda m: m["utcDate"])
    return matches


def poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def dixon_coles_tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    if h == 0 and a == 0:
        return 1 - (lh * la * rho)
    elif h == 0 and a == 1:
        return 1 + (lh * rho)
    elif h == 1 and a == 0:
        return 1 + (la * rho)
    elif h == 1 and a == 1:
        return 1 - rho
    return 1.0


def build_score_matrix(lambda_home: float, lambda_away: float, rho: float) -> dict:
    matrix = {}
    total = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = poisson_pmf(h, lambda_home) * poisson_pmf(a, lambda_away) * dixon_coles_tau(h, a, lambda_home, lambda_away, rho)
            p = max(p, 0.0)
            matrix[(h, a)] = p
            total += p
    for key in matrix:
        matrix[key] /= total
    return matrix


def over_2_5_probability(lambda_home: float, lambda_away: float, rho: float) -> float:
    matrix = build_score_matrix(lambda_home, lambda_away, rho)
    return sum(p for (h, a), p in matrix.items() if h + a > 2.5)


def team_form_before(matches: list, team_id: int, up_to_index: int, home_only: bool | None) -> dict:
    """
    Forme d'une équipe calculée uniquement à partir des matchs AVANT l'index donné
    (walk-forward, pas de fuite de données — même logique que le modèle basket).
    home_only=True -> matchs à domicile seulement, False -> extérieur seulement,
    None -> mixte (domicile + extérieur).
    """
    relevant = []
    for m in matches[:up_to_index]:
        is_home = m["homeTeam"]["id"] == team_id
        is_away = m["awayTeam"]["id"] == team_id
        if not (is_home or is_away):
            continue
        if home_only is True and not is_home:
            continue
        if home_only is False and not is_away:
            continue
        relevant.append((m, is_home))

    relevant = relevant[-6:]
    scored, conceded, played = 0, 0, 0
    for m, is_home in relevant:
        score = m.get("score", {}).get("fullTime", {})
        gf = score.get("home") if is_home else score.get("away")
        ga = score.get("away") if is_home else score.get("home")
        if gf is None or ga is None:
            continue
        scored += gf
        conceded += ga
        played += 1

    if played == 0:
        return {"avg_scored": 1.2 if home_only is None else None, "avg_conceded": 1.2 if home_only is None else None, "played": 0}
    return {"avg_scored": scored / played, "avg_conceded": conceded / played, "played": played}


def blended_value(mixed_value: float, venue_value, venue_played: int, alpha: float) -> float:
    if venue_value is None or venue_played < MIN_MATCHES_VENUE:
        return mixed_value
    return alpha * venue_value + (1 - alpha) * mixed_value


def evaluate_settings(matches: list, rho: float, alpha: float) -> dict:
    """
    Backteste un couple (rho, alpha) sur tous les matchs de la saison en
    walk-forward (chaque match prédit uniquement avec les matchs qui le
    précèdent). Retourne le Brier score sur plus/moins 2.5 et le nombre de
    matchs utilisés.
    """
    brier_sum = 0.0
    n_evaluated = 0

    for i, m in enumerate(matches):
        score = m.get("score", {}).get("fullTime", {})
        real_h, real_a = score.get("home"), score.get("away")
        if real_h is None or real_a is None:
            continue

        home_id = m["homeTeam"]["id"]
        away_id = m["awayTeam"]["id"]

        mixed_home = team_form_before(matches, home_id, i, None)
        mixed_away = team_form_before(matches, away_id, i, None)
        if mixed_home["played"] < MIN_MATCHES_MIXED or mixed_away["played"] < MIN_MATCHES_MIXED:
            continue

        home_form = team_form_before(matches, home_id, i, True)
        away_form = team_form_before(matches, away_id, i, False)

        lambda_home = blended_value(mixed_home["avg_scored"], home_form["avg_scored"], home_form["played"], alpha)
        lambda_away = blended_value(mixed_away["avg_scored"], away_form["avg_scored"], away_form["played"], alpha)

        pred_over = over_2_5_probability(lambda_home, lambda_away, rho)
        actual_over = 1.0 if (real_h + real_a) > 2.5 else 0.0

        brier_sum += (pred_over - actual_over) ** 2
        n_evaluated += 1

    if n_evaluated == 0:
        return {"brier": None, "n": 0}
    return {"brier": brier_sum / n_evaluated, "n": n_evaluated}


def naive_baseline_brier(matches: list) -> float:
    """Prédiction naïve : probabilité fixe = fréquence historique de over 2.5 sur la saison."""
    overs = []
    for m in matches:
        score = m.get("score", {}).get("fullTime", {})
        h, a = score.get("home"), score.get("away")
        if h is None or a is None:
            continue
        overs.append(1.0 if (h + a) > 2.5 else 0.0)
    if not overs:
        return None
    freq = sum(overs) / len(overs)
    return sum((freq - o) ** 2 for o in overs) / len(overs)


def calibrate_league(competition_code: str) -> dict:
    print(f"\n=== {competition_code} ===")
    matches = fetch_season_matches(competition_code)
    print(f"{len(matches)} matchs récupérés")

    baseline = naive_baseline_brier(matches)

    best = None
    for rho in RHO_GRID:
        for alpha in ALPHA_GRID:
            result = evaluate_settings(matches, rho, alpha)
            if result["brier"] is None:
                continue
            if best is None or result["brier"] < best["brier"]:
                best = {"rho": rho, "alpha": alpha, "brier": result["brier"], "n": result["n"]}

    if best is None:
        print("Pas assez de données pour calibrer, championnat ignoré.")
        return None

    beats_naive = baseline is not None and best["brier"] < baseline
    baseline_str = f"{baseline:.4f}" if baseline is not None else "n/a"
    print(f"Meilleur réglage : rho={best['rho']}, alpha={best['alpha']} "
          f"(Brier={best['brier']:.4f}, naïf={baseline_str}, "
          f"bat le naïf : {'OUI' if beats_naive else 'NON'})")

    return {"rho": best["rho"], "alpha": best["alpha"], "brier": best["brier"], "beats_naive": beats_naive}


def main():
    if SETTINGS_PATH.exists():
        current_settings = json.loads(SETTINGS_PATH.read_text())
    else:
        current_settings = {}

    updated_settings = dict(current_settings)
    changes = []

    for code in COMPETITIONS:
        new = calibrate_league(code)
        if new is None:
            continue

        old = current_settings.get(code)
        # On ne remplace que si on a une comparaison possible et que c'est mieux,
        # ou si on n'avait encore aucun réglage pour ce championnat.
        if old is None or new["brier"] < old.get("brier", float("inf")):
            updated_settings[code] = {"rho": new["rho"], "alpha": new["alpha"], "brier": new["brier"]}
            changes.append(code)

    SETTINGS_PATH.write_text(json.dumps(updated_settings, indent=2))

    print("\n=== Résumé ===")
    if changes:
        print(f"Réglages mis à jour pour : {', '.join(changes)}")
    else:
        print("Aucun réglage n'a été amélioré, league_settings.json inchangé.")


if __name__ == "__main__":
    main()
