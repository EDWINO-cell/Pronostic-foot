"""
recalibrate.py — Recalibration automatique de rho/alpha par championnat.

Calibre sur DEUX saisons combinées (saison en cours + précédente, glissant
automatiquement d'une année à chaque exécution) plutôt qu'une seule, pour
éviter le sur-ajustement qu'on avait détecté avec la version single-saison
(des réglages qui semblaient bons sur une saison s'effondraient sur une autre).

Objectif de sélection : le marché résultat du match (1X2), validé par
backtest comme celui où le modèle a un vrai avantage — plus/moins 2.5 est
gardé à titre informatif dans les logs, mais ne pilote plus le choix des
réglages (backtesté comme non fiable, quel que soit le réglage).

Sélection "robuste" : on ne retient pas la combinaison qui excelle sur une
saison et s'effondre sur l'autre, mais celle qui minimise une moyenne
pondérée entre la performance moyenne et la pire des deux saisons (même
méthode que calibrate_robust.py). Le fichier est réécrit intégralement à
chaque exécution (pas de comparaison fragile avec l'ancien snapshot).
"""

import json
import math
import os
import sys
import time
from datetime import date
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


def current_seasons() -> list:
    """
    Les deux dernières saisons ENTIÈREMENT TERMINÉES à combiner pour la
    calibration (jamais la saison en cours, trop peu de matchs en début de
    saison pour être fiable). Glisse automatiquement d'une année à chaque
    nouvelle saison. Ex: exécuté en mars 2027 (saison 2026-27 en cours)
    -> saisons [2024, 2025] (les deux précédentes, complètes).
    """
    today = date.today()
    current_season_start = today.year if today.month >= 7 else today.year - 1
    last_completed = current_season_start - 1
    return [last_completed - 1, last_completed]


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


def fetch_season_matches(competition_code: str, season: int) -> list:
    data = api_get(f"competitions/{competition_code}/matches", {"season": season, "status": "FINISHED"})
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


def outcome_probabilities(lambda_home: float, lambda_away: float, rho: float) -> tuple:
    matrix = build_score_matrix(lambda_home, lambda_away, rho)
    p_home = sum(p for (h, a), p in matrix.items() if h > a)
    p_draw = sum(p for (h, a), p in matrix.items() if h == a)
    p_away = sum(p for (h, a), p in matrix.items() if h < a)
    return p_home, p_draw, p_away


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
    précèdent). Retourne le Brier score sur le 1X2 (objectif de sélection,
    validé par backtest) et sur plus/moins 2.5 (informatif seulement).
    """
    brier_1x2_sum = 0.0
    brier_over_sum = 0.0
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

        p_home, p_draw, p_away = outcome_probabilities(lambda_home, lambda_away, rho)
        actual_home = 1.0 if real_h > real_a else 0.0
        actual_draw = 1.0 if real_h == real_a else 0.0
        actual_away = 1.0 if real_h < real_a else 0.0
        brier_1x2_sum += (p_home - actual_home) ** 2 + (p_draw - actual_draw) ** 2 + (p_away - actual_away) ** 2

        pred_over = over_2_5_probability(lambda_home, lambda_away, rho)
        actual_over = 1.0 if (real_h + real_a) > 2.5 else 0.0
        brier_over_sum += (pred_over - actual_over) ** 2

        n_evaluated += 1

    if n_evaluated == 0:
        return {"brier_1x2": None, "brier_over": None, "n": 0}
    return {"brier_1x2": brier_1x2_sum / n_evaluated, "brier_over": brier_over_sum / n_evaluated, "n": n_evaluated}


def naive_baseline_brier_1x2(matches: list) -> float:
    """Prédiction naïve 1X2 : probabilités fixes = fréquences historiques de la saison."""
    outcomes = []
    for m in matches:
        score = m.get("score", {}).get("fullTime", {})
        h, a = score.get("home"), score.get("away")
        if h is None or a is None:
            continue
        outcomes.append((1.0, 0.0, 0.0) if h > a else (0.0, 0.0, 1.0) if h < a else (0.0, 1.0, 0.0))
    if not outcomes:
        return None
    n = len(outcomes)
    freq_h = sum(o[0] for o in outcomes) / n
    freq_d = sum(o[1] for o in outcomes) / n
    freq_a = sum(o[2] for o in outcomes) / n
    return sum((freq_h - h) ** 2 + (freq_d - d) ** 2 + (freq_a - a) ** 2 for h, d, a in outcomes) / n


def calibrate_league(competition_code: str, seasons: list) -> dict:
    print(f"\n=== {competition_code} ===")
    matches_by_season = {}
    for season in seasons:
        m = fetch_season_matches(competition_code, season)
        print(f"  Saison {season} : {len(m)} matchs récupérés")
        if m:
            matches_by_season[season] = m

    if len(matches_by_season) < len(seasons):
        print("  Pas assez de saisons disponibles, championnat ignoré.")
        return None

    best = None
    for rho in RHO_GRID:
        for alpha in ALPHA_GRID:
            season_scores = {}
            for season, matches in matches_by_season.items():
                r = evaluate_settings(matches, rho, alpha)
                if r["brier_1x2"] is None:
                    continue
                season_scores[season] = r
            if len(season_scores) < len(matches_by_season):
                continue

            briers_1x2 = [r["brier_1x2"] for r in season_scores.values()]
            avg_score = sum(briers_1x2) / len(briers_1x2)
            worst_score = max(briers_1x2)
            objective = avg_score * 0.6 + worst_score * 0.4

            if best is None or objective < best["objective"]:
                best = {
                    "rho": rho, "alpha": alpha, "objective": objective,
                    "avg_brier_1x2": avg_score, "worst_brier_1x2": worst_score,
                    "season_scores": season_scores,
                }

    if best is None:
        print("  Pas assez de données pour calibrer, championnat ignoré.")
        return None

    naive_scores = [naive_baseline_brier_1x2(m) for m in matches_by_season.values()]
    naive_scores = [n for n in naive_scores if n is not None]
    naive_avg = sum(naive_scores) / len(naive_scores) if naive_scores else None
    beats_naive = naive_avg is not None and best["avg_brier_1x2"] < naive_avg

    print(f"  Meilleur réglage robuste : rho={best['rho']}, alpha={best['alpha']}")
    for season, r in sorted(best["season_scores"].items()):
        print(f"    Saison {season} : Brier 1X2={r['brier_1x2']:.4f}  (plus/moins 2.5, informatif : {r['brier_over']:.4f})")
    naive_str = f"{naive_avg:.4f}" if naive_avg is not None else "n/a"
    print(f"  Moyenne 1X2={best['avg_brier_1x2']:.4f}  Pire saison={best['worst_brier_1x2']:.4f}  "
          f"naïf={naive_str}  (bat le naïf : {'OUI' if beats_naive else 'NON'})")

    return {
        "rho": best["rho"], "alpha": best["alpha"],
        "brier_1x2": best["avg_brier_1x2"], "beats_naive": beats_naive,
        "seasons_used": seasons,
    }


def main():
    seasons = current_seasons()
    print(f"Recalibration robuste — saisons combinées : {seasons}")

    updated_settings = {}
    changes = []

    for code in COMPETITIONS:
        new = calibrate_league(code, seasons)
        if new is None:
            continue
        updated_settings[code] = {
            "rho": new["rho"], "alpha": new["alpha"],
            "brier_1x2": new["brier_1x2"], "beats_naive": new["beats_naive"],
            "seasons_used": new["seasons_used"],
        }
        changes.append(code)

    SETTINGS_PATH.write_text(json.dumps(updated_settings, indent=2))

    print("\n=== Résumé ===")
    if changes:
        print(f"Réglages calibrés (saisons {seasons}) pour : {', '.join(changes)}")
    else:
        print("Aucun championnat n'a pu être calibré, league_settings.json inchangé.")


if __name__ == "__main__":
    main()
    
