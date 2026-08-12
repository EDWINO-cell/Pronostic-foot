"""
check_results.py — Vérifie les prédictions loggées (predictions_log.csv) dont
le match est maintenant terminé, récupère le score réel, et met à jour les
colonnes actual_checked / actual_result / was_correct / brier_error.

Tourne via GitHub Action (accès direct git commit/push, pas besoin d'API
GitHub côté contenu comme github_log.py — ce script s'exécute dans un
checkout normal du repo).
"""

import csv
import os
import sys
import time

import requests

API_BASE_URL = "https://api.football-data.org/v4"
LOG_PATH = "predictions_log.csv"


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
        time.sleep(6.5)
        return response.json()
    raise RuntimeError(f"Échec après plusieurs tentatives sur {endpoint}.")


def find_finished_match(team1_id: int, team2_id: int, match_date: str):
    """Cherche le match terminé entre les deux équipes autour de la date loggée."""
    data = api_get(f"teams/{team1_id}/matches", {"status": "FINISHED"})
    for m in data.get("matches", []):
        if m["awayTeam"]["id"] == team2_id or m["homeTeam"]["id"] == team2_id:
            if m["utcDate"][:10] == match_date[:10]:
                return m
    return None


def evaluate_row(row: dict, match: dict) -> dict:
    score = match.get("score", {}).get("fullTime", {})
    h, a = score.get("home"), score.get("away")
    if h is None or a is None:
        return row  # pas encore de score complet malgré status FINISHED (rare, edge case)

    total_goals = h + a
    market = row["market"]
    predicted = row["predicted_value"]

    if market == "1N2":
        actual = "1" if h > a else ("2" if a > h else "N")
        # predicted_value peut être "1", "N", "2" ou une combinaison type "1N" (double chance)
        correct = actual in predicted
    elif market == "over_under_2.5":
        actual = "over" if total_goals > 2.5 else "under"
        correct = actual == predicted.lower()
    elif market == "btts":
        actual = "oui" if (h > 0 and a > 0) else "non"
        correct = actual == predicted.lower()
    elif market == "score_exact_top1":
        actual = f"{h}-{a}"
        correct = actual == predicted
    else:
        return row

    probability = float(row["predicted_probability"]) if row["predicted_probability"] else None
    brier = None
    if probability is not None:
        brier = (probability - (1.0 if correct else 0.0)) ** 2

    row["actual_checked"] = "true"
    row["actual_result"] = actual if market != "score_exact_top1" else f"{h}-{a}"
    row["was_correct"] = "true" if correct else "false"
    row["brier_error"] = f"{brier:.4f}" if brier is not None else ""
    return row


def main():
    if not os.path.exists(LOG_PATH):
        print("Aucun predictions_log.csv trouvé, rien à vérifier.")
        return

    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pending = [r for r in rows if r["actual_checked"] != "true"]
    print(f"{len(pending)} lignes en attente de vérification sur {len(rows)} au total.")

    # Cache des matchs déjà récupérés pour cette paire d'équipes, pour éviter
    # de refaire un appel API par ligne quand plusieurs marchés du même match
    # sont en attente.
    match_cache = {}
    updated_count = 0

    for row in rows:
        if row["actual_checked"] == "true":
            continue
        if not row["match_date"]:
            continue

        key = (row["team1_id"], row["team2_id"], row["match_date"][:10])
        if key not in match_cache:
            try:
                match_cache[key] = find_finished_match(int(row["team1_id"]), int(row["team2_id"]), row["match_date"])
            except RuntimeError as e:
                print(f"Erreur API pour {key} : {e}")
                match_cache[key] = None

        match = match_cache[key]
        if match is None:
            continue  # match pas encore joué, ou pas trouvé

        evaluate_row(row, match)
        updated_count += 1

    if updated_count > 0:
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"{updated_count} lignes mises à jour.")
    else:
        print("Aucune mise à jour cette fois-ci.")


if __name__ == "__main__":
    main()
          
