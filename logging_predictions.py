"""
Bloc à ajouter à app.py — logging automatique des prédictions via l'API GitHub.

Où l'insérer : juste après l'appel à run_prediction_for_teams(team1, team2)
(ou run_prediction(team1_name, team2_name)) dans le code qui gère le clic sur
"Analyser", avant l'affichage du résultat. Nécessite le fichier github_log.py
dans le même repo, et le secret GITHUB_TOKEN dans les Secrets Streamlit Cloud.

Adapté aux vraies clés renvoyées par run_prediction_for_teams :
home_win / draw / away_win, over_2_5, btts_yes / btts_no, top_scores.
"""

import datetime

from github_log import append_rows


def log_prediction(prediction: dict) -> None:
    team1 = prediction["team1"]
    team2 = prediction["team2"]
    logged_at = datetime.datetime.utcnow().isoformat()

    outcomes = {"1": prediction["home_win"], "N": prediction["draw"], "2": prediction["away_win"]}
    best_outcome = max(outcomes, key=outcomes.get)

    top_score = prediction["top_scores"][0] if prediction.get("top_scores") else None

    rows = []

    def make_row(market: str, value, probability) -> dict:
        return {
            "log_id": f"{team1['id']}-{team2['id']}-{market}-{logged_at}",
            "logged_at": logged_at,
            "match_date": prediction["scheduled_match"]["utcDate"] if prediction.get("scheduled_match") else "",
            "competition_code": team1.get("competition_code", ""),
            "team1_id": team1["id"], "team1_name": team1["name"],
            "team2_id": team2["id"], "team2_name": team2["name"],
            "market": market,
            "predicted_value": value,
            "predicted_probability": probability,
            "actual_checked": "false",
            "actual_result": "",
            "was_correct": "",
            "brier_error": "",
        }

    rows.append(make_row("1N2", best_outcome, outcomes[best_outcome]))
    rows.append(make_row("over_under_2.5", "over" if prediction["over_2_5"] >= 0.5 else "under",
                          max(prediction["over_2_5"], 1 - prediction["over_2_5"])))
    rows.append(make_row("btts", "oui" if prediction["btts_yes"] >= 0.5 else "non",
                          max(prediction["btts_yes"], prediction["btts_no"])))
    if top_score:
        (h, a), proba = top_score
        rows.append(make_row("score_exact_top1", f"{h}-{a}", proba))

    if rows:
        append_rows(rows)
