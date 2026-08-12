"""
Bloc à ajouter à app.py — logging automatique des prédictions via l'API GitHub.

Où l'insérer : juste après l'appel à run_prediction_for_teams(team1, team2)
dans le code qui gère le clic sur "Analyser", avant l'affichage du résultat.
Nécessite le fichier github_log.py dans le même repo, et le secret
GITHUB_TOKEN dans les Secrets Streamlit Cloud.
"""

import datetime

from github_log import append_rows


def log_prediction(team1: dict, team2: dict, scheduled_match, prediction: dict) -> None:
    """
    Construit une ligne par marché prédit (résultat 1N2, over/under 2.5, BTTS,
    score exact) pour le match analysé, et les envoie à github_log.append_rows.
    On garde volontairement une ligne par analyse (pas de déduplication) pour
    conserver l'historique de chaque clic sur "Analyser".
    """
    match_date = scheduled_match["utcDate"] if scheduled_match else ""
    logged_at = datetime.datetime.utcnow().isoformat()

    rows = []

    def make_row(market: str, value: str, probability) -> dict:
        return {
            "log_id": f"{team1['id']}-{team2['id']}-{market}-{logged_at}",
            "logged_at": logged_at,
            "match_date": match_date,
            "competition_code": team1["competition_code"],
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

    # Adapte les clés ci-dessous au dict réellement renvoyé par
    # run_prediction_for_teams (résultat 1N2, over/under, BTTS, score exact).
    if "result_1n2" in prediction:
        rows.append(make_row("1N2", prediction["result_1n2"]["value"], prediction["result_1n2"]["probability"]))
    if "over_under_2_5" in prediction:
        rows.append(make_row("over_under_2.5", prediction["over_under_2_5"]["value"], prediction["over_under_2_5"]["probability"]))
    if "btts" in prediction:
        rows.append(make_row("btts", prediction["btts"]["value"], prediction["btts"]["probability"]))
    if "top_scores" in prediction and prediction["top_scores"]:
        top = prediction["top_scores"][0]
        rows.append(make_row("score_exact_top1", top["score"], top["probability"]))

    if rows:
        append_rows(rows)
      
