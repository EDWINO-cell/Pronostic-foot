"""
Logging automatique des prédictions via l'API GitHub (fichier github_log.py + secret
GITHUB_TOKEN dans les Secrets Streamlit Cloud).

- log_prediction(prediction)            : prédictions du modèle statistique seul
                                          (marchés 1N2, over_under_2.5, btts, score_exact_top1).
- log_adjusted_prediction(prediction, adjusted) : prédiction 1N2 "modèle + contexte Gemini",
                                          sous le marché distinct "1N2_contexte", pour comparer
                                          en forward-test le modèle seul au modèle ajusté.
"""

import datetime

from github_log import append_rows

ADJUSTED_MARKET = "1N2_contexte"


def _make_row(prediction: dict, logged_at: str, market: str, value, probability) -> dict:
    team1 = prediction["team1"]
    team2 = prediction["team2"]
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


def log_prediction(prediction: dict) -> None:
    logged_at = datetime.datetime.utcnow().isoformat()

    outcomes = {"1": prediction["home_win"], "N": prediction["draw"], "2": prediction["away_win"]}
    best_outcome = max(outcomes, key=outcomes.get)

    top_score = prediction["top_scores"][0] if prediction.get("top_scores") else None

    rows = []
    rows.append(_make_row(prediction, logged_at, "1N2", best_outcome, outcomes[best_outcome]))
    rows.append(_make_row(prediction, logged_at, "over_under_2.5",
                          "over" if prediction["over_2_5"] >= 0.5 else "under",
                          max(prediction["over_2_5"], 1 - prediction["over_2_5"])))
    rows.append(_make_row(prediction, logged_at, "btts",
                          "oui" if prediction["btts_yes"] >= 0.5 else "non",
                          max(prediction["btts_yes"], prediction["btts_no"])))
    if top_score:
        (h, a), proba = top_score
        rows.append(_make_row(prediction, logged_at, "score_exact_top1", f"{h}-{a}", proba))

    if rows:
        append_rows(rows)


def log_adjusted_prediction(prediction: dict, adjusted: dict) -> None:
    """Enregistre le 1N2 après ajustement de contexte (même schéma de ligne que le modèle seul)."""
    logged_at = datetime.datetime.utcnow().isoformat()
    outcomes = {"1": adjusted["home_win"], "N": adjusted["draw"], "2": adjusted["away_win"]}
    best_outcome = max(outcomes, key=outcomes.get)
    append_rows([_make_row(prediction, logged_at, ADJUSTED_MARKET, best_outcome, outcomes[best_outcome])])
