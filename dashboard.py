"""
dashboard.py — Tableau de bord de suivi de performance des pronostics foot.

Lit predictions_log.csv directement depuis GitHub (via github_log.py) pour
toujours afficher l'état le plus récent, indépendamment du cycle de
redéploiement de l'app. Affiche : nombre de prédictions loggées/vérifiées,
taux de réussite par marché, par championnat, et l'évolution dans le temps.
"""

import pandas as pd
import streamlit as st

from github_log import fetch_predictions_log_as_rows

MARKET_LABELS = {
    "1N2": "Résultat 1N2",
    "over_under_2.5": "Over/Under 2.5",
    "btts": "BTTS",
    "score_exact_top1": "Score exact (top 1)",
}


def load_log_dataframe() -> pd.DataFrame:
    rows = fetch_predictions_log_as_rows()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["actual_checked"] = df["actual_checked"] == "true"
    df["was_correct"] = df["was_correct"].map({"true": True, "false": False})
    df["predicted_probability"] = pd.to_numeric(df["predicted_probability"], errors="coerce")
    df["brier_error"] = pd.to_numeric(df["brier_error"], errors="coerce")
    df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce")
    return df


def render_dashboard() -> None:
    st.subheader("📊 Suivi de performance réelle")
    st.caption(
        "Comparaison entre les probabilités données par le modèle et ce qui "
        "s'est réellement passé, match après match."
    )

    df = load_log_dataframe()

    if df.empty:
        st.info("Aucune prédiction loggée pour le moment. Lance quelques pronostics pour commencer à alimenter ce tableau de bord.")
        return

    total_logged = len(df)
    verified = df[df["actual_checked"]]
    total_verified = len(verified)

    col1, col2, col3 = st.columns(3)
    col1.metric("Prédictions loggées", total_logged)
    col2.metric("Vérifiées", total_verified)
    if total_verified > 0:
        overall_accuracy = verified["was_correct"].mean() * 100
        col3.metric("Taux de réussite global", f"{overall_accuracy:.1f}%")
    else:
        col3.metric("Taux de réussite global", "—")

    if total_verified == 0:
        st.warning(
            "Aucune prédiction vérifiée pour l'instant — le script check_results.py "
            "tourne une fois par mois (avec la recalibration). Les premières "
            "vérifications apparaîtront ici après son prochain passage."
        )
        return

    st.divider()

    st.markdown("**Par marché**")
    by_market = (
        verified.groupby("market")["was_correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Taux de réussite", "count": "Nb vérifiées"})
    )
    by_market["Taux de réussite"] = (by_market["Taux de réussite"] * 100).round(1)
    by_market.index = by_market.index.map(lambda m: MARKET_LABELS.get(m, m))
    st.dataframe(by_market, use_container_width=True)

    st.divider()

    st.markdown("**Par championnat**")
    by_league = (
        verified.groupby("competition_code")["was_correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Taux de réussite", "count": "Nb vérifiées"})
    )
    by_league["Taux de réussite"] = (by_league["Taux de réussite"] * 100).round(1)
    st.dataframe(by_league, use_container_width=True)

    st.divider()

    st.markdown("**Calibration** — un pronostic donné à 70% devrait se réaliser ~70% du temps")
    calibration_bins = pd.cut(
        verified["predicted_probability"], bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    calibration = (
        verified.groupby(calibration_bins, observed=True)["was_correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Taux réel", "count": "Nb"})
    )
    calibration["Taux réel"] = (calibration["Taux réel"] * 100).round(1)
    st.dataframe(calibration, use_container_width=True)
    st.caption(
        "Si le \"Taux réel\" d'une tranche est nettement plus bas que la "
        "tranche elle-même, le modèle est trop confiant sur cette plage — "
        "et inversement s'il est plus haut."
    )

    st.divider()

    if total_verified >= 10:
        st.markdown("**Évolution dans le temps**")
        verified_sorted = verified.sort_values("logged_at").reset_index(drop=True)
        verified_sorted["taux_glissant"] = (
            verified_sorted["was_correct"].rolling(window=10, min_periods=5).mean() * 100
        )
        st.line_chart(verified_sorted.set_index("logged_at")["taux_glissant"])
    else:
        st.caption("L'évolution dans le temps s'affichera une fois qu'il y aura au moins 10 prédictions vérifiées.")
