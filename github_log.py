"""
github_log.py — Écrit predictions_log.csv directement dans le repo GitHub via
l'API (PUT /repos/{owner}/{repo}/contents/{path}), pour contourner le système
de fichiers éphémère de Streamlit Cloud.

Nécessite un secret GITHUB_TOKEN (Personal Access Token avec le scope "repo"
ou, pour un token fine-grained, "Contents: Read and write" sur ce repo précis)
ajouté dans les Secrets de l'app Streamlit Cloud, en plus de FOOTBALL_DATA_KEY.
"""

import base64
import csv
import io
import os

import requests
import streamlit as st

GITHUB_OWNER = "EDWINO-cell"
GITHUB_REPO = "Pronostic-foot"
LOG_PATH_IN_REPO = "predictions_log.csv"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{LOG_PATH_IN_REPO}"

LOG_COLUMNS = [
    "log_id", "logged_at", "match_date", "competition_code",
    "team1_id", "team1_name", "team2_id", "team2_name",
    "market", "predicted_value", "predicted_probability",
    "actual_checked", "actual_result", "was_correct", "brier_error",
]


def get_github_token() -> str:
    if "GITHUB_TOKEN" in st.secrets:
        return st.secrets["GITHUB_TOKEN"]
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Secret GITHUB_TOKEN manquant : ajoute-le dans les Secrets Streamlit Cloud.")
    return token


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {get_github_token()}",
        "Accept": "application/vnd.github+json",
    }


def _fetch_current_file():
    """Retourne (contenu_texte, sha) du fichier existant, ou (None, None) s'il n'existe pas encore."""
    response = requests.get(GITHUB_API_BASE, headers=_headers(), timeout=15)
    if response.status_code == 404:
        return None, None
    response.raise_for_status()
    data = response.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


def append_rows(rows: list[dict]) -> None:
    """
    Ajoute une ou plusieurs lignes à predictions_log.csv dans le repo GitHub.
    Lit le fichier existant, y ajoute les lignes, recommite. Si deux clics
    surviennent au même instant, un conflit de sha peut faire échouer un des
    deux appels : on retente une fois avec le sha à jour avant d'abandonner.
    """
    for attempt in range(2):
        current_content, sha = _fetch_current_file()

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=LOG_COLUMNS)

        if current_content:
            buffer.write(current_content)
            if not current_content.endswith("\n"):
                buffer.write("\n")
        else:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)

        new_content = buffer.getvalue()
        payload = {
            "message": f"Log {len(rows)} prédiction(s) via l'app",
            "content": base64.b64encode(new_content.encode("utf-8")).decode("utf-8"),
        }
        if sha:
            payload["sha"] = sha

        response = requests.put(GITHUB_API_BASE, headers=_headers(), json=payload, timeout=15)
        if response.status_code in (200, 201):
            return
        if response.status_code == 409 and attempt == 0:
            continue  # conflit de sha (écriture concurrente) : on retente avec le sha à jour
        # Ne bloque jamais l'affichage du pronostic à cause d'un souci de log.
        st.warning("Le log de cette prédiction n'a pas pu être enregistré (non bloquant).")
        return
  
