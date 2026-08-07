"""
app.py — Robot de pronostic foot (application Streamlit)
"""

import json
import math
import os
import time
import unicodedata

import requests
import streamlit as st

API_BASE_URL = "https://api.football-data.org/v4"
REQUEST_DELAY_SECONDS = 6.5
MAX_GOALS = 6

COMPETITIONS = ["PL", "PD", "BL1", "SA", "FL1", "DED", "PPL", "ELC", "BSA", "CL"]

LEAGUE_SETTINGS = {
    "PL": (-0.13, 0.6),
    "PD": (-0.18, 0.6),
    "BL1": (0.00, 0.8),
    "SA": (-0.18, 0.4),
    "FL1": (-0.13, 0.6),
    "DED": (-0.13, 0.8),
    "PPL": (-0.18, 0.4),
    "ELC": (-0.18, 0.4),
    "BSA": (-0.18, 0.4),
    "CL": (-0.13, 0.6),
}
DEFAULT_SETTINGS = (-0.13, 0.6)
MIN_MATCHES_MIXED = 6
MIN_MATCHES_VENUE = 4


def get_api_key():
    if "FOOTBALL_DATA_KEY" in st.secrets:
        return st.secrets["FOOTBALL_DATA_KEY"]
    key = os.environ.get("FOOTBALL_DATA_KEY")
    if not key:
        st.error("Clé API manquante. Ajoute FOOTBALL_DATA_KEY dans les Secrets de l'application.")
        st.stop()
    return key


def strip_accents(text):
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def api_get(endpoint, params=None):
    headers = {"X-Auth-Token": get_api_key()}
    for attempt in range(4):
        response = requests.get(f"{API_BASE_URL}/{endpoint}", headers=headers, params=params or {}, timeout=15)
        if response.status_code == 429:
            time.sleep(15)
            continue
        if response.status_code != 200:
            raise RuntimeError(f"Erreur API ({response.status_code}) sur {endpoint} : {response.text}")
        time.sleep(REQUEST_DELAY_SECONDS)
        return response.json()
    raise RuntimeError(f"Échec après plusieurs tentatives sur {endpoint} (limite de requêtes).")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def build_team_directory():
    directory = []
    for code in COMPETITIONS:
        data = api_get(f"competitions/{code}/teams")
        for team in data.get("teams", []):
            directory.append({"id": team["id"], "name": team["name"],
                               "short_name": team.get("shortName", ""),
                               "tla": team.get("tla", ""), "competition_code": code})
    return directory


def find_team(name, directory):
    query = strip_accents(name).lower()
    candidates = []
    for team in directory:
        haystack = strip_accents(f"{team['name']} {team['short_name']} {team['tla']}").lower()
        if query in haystack:
            candidates.append(team)
    return candidates[0] if candidates else None


@st.cache_data(ttl=1800, show_spinner=False)
def get_recent_matches(team_id, n=20):
    data = api_get(f"teams/{team_id}/matches", {"status": "FINISHED", "limit": n})
    matches = data.get("matches", [])
    matches.sort(key=lambda m: m["utcDate"])
    return matches


def compute_form_mixed(team_id, matches):
    scored, conceded, played = 0, 0, 0
    for m in matches[-6:]:
        score = m.get("score", {}).get("fullTime", {})
        home = m["homeTeam"]["id"] == team_id
        gf = score.get("home") if home else score.get("away")
        ga = score.get("away") if home else score.get("home")
        if gf is None or ga is None:
            continue
        scored += gf
        conceded += ga
        played += 1
    if played == 0:
        return {"avg_scored": 1.2, "avg_conceded": 1.2, "played": 0}
    return {"avg_scored": scored / played, "avg_conceded": conceded / played, "played": played}


def compute_form_home(team_id, matches):
    home_matches = [m for m in matches if m["homeTeam"]["id"] == team_id][-6:]
    scored, conceded, played = 0, 0, 0
    for m in home_matches:
        score = m.get("score", {}).get("fullTime", {})
        gf, ga = score.get("home"), score.get("away")
        if gf is None or ga is None:
            continue
        scored += gf
        conceded += ga
        played += 1
    if played == 0:
        return {"avg_scored": None, "avg_conceded": None, "played": 0}
    return {"avg_scored": scored / played, "avg_conceded": conceded / played, "played": played}


def compute_form_away(team_id, matches):
    away_matches = [m for m in matches if m["awayTeam"]["id"] == team_id][-6:]
    scored, conceded, played = 0, 0, 0
    for m in away_matches:
        score = m.get("score", {}).get("fullTime", {})
        gf, ga = score.get("away"), score.get("home")
        if gf is None or ga is None:
            continue
        scored += gf
        conceded += ga
        played += 1
    if played == 0:
        return {"avg_scored": None, "avg_conceded": None, "played": 0}
    return {"avg_scored": scored / played, "avg_conceded": conceded / played, "played": played}


def blended_value(mixed_value, venue_value, venue_played, alpha):
    if venue_value is None or venue_played < MIN_MATCHES_VENUE:
        return mixed_value
    return alpha * venue_value + (1 - alpha) * mixed_value


def find_h2h_match_id(team1_id, team2_id):
    data = api_get(f"teams/{team1_id}/matches", {"status": "SCHEDULED"})
    for m in data.get("matches", []):
        if m["homeTeam"]["id"] == team2_id or m["awayTeam"]["id"] == team2_id:
            return m["id"]
    return None


def get_h2h(team1_id, team2_id, n=5):
    match_id = find_h2h_match_id(team1_id, team2_id)
    if not match_id:
        return {"played": 0, "avg_total_goals": None}
    data = api_get(f"matches/{match_id}/head2head", {"limit": n})
    matches = data.get("matches", [])
    total_goals, played = 0, 0
    for m in matches:
        score = m.get("score", {}).get("fullTime", {})
        h, a = score.get("home"), score.get("away")
        if h is None or a is None:
            continue
        total_goals += h + a
        played += 1
    avg_goals = (total_goals / played) if played else None
    return {"played": played, "avg_total_goals": avg_goals}


@st.cache_data(ttl=1800, show_spinner=False)
def get_standings_table(competition_code):
    data = api_get(f"competitions/{competition_code}/standings")
    for table in data.get("standings", []):
        if table.get("type") == "TOTAL":
            return table.get("table", [])
    return []


def get_standing(team_id, competition_code):
    table = get_standings_table(competition_code)
    for row in table:
        if row["team"]["id"] == team_id:
            return {"rank": row["position"], "points": row["points"]}
    return None


def estimate_stakes_factor(standing1, standing2):
    if not standing1 or not standing2:
        return 1.0, "enjeu inconnu (classement indisponible)"
    rank_gap = abs(standing1["rank"] - standing2["rank"])
    if rank_gap <= 3:
        return 0.92, f"enjeu élevé (écart de {rank_gap} places au classement)"
    elif rank_gap <= 8:
        return 0.97, f"enjeu modéré (écart de {rank_gap} places au classement)"
    return 1.0, f"enjeu faible (écart de {rank_gap} places au classement)"


def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def dixon_coles_tau(h, a, lh, la, rho):
    if h == 0 and a == 0:
        return 1 - (lh * la * rho)
    elif h == 0 and a == 1:
        return 1 + (lh * rho)
    elif h == 1 and a == 0:
        return 1 + (la * rho)
    elif h == 1 and a == 1:
        return 1 - rho
    return 1.0


def build_score_matrix(lambda_home, lambda_away, rho):
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


def run_prediction(team1_name, team2_name):
    directory = build_team_directory()
    team1 = find_team(team1_name, directory)
    team2 = find_team(team2_name, directory)

    if team1 is None:
        raise ValueError(f"Aucune équipe trouvée pour « {team1_name} ».")
    if team2 is None:
        raise ValueError(f"Aucune équipe trouvée pour « {team2_name} ».")

    rho, alpha = LEAGUE_SETTINGS.get(team1["competition_code"], DEFAULT_SETTINGS)

    matches1 = get_recent_matches(team1["id"])
    matches2 = get_recent_matches(team2["id"])

    mixed1 = compute_form_mixed(team1["id"], matches1)
    mixed2 = compute_form_mixed(team2["id"], matches2)
    low_data_warning = mixed1["played"] < MIN_MATCHES_MIXED or mixed2["played"] < MIN_MATCHES_MIXED

    home_form = compute_form_home(team1["id"], matches1)
    away_form = compute_form_away(team2["id"], matches2)

    home_scored = blended_value(mixed1["avg_scored"], home_form["avg_scored"], home_form["played"], alpha)
    home_conceded = blended_value(mixed1["avg_conceded"], home_form["avg_conceded"], home_form["played"], alpha)
    away_scored = blended_value(mixed2["avg_scored"], away_form["avg_scored"], away_form["played"], alpha)
    away_conceded = blended_value(mixed2["avg_conceded"], away_form["avg_conceded"], away_form["played"], alpha)

    h2h = get_h2h(team1["id"], team2["id"])

    standing1 = get_standing(team1["id"], team1["competition_code"])
    standing2 = get_standing(team2["id"], team2["competition_code"])
    stakes_factor, stakes_label = estimate_stakes_factor(standing1, standing2)

    home_bonus = 1 + 0.08 * (1 - alpha)
    away_malus = 1 - 0.05 * (1 - alpha)

    lambda_home = ((home_scored + away_conceded) / 2) * home_bonus
    lambda_away = ((away_scored + home_conceded) / 2) * away_malus

    if h2h["played"] and h2h["avg_total_goals"] is not None:
        h2h_share = h2h["avg_total_goals"] / 2
        lambda_home = lambda_home * 0.8 + h2h_share * 0.2
        lambda_away = lambda_away * 0.8 + h2h_share * 0.2

    lambda_home *= stakes_factor
    lambda_away *= stakes_factor

    matrix = build_score_matrix(lambda_home, lambda_away, rho)
    top_scores = sorted(matrix.items(), key=lambda x: x[1], reverse=True)[:3]
    over_2_5 = sum(p for (h, a), p in matrix.items() if h + a > 2)
    btts_yes = sum(p for (h, a), p in matrix.items() if h > 0 and a > 0)

    return {
        "team1": team1, "team2": team2,
        "home_scored": home_scored, "home_conceded": home_conceded,
        "away_scored": away_scored, "away_conceded": away_conceded,
        "h2h": h2h, "standing1": standing1, "standing2": standing2, "stakes_label": stakes_label,
        "top_scores": top_scores, "over_2_5": over_2_5, "under_2_5": 1 - over_2_5,
        "btts_yes": btts_yes, "btts_no": 1 - btts_yes,
        "low_data_warning": low_data_warning,
    }


st.set_page_config(page_title="Robot de pronostic foot", page_icon="⚽", layout="centered")

st.title("⚽ Robot de pronostic foot")
st.caption("Premier League, Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Liga Portugal, "
           "Championship, Brasileirão, Ligue des Champions")

col1, col2 = st.columns(2)
with col1:
    team1_input = st.text_input("Équipe 1 (domicile)", placeholder="ex: PSG")
with col2:
    team2_input = st.text_input("Équipe 2 (extérieur)", placeholder="ex: Marseille")

if st.button("Prédire", type="primary", use_container_width=True):
    if not team1_input.strip() or not team2_input.strip():
        st.warning("Renseigne les deux équipes.")
    else:
        with st.spinner("Analyse en cours... (peut prendre 30-60s la première fois)"):
            try:
                result = run_prediction(team1_input.strip(), team2_input.strip())
            except ValueError as e:
                st.error(str(e))
                result = None
            except RuntimeError as e:
                st.error(f"Erreur de connexion à l'API : {e}")
                result = None

        if result:
            t1, t2 = result["team1"], result["team2"]

            if result["low_data_warning"]:
                st.warning("Peu de matchs disponibles cette saison pour ces équipes — la prédiction sera peu précise.")

            st.subheader(f"{t1['name']} vs {t2['name']}")

            c1, c2 = st.columns(2)
            with c1:
                st.metric(f"{t1['name']} (domicile)", f"{result['home_scored']:.2f} marqués/match",
                          f"{result['home_conceded']:.2f} encaissés/match", delta_color="off")
            with c2:
                st.metric(f"{t2['name']} (extérieur)", f"{result['away_scored']:.2f} marqués/match",
                          f"{result['away_conceded']:.2f} encaissés/match", delta_color="off")

            if result["h2h"]["played"]:
                st.caption(f"📊 {result['h2h']['played']} confrontation(s) directe(s) récente(s), "
                           f"moyenne {result['h2h']['avg_total_goals']:.2f} buts/match")

            if result["standing1"] and result["standing2"]:
                st.caption(f"🏆 {t1['name']} : {result['standing1']['rank']}e place ({result['standing1']['points']} pts) — "
                           f"{t2['name']} : {result['standing2']['rank']}e place ({result['standing2']['points']} pts)")
            st.caption(f"⚖️ {result['stakes_label']}")

            st.markdown("### Top 3 des scores exacts")
            for (h, a), p in result["top_scores"]:
                st.write(f"**{t1['name']} {h} - {a} {t2['name']}** : {p * 100:.1f}%")

            st.markdown("### Plus/moins 2.5 buts")
            c1, c2 = st.columns(2)
            c1.metric("Plus de 2.5", f"{result['over_2_5'] * 100:.1f}%")
            c2.metric("Moins de 2.5", f"{result['under_2_5'] * 100:.1f}%")

            st.markdown("### BTTS (les deux équipes marquent)")
            c1, c2 = st.columns(2)
            c1.metric("Oui", f"{result['btts_yes'] * 100:.1f}%")
            c2.metric("Non", f"{result['btts_no'] * 100:.1f}%")

st.divider()
st.caption("⚠️ Modèle statistique à titre indicatif. Ne bat pas systématiquement une prédiction "
           "naïve sur tous les championnats.")
