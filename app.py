"""
app.py — Robot de pronostic foot (application Streamlit)

Interface web : deux champs pour les noms d'équipes + un bouton, qui affiche
le pronostic (résultat le plus probable, top 3 scores exacts, plus/moins 2.5
buts, BTTS). Modèle calibré championnat par championnat sur les 10 grands
championnats couverts par football-data.org, avec analyse enrichie de l'enjeu
du match (zones de classement réelles, phase de compétition, derbies).

Déploiement : GitHub + Streamlit Community Cloud. La clé API se configure
dans les "Secrets" de l'application sur Streamlit Cloud, sous la forme :
    FOOTBALL_DATA_KEY = "ton_token_ici"
"""

import datetime
import difflib
import json
import math
import os
import time
import unicodedata

import requests
import streamlit as st

from logging_predictions import log_prediction
from dashboard import render_dashboard

API_BASE_URL = "https://api.football-data.org/v4"
REQUEST_DELAY_SECONDS = 6.5  # le plan gratuit limite à 10 requêtes/minute
MAX_GOALS = 6

COMPETITIONS = ["PL", "PD", "BL1", "SA", "FL1", "DED", "PPL", "ELC", "BSA", "CL"]

import json

DEFAULT_SETTINGS = (-0.13, 0.6)

def load_league_settings() -> dict:
    """
    Charge les réglages rho/alpha depuis league_settings.json (mis à jour
    automatiquement par recalibrate.py). Si le fichier n'existe pas encore
    (première mise en place, avant le premier run de l'Action), on retombe
    sur les valeurs calibrées manuellement en dur, en secours.
    """
    fallback = {
        "PL": (-0.13, 0.6), "PD": (-0.18, 0.6), "BL1": (0.00, 0.8),
        "SA": (-0.18, 0.4), "FL1": (-0.13, 0.6), "DED": (-0.13, 0.8),
        "PPL": (-0.18, 0.4), "ELC": (-0.18, 0.4), "BSA": (-0.18, 0.4),
        "CL": (-0.13, 0.6),
    }
    try:
        with open("league_settings.json") as f:
            raw = json.load(f)
        # recalibrate.py écrit {"PL": {"rho": ..., "alpha": ..., "brier": ...}, ...}
        # on convertit vers le format (rho, alpha) attendu par le reste du code
        return {code: (v["rho"], v["alpha"]) for code, v in raw.items()}
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return fallback

LEAGUE_SETTINGS = load_league_settings()
MIN_MATCHES_MIXED = 6
MIN_MATCHES_VENUE = 4


def get_api_key() -> str:
    if "FOOTBALL_DATA_KEY" in st.secrets:
        return st.secrets["FOOTBALL_DATA_KEY"]
    key = os.environ.get("FOOTBALL_DATA_KEY")
    if not key:
        st.error("Clé API manquante. Ajoute FOOTBALL_DATA_KEY dans les Secrets de l'application.")
        st.stop()
    return key


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def api_get(endpoint: str, params: dict | None = None) -> dict:
    headers = {"X-Auth-Token": get_api_key()}
    for attempt in range(4):
        response = requests.get(f"{API_BASE_URL}/{endpoint}", headers=headers, params=params or {}, timeout=15)
        if response.status_code == 429:
            time.sleep(15)  # limite de requêtes atteinte, on patiente puis on réessaie
            continue
        if response.status_code != 200:
            raise RuntimeError(f"Erreur API ({response.status_code}) sur {endpoint} : {response.text}")
        time.sleep(REQUEST_DELAY_SECONDS)
        return response.json()
    raise RuntimeError(f"Échec après plusieurs tentatives sur {endpoint} (limite de requêtes).")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def build_team_directory() -> list:
    directory = []
    for code in COMPETITIONS:
        data = api_get(f"competitions/{code}/teams")
        for team in data.get("teams", []):
            directory.append({
                "id": team["id"], "name": team["name"],
                "short_name": team.get("shortName", ""), "tla": team.get("tla", ""),
                "competition_code": code,
            })
    return directory


# Surnoms très courants qui ne correspondent à aucun mot du nom officiel
# (donc que la recherche par mots ne peut pas trouver toute seule).
TEAM_ALIASES = {
    "barca": "barcelona", "man utd": "manchester united", "man u": "manchester united",
    "man united": "manchester united", "man city": "manchester city", "inter": "internazionale",
    "juve": "juventus", "atleti": "atletico madrid", "psg": "paris saint-germain",
    "gunners": "arsenal", "spurs": "tottenham", "wolves": "wolverhampton",
    "atletico mg": "atletico mineiro", "atletico-mg": "atletico mineiro",
}


def find_team(name: str, directory: list) -> dict | None:
    """
    Cherche une équipe par nom, en plusieurs passes de plus en plus tolérantes :
    1) phrase exacte, 2) surnom connu (Barça, Man Utd...), 3) tous les mots tapés
    présents (dans n'importe quel ordre), 4) recherche floue (tolère fautes de
    frappe et variantes d'orthographe) — pour couvrir n'importe quel club, pas
    seulement les plus connus.
    """
    query = strip_accents(name).lower().strip()
    if not query:
        return None

    query = TEAM_ALIASES.get(query, query)

    # 1) phrase exacte
    exact = [t for t in directory
             if query in strip_accents(f"{t['name']} {t['short_name']} {t['tla']}").lower()]
    if exact:
        return min(exact, key=lambda t: len(t["name"]))

    # 2) repli : tous les mots tapés doivent apparaître (dans n'importe quel ordre)
    words = [w for w in query.replace("-", " ").split() if w]
    if words:
        partial = []
        for t in directory:
            haystack = strip_accents(f"{t['name']} {t['short_name']} {t['tla']}").lower()
            if all(w in haystack for w in words):
                partial.append(t)
        if partial:
            return min(partial, key=lambda t: len(t["name"]))

    # 3) recherche floue (typos, orthographe légèrement différente) — générique,
    # fonctionne pour n'importe quel club sans avoir à le lister à la main.
    name_index = {}
    for t in directory:
        for candidate in (t["name"], t["short_name"], t["tla"]):
            if candidate:
                name_index[strip_accents(candidate).lower()] = t
    close = difflib.get_close_matches(query, name_index.keys(), n=1, cutoff=0.72)
    if close:
        return name_index[close[0]]

    return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_recent_matches(team_id: int, n: int = 20) -> list:
    data = api_get(f"teams/{team_id}/matches", {"status": "FINISHED", "limit": n})
    matches = data.get("matches", [])
    matches.sort(key=lambda m: m["utcDate"])
    return matches


def compute_form_mixed(team_id: int, matches: list) -> dict:
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


def compute_form_home(team_id: int, matches: list) -> dict:
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


def compute_form_away(team_id: int, matches: list) -> dict:
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


def blended_value(mixed_value: float, venue_value, venue_played: int, alpha: float) -> float:
    if venue_value is None or venue_played < MIN_MATCHES_VENUE:
        return mixed_value
    return alpha * venue_value + (1 - alpha) * mixed_value


def get_scheduled_match(team1_id: int, team2_id: int):
    """Trouve le match programmé entre les deux équipes (pour le H2H et la phase de compétition : finale, demi, etc.)."""
    data = api_get(f"teams/{team1_id}/matches", {"status": "SCHEDULED"})
    for m in data.get("matches", []):
        if m["homeTeam"]["id"] == team2_id or m["awayTeam"]["id"] == team2_id:
            return m
    return None


def get_h2h(scheduled_match, n: int = 5) -> dict:
    if not scheduled_match:
        return {"played": 0, "avg_total_goals": None}
    data = api_get(f"matches/{scheduled_match['id']}/head2head", {"limit": n})
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
def get_standings_table(competition_code: str) -> list:
    data = api_get(f"competitions/{competition_code}/standings")
    for table in data.get("standings", []):
        if table.get("type") == "TOTAL":
            return table.get("table", [])
    return []


def get_standing(team_id: int, competition_code: str):
    table = get_standings_table(competition_code)
    total_teams = len(table)
    for row in table:
        if row["team"]["id"] == team_id:
            return {"rank": row["position"], "points": row["points"], "total_teams": total_teams}
    return None


# Derbies connus (paires de fragments de noms, en minuscules sans accents).
# Liste non exhaustive : les grandes rivalités des championnats couverts.
DERBIES = [
    ("real madrid", "barcelona"), ("real madrid", "atletico"), ("atletico", "barcelona"),
    ("sevilla", "real betis"),
    ("manchester united", "manchester city"), ("liverpool", "everton"),
    ("arsenal", "tottenham"), ("arsenal", "chelsea"), ("chelsea", "tottenham"),
    ("ac milan", "internazionale"), ("ac milan", "inter"), ("roma", "lazio"),
    ("juventus", "torino"), ("napoli", "roma"),
    ("borussia dortmund", "schalke"), ("bayern", "1860 munich"), ("borussia dortmund", "bayern"),
    ("hamburg", "st. pauli"),
    ("paris saint-germain", "marseille"), ("lyon", "saint-etienne"), ("lyon", "saint-étienne"),
    ("psg", "marseille"),
    ("ajax", "feyenoord"), ("ajax", "psv"), ("feyenoord", "psv"),
    ("porto", "benfica"), ("porto", "sporting"), ("benfica", "sporting"),
    ("flamengo", "fluminense"), ("flamengo", "vasco"), ("flamengo", "botafogo"),
    ("corinthians", "palmeiras"), ("sao paulo", "corinthians"), ("gremio", "internacional"),
]


def is_derby(name1: str, name2: str) -> bool:
    n1, n2 = strip_accents(name1).lower(), strip_accents(name2).lower()
    for frag1, frag2 in DERBIES:
        if (frag1 in n1 and frag2 in n2) or (frag2 in n1 and frag1 in n2):
            return True
    return False


# Règles réelles de qualification/relégation par championnat (saison 2025-26, vérifiées).
# "cl" / "europa" / "conference" = nombre de places à partir du rang 1 (cumulatif).
# "relegation" = nombre d'équipes reléguées directement. "relegation_playoff" = 1 si
# une place supplémentaire dispute un barrage de maintien (juste au-dessus de la zone rouge).
# Limite résiduelle assumée : certaines places "swing" (ex. Angleterre certaines saisons)
# dépendent aussi des vainqueurs de coupes nationales, non suivis ici — approximation
# raisonnable mais pas garantie à la place près dans ces cas de figure.
LEAGUE_ZONE_RULES = {
    "PL":  {"total": 20, "cl": 4, "europa": 2, "conference": 1, "relegation": 3},
    "PD":  {"total": 20, "cl": 4, "europa": 1, "conference": 1, "relegation": 3},
    "BL1": {"total": 18, "cl": 4, "europa": 1, "conference": 1, "relegation": 2, "relegation_playoff": 1},
    "SA":  {"total": 20, "cl": 4, "europa": 1, "conference": 1, "relegation": 3},
    "FL1": {"total": 18, "cl": 4, "europa": 2, "conference": 1, "relegation": 2},
    "DED": {"total": 18, "cl": 3, "europa": 1, "conference": 5, "relegation": 2, "relegation_playoff": 1},
    "PPL": {"total": 18, "cl": 2, "europa": 1, "conference": 1, "relegation": 2, "relegation_playoff": 1},
}


def classify_zone(rank: int, total_teams: int, competition_code: str) -> str:
    if competition_code == "CL":
        # Phase de ligue UEFA : la notion de zone de classement classique ne s'applique
        # pas — l'enjeu vient surtout du stade de la compétition (finale, demi, etc.).
        return "mid_table"

    if competition_code == "BSA":
        if not total_teams or rank is None:
            return "mid_table"
        if rank <= 6:
            return "continental_zone"  # Copa Libertadores
        if rank <= 12:
            return "sudamericana_zone"
        if rank > total_teams - 4:  # 4 relégués au Brésil
            return "relegation_zone"
        return "mid_table"

    if competition_code == "ELC":
        if not total_teams or rank is None:
            return "mid_table"
        if rank <= 2:
            return "automatic_promotion"
        if rank <= 6:
            return "playoff_zone"
        if rank > total_teams - 3:
            return "relegation_zone"
        return "mid_table"

    rules = LEAGUE_ZONE_RULES.get(competition_code)
    if not rules or rank is None or not total_teams:
        if rank is None or not total_teams:
            return "mid_table"
        if rank <= 4:
            return "ucl_zone"
        if rank <= 6:
            return "european_zone"
        if rank > total_teams - 3:
            return "relegation_zone"
        return "mid_table"

    cl_cutoff = rules["cl"]
    european_cutoff = cl_cutoff + rules.get("europa", 0) + rules.get("conference", 0)
    relegation_size = rules.get("relegation", 3)
    playoff_rank = (total_teams - relegation_size) if rules.get("relegation_playoff") else None

    if rank <= 1:
        return "leader"
    if rank <= cl_cutoff:
        return "ucl_zone"
    if rank <= european_cutoff:
        return "european_zone"
    if rank > total_teams - relegation_size:
        return "relegation_zone"
    if playoff_rank and rank == playoff_rank:
        return "relegation_playoff_zone"
    return "mid_table"


STAGE_LABELS = {
    "FINAL": ("finale", 0.85),
    "SEMI_FINALS": ("demi-finale", 0.90),
    "QUARTER_FINALS": ("quart de finale", 0.93),
    "LAST_16": ("8e de finale", 0.95),
    "PLAYOFFS": ("barrage / match de qualification", 0.93),
}


def analyze_match_context(standing1, standing2, competition_code, team1_name, team2_name, scheduled_match):
    """
    Analyse le contexte réel du match : derby, phase de compétition (finale, demi...),
    et lutte de classement (titre, Europe, maintien, barrage). Renvoie un facteur
    d'ajustement des buts attendus et un libellé explicatif.
    """
    labels = []
    stakes_factor = 1.0

    if is_derby(team1_name, team2_name):
        labels.append("derby")
        stakes_factor *= 0.92

    stage = scheduled_match.get("stage") if scheduled_match else None
    if stage in STAGE_LABELS:
        label, factor = STAGE_LABELS[stage]
        labels.append(label)
        stakes_factor *= factor

    if standing1 and standing2:
        total_teams = standing1.get("total_teams") or standing2.get("total_teams")
        zone1 = classify_zone(standing1["rank"], total_teams, competition_code)
        zone2 = classify_zone(standing2["rank"], total_teams, competition_code)
        point_gap = abs(standing1["points"] - standing2["points"])
        close = point_gap <= 6

        if (zone1 in ("leader", "ucl_zone") and zone2 in ("leader", "ucl_zone")
                and standing1["rank"] <= 3 and standing2["rank"] <= 3 and close):
            labels.append("lutte pour le titre")
            stakes_factor *= 0.90
        elif zone1 == "automatic_promotion" and zone2 == "automatic_promotion" and close:
            labels.append("lutte pour la montée directe")
            stakes_factor *= 0.90
        elif zone1 == "relegation_zone" and zone2 == "relegation_zone":
            labels.append("lutte pour le maintien")
            stakes_factor *= 0.93
        elif zone1 == "relegation_playoff_zone" and zone2 == "relegation_playoff_zone" and close:
            labels.append("lutte pour éviter le barrage de relégation")
            stakes_factor *= 0.94
        elif (zone1 in ("relegation_zone", "relegation_playoff_zone")
              or zone2 in ("relegation_zone", "relegation_playoff_zone")) and close:
            labels.append("enjeu de maintien")
            stakes_factor *= 0.95
        elif zone1 == "playoff_zone" and zone2 == "playoff_zone":
            labels.append("lutte pour le barrage de promotion")
            stakes_factor *= 0.93
        elif zone1 == "continental_zone" and zone2 == "continental_zone" and close:
            labels.append("lutte pour une place en Copa Libertadores")
            stakes_factor *= 0.94
        elif zone1 == "sudamericana_zone" and zone2 == "sudamericana_zone" and close:
            labels.append("lutte pour une place en Copa Sudamericana")
            stakes_factor *= 0.96
        elif (zone1 in ("ucl_zone", "european_zone") and zone2 in ("ucl_zone", "european_zone") and close):
            labels.append("lutte pour une place européenne")
            stakes_factor *= 0.95
        elif zone1 == "mid_table" and zone2 == "mid_table":
            labels.append("match sans grand enjeu de classement")
            stakes_factor *= 1.05

    if not labels:
        labels.append("enjeu standard")

    label_text = ", ".join(labels)
    return stakes_factor, label_text[0].upper() + label_text[1:]


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


def run_prediction_for_teams(team1: dict, team2: dict) -> dict:
    """Calcule le pronostic pour deux équipes déjà identifiées (id, name, competition_code)."""
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

    scheduled_match = get_scheduled_match(team1["id"], team2["id"])
    h2h = get_h2h(scheduled_match)

    standing1 = get_standing(team1["id"], team1["competition_code"])
    standing2 = get_standing(team2["id"], team2["competition_code"])
    stakes_factor, stakes_label = analyze_match_context(
        standing1, standing2, team1["competition_code"], team1["name"], team2["name"], scheduled_match
    )

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
    home_win = sum(p for (h, a), p in matrix.items() if h > a)
    draw = sum(p for (h, a), p in matrix.items() if h == a)
    away_win = sum(p for (h, a), p in matrix.items() if h < a)

    return {
        "team1": team1, "team2": team2,
        "home_scored": home_scored, "home_conceded": home_conceded,
        "away_scored": away_scored, "away_conceded": away_conceded,
        "h2h": h2h, "standing1": standing1, "standing2": standing2, "stakes_label": stakes_label,
        "scheduled_match": scheduled_match,
        "scheduled_match": scheduled_match,
        "top_scores": top_scores, "over_2_5": over_2_5, "under_2_5": 1 - over_2_5,
        "btts_yes": btts_yes, "btts_no": 1 - btts_yes,
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "low_data_warning": low_data_warning,
    }


def run_prediction(team1_name: str, team2_name: str) -> dict:
    directory = build_team_directory()
    team1 = find_team(team1_name, directory)
    team2 = find_team(team2_name, directory)

    if team1 is None:
        raise ValueError(f"Aucune équipe trouvée pour « {team1_name} ». Essaie un nom plus simple "
                          f"(ex: juste le nom de la ville ou l'abréviation officielle du club).")
    if team2 is None:
        raise ValueError(f"Aucune équipe trouvée pour « {team2_name} ». Essaie un nom plus simple "
                          f"(ex: juste le nom de la ville ou l'abréviation officielle du club).")

    return run_prediction_for_teams(team1, team2)


@st.cache_data(ttl=1800, show_spinner=False)
def get_upcoming_fixtures(days: int = 7) -> list:
    """Liste les vrais matchs programmés dans les prochains jours, tous championnats couverts confondus."""
    today = datetime.date.today()
    date_to = today + datetime.timedelta(days=days)
    data = api_get("matches", {
        "competitions": ",".join(COMPETITIONS),
        "dateFrom": today.isoformat(),
        "dateTo": date_to.isoformat(),
    })
    matches = data.get("matches", [])
    matches.sort(key=lambda m: m["utcDate"])
    return matches


# ---------- Interface ----------

def render_prediction(result: dict, subtitle: str | None = None) -> None:
    t1, t2 = result["team1"], result["team2"]

    if result["low_data_warning"]:
        st.warning("Peu de matchs disponibles cette saison pour ces équipes — la prédiction sera peu précise "
                   "(probablement début de saison).")

    st.subheader(f"{t1['name']} vs {t2['name']}")
    if subtitle:
        st.caption(subtitle)

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
    st.caption(f"⚖️ Enjeu du match : {result['stakes_label']}")

    outcomes = [("home", result["home_win"], f"Victoire {t1['name']}"),
                ("draw", result["draw"], "Match nul"),
                ("away", result["away_win"], f"Victoire {t2['name']}")]
    best_outcome = max(outcomes, key=lambda x: x[1])
    st.markdown(f"### 🎯 Résultat le plus probable : {best_outcome[2]} ({best_outcome[1] * 100:.1f}%)")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{t1['name']}", f"{result['home_win'] * 100:.1f}%")
    c2.metric("Nul", f"{result['draw'] * 100:.1f}%")
    c3.metric(f"{t2['name']}", f"{result['away_win'] * 100:.1f}%")

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


st.set_page_config(page_title="Robot de pronostic foot", page_icon="⚽", layout="centered")

st.title("⚽ Robot de pronostic foot")
st.caption("Premier League, Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Liga Portugal, "
           "Championship, Brasileirão, Ligue des Champions")

tab_manual, tab_upcoming, tab_dashboard = st.tabs(["🔎 Prédiction manuelle", "📅 Matchs à venir", "📊 Performance"])

with tab_manual:
    col1, col2 = st.columns(2)
    with col1:
        team1_input = st.text_input("Équipe 1 (domicile)", placeholder="ex: PSG")
    with col2:
        team2_input = st.text_input("Équipe 2 (extérieur)", placeholder="ex: Marseille")

    if st.button("Prédire", type="primary", use_container_width=True):
        if not team1_input.strip() or not team2_input.strip():
            st.warning("Renseigne les deux équipes.")
        else:
            with st.spinner("Analyse en cours..."):
                try:
                    result = run_prediction(team1_input.strip(), team2_input.strip())
                except ValueError as e:
                    st.error(str(e))
                    result = None
                except RuntimeError as e:
                    st.error(f"Erreur de connexion à l'API : {e}")
                    result = None

            if result:
                log_prediction(result)
                t1, t2 = result["team1"], result["team2"]
                subtitle = (f"Trouvé pour « {team1_input.strip()} » → {t1['name']} ({t1['competition_code']})  |  "
                            f"« {team2_input.strip()} » → {t2['name']} ({t2['competition_code']})")
                render_prediction(result, subtitle)

with tab_upcoming:
    st.caption("Matchs programmés dans les 7 prochains jours, tous championnats couverts confondus.")
    try:
        with st.spinner("Récupération des matchs à venir..."):
            fixtures = get_upcoming_fixtures()
    except RuntimeError as e:
        st.error(f"Erreur de connexion à l'API : {e}")
        fixtures = []

    if not fixtures:
        st.info("Aucun match programmé trouvé dans les 7 prochains jours pour ces championnats.")

    selected_key = st.session_state.get("selected_fixture")

    for m in fixtures:
        home, away = m["homeTeam"]["name"], m["awayTeam"]["name"]
        comp = m["competition"]["code"]
        when = m["utcDate"][:16].replace("T", " ")
        key = f"fx_{m['id']}"
        col_a, col_b = st.columns([4, 1])
        col_a.write(f"**{home} vs {away}**  \n{comp} — {when} UTC")
        if col_b.button("Voir", key=key, use_container_width=True):
            st.session_state["selected_fixture"] = m["id"]
            st.session_state["selected_fixture_data"] = {
                "home": {"id": m["homeTeam"]["id"], "name": home, "competition_code": comp},
                "away": {"id": m["awayTeam"]["id"], "name": away, "competition_code": comp},
            }
with tab_dashboard:
    render_dashboard()

    if st.session_state.get("selected_fixture_data"):
        st.divider()
        teams = st.session_state["selected_fixture_data"]
        with st.spinner("Analyse en cours..."):
            try:
                result = run_prediction_for_teams(teams["home"], teams["away"])
                log_prediction(result)
                render_prediction(result)
            except RuntimeError as e:
                st.error(f"Erreur de connexion à l'API : {e}")

st.divider()
st.caption("⚠️ Modèle statistique à titre indicatif. Ne bat pas systématiquement une prédiction "
           "naïve sur tous les championnats — voir le détail du backtesting du projet.")
