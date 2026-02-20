"""
LaLiga match score predictor using current standings.

Usage:
  python predict_score.py

The script fetches current LaLiga standings from ESPN, computes simple
attack/defense strengths, and predicts the most likely scoreline between
two chosen teams using independent Poisson models.
"""
import sys
import math
import random
##import requests
from collections import defaultdict

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

ESP_NATIVES = [
    "https://www.espn.com/soccer/standings/_/league/ESP.1",
    "https://www.espn.com/soccer/table/_/league/ESP.1",
]


def fetch_html(url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible)"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    return r.text


def parse_standings(html):
    """Return dict: team_name -> stats dict with keys GP, W, D, L, F, A, GD, PTS"""
    if BeautifulSoup is None:
        raise RuntimeError("BeautifulSoup (bs4) is required. Install with: pip install beautifulsoup4")

    soup = BeautifulSoup(html, "html.parser")
    # Find table containing headers like 'W' 'L' 'F' 'A' 'GD' 'P'
    candidates = soup.find_all("table")
    for table in candidates:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        header_text = " ".join(headers).upper()
        if all(x in header_text for x in ("W", "L", "F", "A", "GD", "PTS")):
            # parse rows
            rows = table.find_all("tr")
            teams = {}
            for tr in rows:
                cols = [td.get_text(strip=True) for td in tr.find_all(["td"]) ]
                if not cols:
                    continue
                # Many ESPN tables put team name in first or second column; find a name column
                name = None
                for c in cols[:3]:
                    if any(ch.isalpha() for ch in c):
                        name = c
                        break
                if name is None:
                    continue
                # Normalize name: remove rank prefix if present
                name = name.strip()
                # Attempt to extract numeric stats from trailing columns
                nums = []
                for c in cols:
                    try:
                        nums.append(int(c))
                    except Exception:
                        # maybe contains '34-21' style or 'F-A'
                        pass
                if len(nums) >= 8:
                    GP, W, D, L, F, A, GD, PTS = nums[:8]
                    teams[name] = {"GP": GP, "W": W, "D": D, "L": L, "F": F, "A": A, "GD": GD, "PTS": PTS}
            if teams:
                return teams

    # Fallback: try to parse textual listings like '1 Real Madrid 24 19 3 2 53 19 60'
    text = soup.get_text(separator="\n")
    teams = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 9 and parts[0].isdigit():
            # position, maybe shortclub, then name tokens, then numbers
            # find last 8 numeric tokens
            nums = []
            for token in reversed(parts):
                if token.lstrip('+-').isdigit():
                    nums.insert(0, int(token))
                    if len(nums) == 8:
                        break
            if len(nums) == 8:
                # name is middle
                name_tokens = parts[1:len(parts)-8]
                name = " ".join(name_tokens)
                GP, W, D, L, F, A, GD, PTS = nums
                teams[name] = {"GP": GP, "W": W, "D": D, "L": L, "F": F, "A": A, "GD": GD, "PTS": PTS}
    return teams


def get_current_standings():
    last_err = None
    for url in ESP_NATIVES:
        try:
            html = fetch_html(url)
            teams = parse_standings(html)
            if teams:
                return teams
        except Exception as e:
            last_err = e
    # If fetching/parsing failed, use a randomized fallback so the script still runs
    print("Warning: could not fetch live standings — using random fallback standings.")
    return generate_random_standings()


def generate_random_standings(seed: int | None = None):
    """Create a plausible random standings dict for common LaLiga teams.

    Only `GP`, `F`, and `A` are required for the predictor; other fields are
    filled with simple derived values.
    """
    if seed is not None:
        random.seed(seed)
    teams_list = [
        "Real Madrid", "Barcelona", "Villarreal", "Atletico Madrid", "Real Betis",
        "Espanyol", "Celta Vigo", "Real Sociedad", "Athletic Club", "Osasuna",
        "Getafe", "Girona", "Sevilla", "Alaves", "Valencia",
        "Elche", "Rayo Vallecano", "Mallorca", "Levante", "Real Oviedo",
    ]
    teams = {}
    for name in teams_list:
        gp = random.randint(20, 24)
        # goals for/against scaled by strength proxy
        strength = random.uniform(0.6, 1.6)
        f = max(5, int(strength * random.randint(20, 60)))
        a = max(5, int((2.2 - strength) * random.randint(15, 50)))
        gd = f - a
        # crude wins/draws/losses distribution (not used directly in model)
        w = random.randint(3, gp - 6)
        d = random.randint(0, gp - w - 3)
        l = gp - w - d
        pts = w * 3 + d
        teams[name] = {"GP": gp, "W": w, "D": d, "L": l, "F": f, "A": a, "GD": gd, "PTS": pts}
    return teams


def poisson_pmf(k, lam):
    return (lam**k) * math.exp(-lam) / math.factorial(k)


def predict_scoreline(team1_stats, team2_stats, max_goals=6):
    # Compute per-game rates
    t1_gp = team1_stats.get('GP', 1)
    t2_gp = team2_stats.get('GP', 1)
    t1_gf_pg = team1_stats.get('F', 0) / max(1, t1_gp)
    t1_ga_pg = team1_stats.get('A', 0) / max(1, t1_gp)
    t2_gf_pg = team2_stats.get('F', 0) / max(1, t2_gp)
    t2_ga_pg = team2_stats.get('A', 0) / max(1, t2_gp)

    # Simple expected goals model: average of attack and opponent defense
    lamb1 = (t1_gf_pg + t2_ga_pg) / 2.0
    lamb2 = (t2_gf_pg + t1_ga_pg) / 2.0

    # compute joint probabilities for scorelines up to max_goals-1
    best = None
    all_probs = []
    for g1 in range(0, max_goals):
        p1 = poisson_pmf(g1, lamb1)
        for g2 in range(0, max_goals):
            p2 = poisson_pmf(g2, lamb2)
            prob = p1 * p2
            all_probs.append(((g1, g2), prob))
            if best is None or prob > best[1]:
                best = ((g1, g2), prob)

    # sort top results
    top = sorted(all_probs, key=lambda x: x[1], reverse=True)[:6]
    return {"expected": best[0], "prob": best[1], "top": top, "lambda": (lamb1, lamb2)}


def choose_team(teams, prompt):
    names = sorted(teams.keys())
    print("Available teams:")
    for i, n in enumerate(names, 1):
        print(f"{i}. {n}")
    choice = input(prompt)
    try:
        idx = int(choice)
        return names[idx-1]
    except Exception:
        # try match by name
        for n in names:
            if choice.lower() in n.lower():
                return n
    raise ValueError("Invalid team selection")


def main():
    print("Fetching current LaLiga standings...")
    teams = get_current_standings()
    if not teams:
        print("Could not fetch standings. Exiting.")
        return

    # allow user to pick two teams
    try:
        t1 = choose_team(teams, "Choose home team (number or name): ")
        t2 = choose_team(teams, "Choose away team (number or name): ")
    except Exception as e:
        print("Error selecting teams:", e)
        return

    if t1 == t2:
        print("Choose two different teams.")
        return

    res = predict_scoreline(teams[t1], teams[t2], max_goals=6)
    l1, l2 = res['lambda']
    print(f"\nPrediction for {t1} vs {t2}:")
    print(f"Expected goals (lambda): {t1}: {l1:.2f}, {t2}: {l2:.2f}")
    print(f"Most likely score: {res['expected'][0]} - {res['expected'][1]} (prob {res['prob']:.4f})")
    print("Top scorelines:")
    for (g1, g2), p in res['top']:
        print(f"  {g1}-{g2}: {p:.4f}")


if __name__ == '__main__':
    main()
