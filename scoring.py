"""
Scoring and ranking engine for Season Radar.

Computes a weighted score for each city based on user preferences:
  - Temperature match
  - Precipitation tolerance
  - Crowd level preference
  - Environment tag affinity
"""

import math
from typing import Optional


# ─── Component Scorers ───────────────────────────────────────────────────────

def score_temperature(city_temp: float, temp_min: Optional[float], temp_max: Optional[float]) -> float:
    """
    Score 0–1 based on how well the city temperature matches the preferred range.
    Uses a Gaussian decay outside the range, full score inside.
    """
    if temp_min is None and temp_max is None:
        return 0.65  # neutral — no preference stated

    # Fill missing bound with a generous default
    if temp_min is None:
        temp_min = city_temp - 10
    if temp_max is None:
        temp_max = city_temp + 10

    # Clamp so min <= max
    temp_min, temp_max = min(temp_min, temp_max), max(temp_min, temp_max)

    # Inside preferred range → high score
    if temp_min <= city_temp <= temp_max:
        # Small bonus for being near the midpoint
        mid = (temp_min + temp_max) / 2
        half_range = max((temp_max - temp_min) / 2, 1)
        closeness = 1.0 - 0.15 * abs(city_temp - mid) / half_range
        return round(max(0.85, closeness), 4)

    # Outside range — penalise proportionally to how far outside
    gap = max(temp_min - city_temp, city_temp - temp_max)
    sigma = max((temp_max - temp_min) / 2, 3)  # min sigma of 3°C
    score = math.exp(-0.5 * (gap / sigma) ** 2)
    return round(max(0.0, score), 4)


def score_precipitation(monthly_precip: float, rain_tolerance: str) -> float:
    """
    Score 0–1 based on precipitation.
    rain_tolerance: 'low' | 'medium' | 'high'
    """
    # Normalise precip: 0mm → 1.0, 300mm+ → 0.0
    base = max(0.0, 1.0 - monthly_precip / 300.0)

    if rain_tolerance == "low":
        # Amplify — dry travellers strongly prefer low rain
        return round(base ** 0.6, 4)
    elif rain_tolerance == "high":
        # Dampen — rain-tolerant travellers barely care
        return round(0.5 + base * 0.5, 4)
    else:
        # medium — linear
        return round(base, 4)


def score_crowd(month: int, peak_months: list, shoulder_months: list, crowd_preference: str) -> float:
    """
    Score 0–1 based on season type vs user preference.
    """
    if month in peak_months:
        season = "peak"
    elif month in shoulder_months:
        season = "shoulder"
    else:
        season = "off"

    table = {
        "off_peak":  {"off": 1.0, "shoulder": 0.55, "peak": 0.05},
        "shoulder":  {"shoulder": 1.0, "off": 0.7,  "peak": 0.25},
        "any":       {"shoulder": 1.0, "off": 0.85,  "peak": 0.75},
    }
    return table.get(crowd_preference, table["any"])[season]


def score_tags(city_tags: list, preferred_tags: list) -> float:
    """
    Score 0–1 based on overlap between city tags and user's preferred environment types.
    """
    if not preferred_tags:
        return 0.65  # neutral

    if not city_tags:
        return 0.2

    city_tags_l    = {t.lower() for t in city_tags}
    preferred_tags_l = [t.lower() for t in preferred_tags]

    # Count exact matches + partial substring matches
    matches = sum(
        1 for pt in preferred_tags_l
        if pt in city_tags_l or any(pt in ct or ct in pt for ct in city_tags_l)
    )
    return round(min(1.0, matches / len(preferred_tags_l)), 4)


# ─── Main Ranking Function ────────────────────────────────────────────────────

def rank_cities(cities: list, preferences: dict, top_n: int = 8) -> list:
    """
    Score and rank all cities against user preferences.

    preferences keys (all optional except travel_month, crowd_preference):
      travel_month      int   1–12
      temp_min          float °C
      temp_max          float °C
      rain_tolerance    str   'low' | 'medium' | 'high'
      crowd_preference  str   'off_peak' | 'shoulder' | 'any'
      environment_tags  list  e.g. ['beach', 'city']
      exclude_regions   list  e.g. ['Europe']
      num_results       int   default top_n
    """
    month_0        = preferences.get("travel_month", 1) - 1   # convert to 0-index
    temp_min       = preferences.get("temp_min")
    temp_max       = preferences.get("temp_max")
    rain_tolerance = preferences.get("rain_tolerance", "medium")
    crowd_pref     = preferences.get("crowd_preference", "any")
    pref_tags      = preferences.get("environment_tags", [])
    exclude        = [r.lower() for r in preferences.get("exclude_regions", [])]
    num_results    = preferences.get("num_results", top_n)

    # Weight profile — must sum to 1.0
    W_TEMP  = 0.40
    W_RAIN  = 0.30
    W_CROWD = 0.20
    W_TAGS  = 0.10

    results = []

    for city in cities:
        # Apply region/country exclusion filter
        region  = city.get("region", "").lower()
        country = city.get("country", "").lower()
        if any(ex in region or ex in country for ex in exclude):
            continue

        city_temp   = city["monthly_temp"][month_0]
        city_precip = city["monthly_precip"][month_0]
        peak_months     = city.get("peak_months", [])
        shoulder_months = city.get("shoulder_months", [])

        # Component scores
        t_score = score_temperature(city_temp, temp_min, temp_max)
        r_score = score_precipitation(city_precip, rain_tolerance)
        c_score = score_crowd(month_0 + 1, peak_months, shoulder_months, crowd_pref)
        g_score = score_tags(city.get("tags", []), pref_tags)

        final = round(
            W_TEMP * t_score
            + W_RAIN * r_score
            + W_CROWD * c_score
            + W_TAGS * g_score,
            4,
        )

        # Determine human-readable season label
        m = month_0 + 1
        if m in peak_months:
            season_label = "peak season"
        elif m in shoulder_months:
            season_label = "shoulder season"
        else:
            season_label = "off season"

        results.append({
            "city": city,
            "scores": {
                "temp":   t_score,
                "rain":   r_score,
                "crowd":  c_score,
                "tags":   g_score,
                "final":  final,
            },
            "month_data": {
                "temp":   city_temp,
                "precip": city_precip,
                "season": season_label,
            },
        })

    results.sort(key=lambda x: x["scores"]["final"], reverse=True)
    return results[:num_results]


def format_results_for_claude(ranked_results: list, month_name: str) -> str:
    """Format ranked city data as structured context for Claude to reason over."""
    if not ranked_results:
        return f"[No destinations matched the criteria for {month_name}. Suggest broadening preferences.]"

    lines = [f"[DATASET: TOP DESTINATIONS FOR {month_name.upper()}]", ""]

    for i, result in enumerate(ranked_results, 1):
        city       = result["city"]
        scores     = result["scores"]
        month_data = result["month_data"]

        lines.append(
            f"{i}. {city['name']}, {city['country']}  "
            f"[Score: {scores['final']:.2f}]"
        )
        lines.append(
            f"   Avg temp: {month_data['temp']}°C | "
            f"Precipitation: {month_data['precip']}mm | "
            f"Status: {month_data['season']}"
        )
        lines.append(
            f"   Score breakdown — temp:{scores['temp']:.2f}  "
            f"rain:{scores['rain']:.2f}  crowd:{scores['crowd']:.2f}  "
            f"tags:{scores['tags']:.2f}"
        )
        lines.append(f"   Tags: {', '.join(city.get('tags', []))}")
        lines.append("")

    lines.append(
        "[INSTRUCTION: Use the data above to explain your recommendations. "
        "Cite actual temperatures and crowd status. "
        "Do NOT invent data not shown above.]"
    )
    return "\n".join(lines)
