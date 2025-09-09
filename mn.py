import urllib.parse
from datetime import date, timedelta
import streamlit as st

# -----------------------------
# Simple helpers
# -----------------------------
def deeplink_booking(query_text: str, checkin: date, checkout: date, adults: int = 2):
    q = urllib.parse.quote(query_text)
    return (
        "https://www.booking.com/searchresults.html"
        f"?ss={q}&checkin={checkin:%Y-%m-%d}&checkout={checkout:%Y-%m-%d}"
        f"&group_adults={adults}&no_rooms=1&group_children=0"
    )

def combined_query(city: str, area: str | None) -> str:
    return (f"{city} {area}".strip() if area else city).strip()

# -----------------------------
# Offline Itinerary (no APIs)
# -----------------------------
def build_itinerary(city: str, start_date: date, end_date: date, area: str | None, interests: list[str], budget: str):
    days = max((end_date - start_date).days, 1)
    loc = f"{city} ({area})" if area else city
    interest_str = ", ".join(interests) if interests else "sightseeing"
    lines = [f"## {loc} — {days}-Day Itinerary (Budget: {budget})",
             f"_Interests: {interest_str}_"]
    daily_blocks = [
        ("Morning",  "iconic landmark & neighborhood walk"),
        ("Afternoon","museum/park + local café"),
        ("Evening",  "viewpoint / riverfront + dinner")
    ]
    for i in range(days):
        lines.append(f"\n### Day {i+1}")
        for part, plan in daily_blocks:
            lines.append(f"- **{part}**: {plan}")
        lines.append("- **Cost guide**: " + {"tight":"$", "moderate":"$$", "luxury":"$$$"}[budget])
    lines += [
        "\n**Tips**",
        "- Get a day transport pass; markets and small eateries may be cash-first.",
        "- Check opening hours; prebook popular spots.",
        "- Respect local etiquette; pack for the weather."
    ]
    return "\n".join(lines)

# -----------------------------
# Offline Recommendation Engine
# -----------------------------
BUDGET_BANDS = {
    "tight":    {"label": "$",   "target_ppn": (30, 90)},
    "moderate": {"label": "$$",  "target_ppn": (90, 180)},
    "luxury":   {"label": "$$$", "target_ppn": (180, 450)}
}

# Archetype templates we can mix & match without live data.
ARCHETYPES = [
    {"key":"historic-boutique", "title":"Boutique near Old Town",
     "tags":["walkable","character"], "good_for":["history","museums","architecture"]},
    {"key":"central-midscale", "title":"Midscale near City Center",
     "tags":["convenient","transport"], "good_for":["shopping","food","architecture","history"]},
    {"key":"trendy-nightlife", "title":"Trendy spot in Nightlife District",
     "tags":["bars","music"], "good_for":["nightlife","food","shopping"]},
    {"key":"family-aparthotel", "title":"Aparthotel in Family Area",
     "tags":["kitchen","space"], "good_for":["family","nature","shopping"]},
    {"key":"waterfront-view", "title":"Waterfront / Park-side Hotel",
     "tags":["views","quiet"], "good_for":["nature","architecture","family"]},
    {"key":"business-chain", "title":"Reliable Business Chain near Metro",
     "tags":["quiet","clean"], "good_for":["shopping","history","architecture","food"]},
    {"key":"design-hotel", "title":"Design-Led Hotel near Arts District",
     "tags":["aesthetic","boutiques"], "good_for":["architecture","museums","shopping","nightlife"]},
]

def score_archetype(arch, interests: list[str], budget: str, area_hint: str | None):
    score = 0.0
    # Interest alignment
    if interests:
        overlap = len(set(i.lower() for i in interests) & set(arch["good_for"]))
        score += 2.0 * overlap
    else:
        score += 1.0  # neutral default
    # Budget fit (some archetypes lean to budgets)
    if budget == "tight" and arch["key"] in {"central-midscale","family-aparthotel","business-chain"}:
        score += 2.0
    if budget == "moderate" and arch["key"] in {"historic-boutique","central-midscale","family-aparthotel","design-hotel","business-chain"}:
        score += 2.0
    if budget == "luxury" and arch["key"] in {"design-hotel","waterfront-view","historic-boutique"}:
        score += 2.5
    # Area hint
    if area_hint:
        area_l = area_hint.lower()
        if "old" in area_l or "historic" in area_l or "city" in area_l or "downtown" in area_l:
            if arch["key"] in {"historic-boutique","central-midscale","design-hotel"}:
                score += 1.5
        if "beach" in area_l or "bay" in area_l or "marina" in area_l or "park" in area_l or "water" in area_l:
            if arch["key"] in {"waterfront-view","family-aparthotel"}:
                score += 1.5
        if "night" in area_l or "soho" in area_l or "party" in area_l:
            if arch["key"] in {"trendy-nightlife","design-hotel"}:
                score += 1.5
        if "family" in area_l or "residential" in area_l:
            if arch["key"] in {"family-aparthotel","business-chain"}:
                score += 1.5
    return score

def synthesize_hotel_cards(city: str, area: str | None, start: date, end: date,
                           adults: int, interests: list[str], budget: str, k: int = 10):
    """
    Produce k archetype-based hotel suggestions with explanations and a Booking.com link.
    """
    area_txt = (area or "").strip()
    base_query = combined_query(city, area_txt if area_txt else None)
    price_band = BUDGET_BANDS[budget]["label"]
    ppn_low, ppn_high = BUDGET_BANDS[budget]["target_ppn"]

    # Rank archetypes
    ranked = sorted(
        ARCHETYPES,
        key=lambda a: score_archetype(a, interests, budget, area_txt),
        reverse=True
    )

    out = []
    for a in ranked[:k]:
        why_bits = []
        if interests:
            matched = set(i.lower() for i in interests) & set(a["good_for"])
            if matched:
                why_bits.append(f"matches interests: {', '.join(sorted(matched))}")
        if area_txt:
            why_bits.append(f"good fit for **{area_txt}**")
        why_bits.append(f"budget target {price_band}")
        why = " • ".join(why_bits)

        out.append({
            "title": f"{a['title']} — {price_band}",
            "why": why,
            "tags": a["tags"],
            "link": deeplink_booking(base_query, start, end, adults)
        })
    return out

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="MonTravels — Itinerary & Recommendations", page_icon="🧭", layout="wide")
st.title("🧭 MonTravels")

with st.sidebar:
    st.header("Trip Inputs")
    city = st.text_input("Destination city*", placeholder="e.g., London, Karachi, Dubai").strip()
    area = st.text_input("Area / neighborhood (optional)", placeholder="e.g., Clifton, Soho, Marina").strip()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=date.today() + timedelta(days=14))
    with col2:
        end_date = st.date_input("End date", value=date.today() + timedelta(days=18))
    adults = st.number_input("Adults", min_value=1, max_value=10, value=2, step=1)

    st.markdown("**Interests**")
    interests = st.multiselect(
        "",
        ["history", "museums", "food", "nature", "nightlife", "architecture", "shopping", "family"],
        default=["food", "history"]
    )
    budget = st.radio("Budget", options=["tight", "moderate", "luxury"], index=1, horizontal=True)

    go = st.button("✨ Build Plan")

if go:
    if not city:
        st.error("Please enter a destination city.")
    elif end_date <= start_date:
        st.error("End date must be after Start date.")
    else:
        # Itinerary
        st.subheader("🗓️ Your Itinerary")
        st.markdown(build_itinerary(city, start_date, end_date, area or None, interests, budget))

        # Recommendations (offline engine)
        st.subheader("🏨 Recommended Places to Stay (Model Suggestions)")
        cards = synthesize_hotel_cards(city, area or None, start_date, end_date, adults, interests, budget, k=10)
        for c in cards:
            with st.container(border=True):
                st.markdown(f"**{c['title']}**")
                st.caption(c["why"])
                st.write("Tags:", ", ".join(c["tags"]))
                st.link_button("Open on Booking.com", c["link"])

        # Global link too
        st.link_button("🔗 See full results on Booking.com",
                       deeplink_booking(combined_query(city, area or None), start_date, end_date, adults))
else:
    st.info("Enter a city (optionally an area), choose dates & preferences, then click **Build Plan**.")
