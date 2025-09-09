import math
import json
from datetime import date, timedelta
import urllib.parse
import requests
import streamlit as st

# =========================================================
# Utilities
# =========================================================

def haversine_km(lat1, lon1, lat2, lon2):
    # distance between two lat/lon pairs in KM
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def deeplink_booking(query_text: str, checkin: date, checkout: date, adults: int = 2):
    q = urllib.parse.quote(query_text)
    return (
        "https://www.booking.com/searchresults.html"
        f"?ss={q}&checkin={checkin:%Y-%m-%d}&checkout={checkout:%Y-%m-%d}"
        f"&group_adults={adults}&no_rooms=1&group_children=0"
    )

def combined_query(city: str, area: str | None) -> str:
    return (f"{city} {area}".strip() if area else city).strip()

# =========================================================
# Geocoding & POIs (OpenStreetMap)
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def geocode_osm(query: str):
    """Geocode a place via OSM Nominatim (no key)."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": query, "format": "json", "limit": 1}
        headers = {"User-Agent": "MonTravels/1.0 (contact@montravels.app)"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        js = r.json() or []
        if js:
            return {
                "lat": float(js[0]["lat"]),
                "lon": float(js[0]["lon"]),
                "name": js[0]["display_name"]
            }
    except Exception:
        pass
    return None

OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"

OSM_TARGETS = {
    # tourism=attraction also catches famous landmarks/monuments
    "landmark": [
        ('tourism', 'attraction'),
        ('historic', '~.*'),       # any historic
        ('building', 'cathedral'),
        ('amenity', 'place_of_worship'),
        ('man_made', 'tower'),
    ],
    "museum": [
        ('tourism', 'museum')
    ],
    "park": [
        ('leisure', 'park'),
        ('leisure', 'garden'),
        ('natural', 'wood'),
        ('landuse', 'recreation_ground'),
    ],
    "cafe": [
        ('amenity', 'cafe'),
        ('amenity', 'fast_food')
    ],
    "restaurant": [
        ('amenity', 'restaurant'),
        ('amenity', 'food_court')
    ],
    "viewpoint": [
        ('tourism', 'viewpoint'),
        ('natural', 'peak'),
        ('tourism', 'information')  # sometimes scenic info points
    ],
}

def build_overpass_query(lat, lon, radius_m, kv_pairs):
    # kv_pairs is list of (key, value) where value may be literal or regex (~)
    parts = []
    for k, v in kv_pairs:
        if v.startswith('~'):
            parts.append(f'node["{k}"{v}](around:{radius_m},{lat},{lon});')
            parts.append(f'way["{k}"{v}](around:{radius_m},{lat},{lon});')
            parts.append(f'relation["{k}"{v}](around:{radius_m},{lat},{lon});')
        else:
            parts.append(f'node["{k}"="{v}"](around:{radius_m},{lat},{lon});')
            parts.append(f'way["{k}"="{v}"](around:{radius_m},{lat},{lon});')
            parts.append(f'relation["{k}"="{v}"](around:{radius_m},{lat},{lon});')
    core = "\n".join(parts)
    q = f"""
    [out:json][timeout:30];
    (
      {core}
    );
    out center 60;
    """
    return q

@st.cache_data(ttl=900, show_spinner=False)
def fetch_pois(lat, lon, radius_m=2500, kind="landmark", limit=30):
    """Fetch POIs from Overpass for a given category near lat/lon."""
    try:
        kv = OSM_TARGETS.get(kind, [])
        if not kv:
            return []
        query = build_overpass_query(lat, lon, radius_m, kv)
        r = requests.post(OVERPASS_ENDPOINT, data={"data": query}, timeout=40)
        r.raise_for_status()
        elements = r.json().get("elements", [])
        out = []
        seen_names = set()
        for e in elements:
            tags = e.get("tags", {})
            name = tags.get("name")
            # Some objects may lack names; skip
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            center = e.get("center", {})
            elat = center.get("lat", e.get("lat"))
            elon = center.get("lon", e.get("lon"))
            if elat is None or elon is None:
                continue
            out.append({
                "name": name,
                "lat": float(elat),
                "lon": float(elon),
                "tags": tags
            })
        # simple de-dup and trim
        return out[:limit]
    except Exception:
        return []

def pick_unique(pois, n, used_names, origin):
    """Pick up to n POIs not used yet, closest-first to origin (lat,lon)."""
    if not pois:
        return []
    olat, olon = origin
    enriched = []
    for p in pois:
        if p["name"] in used_names:
            continue
        d = haversine_km(olat, olon, p["lat"], p["lon"])
        enriched.append((d, p))
    enriched.sort(key=lambda x: x[0])
    chosen = [p for _, p in enriched[:n]]
    for c in chosen:
        used_names.add(c["name"])
    return chosen

# =========================================================
# Budget (amount/day) policy
# =========================================================

def budget_notes(amount: int) -> str:
    if amount < 50:
        return f"""
**Budget (~${amount}/day)**
- Prioritize free landmarks, mosques/temples, and public parks.
- Street food & local cafés (~$5–10/meal).
- Public transport; avoid pricey tours unless must-see.
"""
    elif amount < 150:
        return f"""
**Budget (~${amount}/day)**
- Mix of free & ticketed attractions (~$10–20 entry).
- Casual sit-down restaurants (~$15–30/meal).
- A couple of guided activities during the trip.
"""
    else:
        return f"""
**Budget (~${amount}/day)**
- Premium/unique attractions ($50+), private or small-group tours.
- Fine dining ($50–100+/meal).
- Upscale neighborhoods & experiences.
"""

def budget_profile(amount: int):
    """
    Decide how many 'paid' vs 'free' items to include per day.
    We approximate:
      landmarks/viewpoints/parks = free
      museums = likely paid
      restaurants/cafes = food cost driven by budget
    """
    if amount < 50:
        return {"museums_per_day": 0, "food_style": "cheap"}
    elif amount < 150:
        return {"museums_per_day": 1, "food_style": "mid"}
    else:
        return {"museums_per_day": 2, "food_style": "fine"}

# =========================================================
# Personalized hotel recommender (offline archetypes)
# =========================================================

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

def score_archetype(arch, interests: list[str], amount: int, area_hint: str | None, user_interest_bias: set[str]):
    score = 0.0
    # direct interests
    overlap = len(set(i.lower() for i in interests) & set(arch["good_for"]))
    score += 2.0 * overlap
    # user historical interest bias
    if user_interest_bias:
        hist_overlap = len(user_interest_bias & set(arch["good_for"]))
        score += 1.0 * hist_overlap
    # budget tilt
    if amount < 50 and arch["key"] in {"central-midscale","family-aparthotel","business-chain"}:
        score += 1.5
    if 50 <= amount < 150 and arch["key"] in {"historic-boutique","central-midscale","family-aparthotel","design-hotel","business-chain"}:
        score += 1.8
    if amount >= 150 and arch["key"] in {"design-hotel","waterfront-view","historic-boutique"}:
        score += 2.2
    # area hint heuristics
    if area_hint:
        a = area_hint.lower()
        if any(x in a for x in ["old", "historic", "city", "downtown", "bazaar"]):
            if arch["key"] in {"historic-boutique","central-midscale","design-hotel"}:
                score += 1.2
        if any(x in a for x in ["beach", "bay", "marina", "park", "water", "lake", "river"]):
            if arch["key"] in {"waterfront-view","family-aparthotel"}:
                score += 1.2
        if any(x in a for x in ["night", "soho", "party", "club"]):
            if arch["key"] in {"trendy-nightlife","design-hotel"}:
                score += 1.2
    return score

def synthesize_hotel_cards(city: str, area: str | None, start: date, end: date,
                           adults: int, interests: list[str], amount: int,
                           user_interest_bias: set[str], k: int = 8):
    area_txt = (area or "").strip()
    base_query = combined_query(city, area_txt or None)
    archetypes = sorted(
        ARCHETYPES,
        key=lambda a: score_archetype(a, interests, amount, area_txt, user_interest_bias),
        reverse=True
    )
    out = []
    for a in archetypes[:k]:
        why = []
        matched = set(i.lower() for i in interests) & set(a["good_for"])
        if matched:
            why.append("interests: " + ", ".join(sorted(matched)))
        if user_interest_bias:
            hist_match = user_interest_bias & set(a["good_for"])
            if hist_match:
                why.append("history: " + ", ".join(sorted(hist_match)))
        if area_txt:
            why.append(f"good near **{area_txt}**")
        # budget note
        if amount < 50: why.append("budget: value")
        elif amount < 150: why.append("budget: mid")
        else: why.append("budget: premium")

        out.append({
            "title": a["title"],
            "why": " • ".join(why),
            "tags": a["tags"],
            "link": deeplink_booking(base_query, start, end, adults)
        })
    return out

# =========================================================
# Personalized Itinerary Builder (with real POIs)
# =========================================================

def assemble_itinerary(lat, lon, city, area, start_date, end_date, interests, amount):
    days = max((end_date - start_date).days, 1)
    profile = budget_profile(amount)
    museums_per_day = profile["museums_per_day"]
    food_style = profile["food_style"]  # cheap / mid / fine

    # Fetch pools of places (once)
    pools = {}
    for k in ["landmark", "museum", "park", "cafe", "restaurant", "viewpoint"]:
        pools[k] = fetch_pois(lat, lon, radius_m=3000, kind=k, limit=50)

    # Heuristics for food style: prefer café vs restaurant
    def pick_food(used, origin):
        if food_style == "cheap":
            # prioritize cafés
            picks = pick_unique(pools["cafe"], 1, used, origin)
            if not picks:
                picks = pick_unique(pools["restaurant"], 1, used, origin)
        elif food_style == "mid":
            # mix
            picks = pick_unique(pools["restaurant"], 1, used, origin) or pick_unique(pools["cafe"], 1, used, origin)
        else:
            # fine -> restaurant first
            picks = pick_unique(pools["restaurant"], 1, used, origin) or pick_unique(pools["cafe"], 1, used, origin)
        return picks

    # Build days
    used_names = set()
    origin = (lat, lon)
    days_out = []
    for d in range(days):
        morning = []
        afternoon = []
        evening = []

        # Morning: 1 landmark + optional museum (budget allows)
        morning += pick_unique(pools["landmark"], 1, used_names, origin)
        if museums_per_day >= 1:
            morning += pick_unique(pools["museum"], 1, used_names, origin)

        # Afternoon: park + café/restaurant
        afternoon += pick_unique(pools["park"], 1, used_names, origin)
        afternoon += pick_food(used_names, origin)

        # Evening: viewpoint or landmark + restaurant/café
        ev_view = pick_unique(pools["viewpoint"], 1, used_names, origin)
        if not ev_view:
            ev_view = pick_unique(pools["landmark"], 1, used_names, origin)
        evening += ev_view
        evening += pick_food(used_names, origin)

        day_plan = {
            "Morning": morning,
            "Afternoon": afternoon,
            "Evening": evening
        }
        days_out.append(day_plan)

    header = f"## {city}" + (f" ({area})" if area else "") + f" — {days}-Day Itinerary"
    return header, days_out

def render_itinerary_markdown(header, days_plan):
    lines = [header]
    for idx, slots in enumerate(days_plan, start=1):
        lines.append(f"\n### Day {idx}")
        for part in ["Morning", "Afternoon", "Evening"]:
            items = slots.get(part, [])
            if not items:
                continue
            names = ", ".join([i["name"] for i in items])
            lines.append(f"- **{part}**: {names}")
    return "\n".join(lines)

# =========================================================
# User history (per-user via st.experimental_user)
# =========================================================

def get_user_id():
    try:
        user = st.experimental_user or {}
    except Exception:
        user = {}
    uid = user.get("id") or "guest"
    return str(uid)

def get_user_history(uid: str):
    if "history" not in st.session_state:
        st.session_state["history"] = {}
    if uid not in st.session_state["history"]:
        st.session_state["history"][uid] = []
    return st.session_state["history"][uid]

def add_history(uid: str, record: dict):
    hist = get_user_history(uid)
    hist.append(record)
    st.session_state["history"][uid] = hist

def derive_interest_bias(uid: str) -> set[str]:
    hist = get_user_history(uid)
    freq = {}
    for trip in hist:
        for i in trip.get("interests", []):
            k = i.lower()
            freq[k] = freq.get(k, 0) + 1
    # take top 3 interests as bias
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
    return set(k for k, _ in top)

# =========================================================
# UI
# =========================================================

st.set_page_config(page_title="MonTravels — Personalized Planner", page_icon="🧭", layout="wide")
st.title("🧭 MonTravels")

with st.sidebar:
    st.header("Trip Inputs")
    city = st.text_input("Destination city*", placeholder="e.g., Istanbul, Dubai, Karachi").strip()
    area = st.text_input("Area / neighborhood (optional)", placeholder="e.g., Sultanahmet, Marina, Clifton").strip()
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start date", value=date.today() + timedelta(days=14))
    with c2:
        end_date = st.date_input("End date", value=date.today() + timedelta(days=18))
    adults = st.number_input("Adults", min_value=1, max_value=10, value=2, step=1)
    budget_amount = st.number_input("Budget ($/day)", min_value=10, max_value=1000, value=100, step=5)

    interests = st.multiselect(
        "Interests",
        ["food","history","museums","nature","nightlife","architecture","shopping","family"],
        default=["food","history"]
    )

    go = st.button("✨ Build Personalized Plan")

uid = get_user_id()
st.caption(f"User: `{uid}`")

if go:
    # Basic validations
    if not city:
        st.error("Please enter a destination city.")
        st.stop()
    if end_date <= start_date:
        st.error("End date must be after Start date.")
        st.stop()

    # Geocode the specific area if given, else the city
    q = combined_query(city, area or None)
    geo = geocode_osm(q) or geocode_osm(city)
    if not geo:
        st.error("Could not geolocate that place. Try a simpler query (just the city).")
        st.stop()

    st.caption(f"📍 {geo['name']}  ({geo['lat']:.4f}, {geo['lon']:.4f})")

    # Build itinerary with real POIs
    with st.status("Finding nearby places & crafting itinerary...", expanded=False):
        header, days_plan = assemble_itinerary(
            geo["lat"], geo["lon"], city, (area or "").strip(),
            start_date, end_date, interests, budget_amount
        )
    st.subheader("🗓️ Your Itinerary")
    st.markdown(render_itinerary_markdown(header, days_plan))

    # Budget summary (once, at the end)
    st.markdown(budget_notes(int(budget_amount)))
    st.caption("*Note: flight & visa costs are not included.*")

    # Personalized hotel archetype suggestions (offline)
    user_bias = derive_interest_bias(uid)
    st.subheader("🏨 Recommended Places to Stay (Personalized)")
    hotel_cards = synthesize_hotel_cards(
        city, (area or None), start_date, end_date, adults, interests, int(budget_amount), user_bias, k=8
    )
    for c in hotel_cards:
        with st.container(border=True):
            st.markdown(f"**{c['title']}**")
            st.caption(c["why"])
            st.write("Tags:", ", ".join(c["tags"]))
            st.link_button("Open on Booking.com", c["link"])

    # Save trip to history
    add_history(uid, {
        "city": city,
        "area": (area or "").strip(),
        "start": f"{start_date}",
        "end": f"{end_date}",
        "adults": adults,
        "budget": int(budget_amount),
        "interests": interests
    })

    # Export
    pkg = {
        "itinerary_header": header,
        "itinerary": days_plan,
        "budget_per_day": int(budget_amount),
        "budget_notes": budget_notes(int(budget_amount)),
        "stay_recommendations": hotel_cards,
        "deeplink_city": deeplink_booking(q, start_date, end_date, adults)
    }
    st.download_button("⬇️ Download Plan (JSON)",
                       data=json.dumps(pkg, ensure_ascii=False, indent=2),
                       file_name=f"montravels_{city.lower().replace(' ','_')}.json",
                       mime="application/json")

    # Show history
    st.subheader("🧠 Your Saved History (Private to this user)")
    st.json(get_user_history(uid))

else:
    st.info("Enter a city (and optional area), choose dates & budget, select interests, then click **Build Personalized Plan**.")
