# mn.py

import math
import json
import time
import hashlib
from datetime import date, timedelta
import urllib.parse
import requests
import streamlit as st
from typing import Optional, List, Set, Dict, Any, Tuple

# =========================================================
# THEME (Pokémon-inspired Travel Guide)
# =========================================================
def apply_pokemon_theme():
    st.markdown("""
        <style>
        .stApp { background-color: #F5F7FA; font-family: 'Trebuchet MS', sans-serif; color: #2C2C2C; }
        h1 { color: #FFCC00; text-shadow: 2px 2px 0px #3B4CCA; }
        h2, h3 { color: #3B4CCA; }

        /* Sidebar */
        section[data-testid="stSidebar"] { background-color: #3B4CCA; color: white; }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span[role="img"] { color: white !important; }

        /* Inputs readable */
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] select,
        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] .stNumberInput input,
        section[data-testid="stSidebar"] .stDateInput input,
        section[data-testid="stSidebar"] .stMultiSelect input {
            color: #111 !important; background-color: #F0F3FF !important; border-radius: 10px !important;
        }
        section[data-testid="stSidebar"] ::placeholder { color: #2C2C2C !important; opacity: 0.8; }

        /* Multiselect chips */
        section[data-testid="stSidebar"] div[data-baseweb="tag"] {
            background: #FFCC00 !important; color: #2C2C2C !important; border-radius: 10px !important; font-weight: 600;
        }

        /* Buttons / links */
        .btn-link {
            display:inline-block; background:#FF1C1C; color:#fff !important; border-radius:12px;
            border:2px solid #3B4CCA; font-weight:bold; padding:8px 14px; text-decoration:none !important;
            transition:.3s; cursor:pointer;
        }
        .btn-link:hover { background:#FFCC00; color:#2C2C2C !important; border-color:#FF1C1C; }

        div.stButton>button {
            background:#FF1C1C; color:#fff; border-radius:12px; border:2px solid #3B4CCA; font-weight:bold; padding:8px 14px;
            transition:.3s;
        }
        div.stButton>button:hover { background:#FFCC00; color:#2C2C2C; border:2px solid #FF1C1C; }

        /* Card */
        .pokecard {
            background:#fff; border-radius:16px; padding:12px; margin-bottom:12px; border:2px solid #FFCC00;
            box-shadow:2px 2px 6px rgba(0,0,0,0.1);
        }

        a { color:#3B4CCA; font-weight:bold; text-decoration:none; }
        a:hover { color:#FF1C1C; }
        .stCaption, span[data-baseweb="tag"] { color:#4CAF50 !important; }
        </style>
    """, unsafe_allow_html=True)

# =========================================================
# Helpers
# =========================================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def _cachebuster(seed: str) -> str:
    raw = f"{seed}-{time.time_ns()}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]

def parse_city_country(display_name: str) -> Tuple[str, Optional[str]]:
    """
    Very light parser: Booking understands 'City, Country' reliably.
    display_name is like 'Clifton, Karachi, Sindh, Pakistan'.
    We return (city_like, country).
    """
    parts = [p.strip() for p in display_name.split(",")]
    if len(parts) >= 2:
        country = parts[-1]
        # choose a 'city-like' piece near the end (avoid micro-neighborhoods)
        city_like = parts[-2]
        return city_like, country
    return display_name, None

def build_precise_ss(city: str, area: Optional[str], geo: Dict[str, Any]) -> str:
    city_guess, country = parse_city_country(geo.get("name", city))
    if area:
        base = f"{area}, {city}"
    else:
        base = city
    if country:
        return f"{base}, {country}"
    return base

def external_link_button(label: str, url: str):
    st.markdown(f'<a class="btn-link" target="_blank" rel="noopener" href="{url}">{label}</a>',
                unsafe_allow_html=True)

# =========================================================
# Booking deep links (location-first, softer filters)
# =========================================================
def deeplink_booking_city(precise_ss: str, lat: float, lon: float,
                          checkin: date, checkout: date, adults: int = 2) -> str:
    ss_q = urllib.parse.quote(precise_ss)
    return (
        "https://www.booking.com/searchresults.html"
        f"?ss={ss_q}"
        f"&checkin={checkin:%Y-%m-%d}"
        f"&checkout={checkout:%Y-%m-%d}"
        f"&group_adults={adults}&no_rooms=1&group_children=0"
        f"&lang=en-us&src=index&sb=1"
        # soft radius hint around coordinates (Booking often honors this)
        f"&latitude={lat:.6f}&longitude={lon:.6f}&nflt=distance%3D0-5"
        f"&_mtu={_cachebuster(precise_ss)}"
    )

def deeplink_booking_card(precise_ss: str, lat: float, lon: float,
                          checkin: date, checkout: date, adults: int = 2) -> str:
    # For each card, keep ss identical (accurate location); let results differ naturally by Booking sorting.
    return deeplink_booking_city(precise_ss, lat, lon, checkin, checkout, adults)

# =========================================================
# Geocoding & POIs (OpenStreetMap)
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_osm(query: str) -> Optional[Dict[str, Any]]:
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": query, "format": "json", "limit": 1}
        headers = {"User-Agent": "MonTravels/1.0 (contact@montravels.app)"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        js = r.json() or []
        if js:
            return {"lat": float(js[0]["lat"]), "lon": float(js[0]["lon"]), "name": js[0]["display_name"]}
    except Exception:
        pass
    return None

OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"

OSM_TARGETS = {
    "landmark": [
        ('tourism', 'attraction'),
        ('historic', '~.*'),
        ('building', 'cathedral'),
        ('amenity', 'place_of_worship'),
        ('man_made', 'tower'),
    ],
    "museum": [('tourism', 'museum')],
    "park": [
        ('leisure', 'park'),
        ('leisure', 'garden'),
        ('natural', 'wood'),
        ('landuse', 'recreation_ground'),
    ],
    "cafe": [('amenity', 'cafe'), ('amenity', 'fast_food')],
    "restaurant": [('amenity', 'restaurant'), ('amenity', 'food_court')],
    "viewpoint": [('tourism', 'viewpoint'), ('natural', 'peak'), ('tourism', 'information')],
}

def build_overpass_query(lat: float, lon: float, radius_m: int, kv_pairs: List[List[str]]) -> str:
    parts: List[str] = []
    for k, v in kv_pairs:
        if str(v).startswith('~'):
            parts += [
                f'node["{k}"{v}](around:{radius_m},{lat},{lon});',
                f'way["{k}"{v}](around:{radius_m},{lat},{lon});',
                f'relation["{k}"{v}](around:{radius_m},{lat},{lon});',
            ]
        else:
            parts += [
                f'node["{k}"="{v}"](around:{radius_m},{lat},{lon});',
                f'way["{k}"="{v}"](around:{radius_m},{lat},{lon});',
                f'relation["{k}"="{v}"](around:{radius_m},{lat},{lon});',
            ]
    core = "\n".join(parts)
    return f"[out:json][timeout:30];({core});out center 60;"

@st.cache_data(ttl=900, show_spinner=False)
def fetch_pois(lat: float, lon: float, radius_m: int = 3000, kind: str = "landmark", limit: int = 50) -> List[Dict[str, Any]]:
    try:
        kv = OSM_TARGETS.get(kind, [])
        if not kv:
            return []
        q = build_overpass_query(lat, lon, radius_m, kv)
        r = requests.post(OVERPASS_ENDPOINT, data={"data": q}, timeout=40)
        r.raise_for_status()
        elements = r.json().get("elements", [])
        out, seen = [], set()
        for e in elements:
            tags = e.get("tags", {})
            name = tags.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            center = e.get("center", {})
            elat = center.get("lat", e.get("lat"))
            elon = center.get("lon", e.get("lon"))
            if elat is None or elon is None:
                continue
            out.append({"name": name, "lat": float(elat), "lon": float(elon), "tags": tags})
        return out[:limit]
    except Exception:
        return []

def pick_unique(pois: List[Dict[str, Any]], n: int, used_names: Set[str], origin: tuple) -> List[Dict[str, Any]]:
    if not pois: return []
    olat, olon = origin
    enriched = []
    for p in pois:
        if p["name"] in used_names: continue
        d = haversine_km(olat, olon, p["lat"], p["lon"])
        enriched.append((d, p))
    enriched.sort(key=lambda x: x[0])
    chosen = [p for _, p in enriched[:n]]
    for c in chosen: used_names.add(c["name"])
    return chosen

# =========================================================
# Budget
# =========================================================
def budget_notes(amount: int) -> str:
    if amount < 50:
        return f"""
**Budget (~${amount}/day)**
- Prioritize free landmarks and public parks.
- Street food & local cafés (~$5–10/meal).
- Use public transport; limit paid tours.
"""
    elif amount < 150:
        return f"""
**Budget (~${amount}/day)**
- Mix free & ticketed attractions (~$10–20 entry).
- Casual sit-down restaurants (~$15–30/meal).
- A couple of guided activities for the trip.
"""
    else:
        return f"""
**Budget (~${amount}/day)**
- Premium attractions ($50+), private/small-group tours.
- Fine dining ($50–100+/meal).
- Upscale neighborhoods & experiences.
"""

def budget_profile(amount: int) -> Dict[str, Any]:
    if amount < 50: return {"museums_per_day": 0, "food_style": "cheap"}
    elif amount < 150: return {"museums_per_day": 1, "food_style": "mid"}
    else: return {"museums_per_day": 2, "food_style": "fine"}

# =========================================================
# Hotels (archetypes)
# =========================================================
ARCHETYPES = [
    {"key":"historic-boutique","title":"Boutique near Old Town","tags":["walkable","character"],"good_for":["history","museums","architecture"]},
    {"key":"central-midscale","title":"Midscale near City Center","tags":["convenient","transport"],"good_for":["shopping","food","architecture","history"]},
    {"key":"trendy-nightlife","title":"Trendy spot in Nightlife District","tags":["bars","music"],"good_for":["nightlife","food","shopping"]},
    {"key":"family-aparthotel","title":"Aparthotel in Family Area","tags":["kitchen","space"],"good_for":["family","nature","shopping"]},
    {"key":"waterfront-view","title":"Waterfront / Park-side Hotel","tags":["views","quiet"],"good_for":["nature","architecture","family"]},
    {"key":"business-chain","title":"Reliable Business Chain near Metro","tags":["quiet","clean"],"good_for":["shopping","history","architecture","food"]},
    {"key":"design-hotel","title":"Design-Led Hotel near Arts District","tags":["aesthetic","boutiques"],"good_for":["architecture","museums","shopping","nightlife"]},
]

def score_archetype(arch: Dict[str, Any], interests: List[str], amount: int,
                    area_hint: Optional[str], user_interest_bias: Set[str]) -> float:
    score = 0.0
    overlap = len(set(i.lower() for i in interests) & set(arch["good_for"]))
    score += 2.0 * overlap
    if user_interest_bias: score += 1.0 * len(user_interest_bias & set(arch["good_for"]))
    if amount < 50 and arch["key"] in {"central-midscale","family-aparthotel","business-chain"}: score += 1.5
    if 50 <= amount < 150 and arch["key"] in {"historic-boutique","central-midscale","family-aparthotel","design-hotel","business-chain"}: score += 1.8
    if amount >= 150 and arch["key"] in {"design-hotel","waterfront-view","historic-boutique"}: score += 2.2
    if area_hint:
        a = area_hint.lower()
        if any(x in a for x in ["old","historic","city","downtown","bazaar"]) and arch["key"] in {"historic-boutique","central-midscale","design-hotel"}: score += 1.2
        if any(x in a for x in ["beach","bay","marina","park","water","lake","river"]) and arch["key"] in {"waterfront-view","family-aparthotel"}: score += 1.2
        if any(x in a for x in ["night","soho","party","club"]) and arch["key"] in {"trendy-nightlife","design-hotel"}: score += 1.2
    return score

def synthesize_hotel_cards(city: str, area: Optional[str], start: date, end: date,
                           adults: int, interests: List[str], amount: int,
                           user_interest_bias: Set[str], precise_ss: str, lat: float, lon: float,
                           k: int = 8) -> List[Dict[str, Any]]:
    area_txt = (area or "").strip()
    ranked = sorted(
        ARCHETYPES,
        key=lambda a: score_archetype(a, interests, amount, area_txt, user_interest_bias),
        reverse=True
    )
    out: List[Dict[str, Any]] = []
    for a in ranked[:k]:
        why = []
        matched = set(i.lower() for i in interests) & set(a["good_for"])
        if matched: why.append("interests: " + ", ".join(sorted(matched)))
        if user_interest_bias:
            hist_match = user_interest_bias & set(a["good_for"])
            if hist_match: why.append("history: " + ", ".join(sorted(hist_match)))
        if area_txt: why.append(f"good near **{area_txt}**")
        if amount < 50:      why.append("budget: value")
        elif amount < 150:   why.append("budget: mid")
        else:                why.append("budget: premium")

        link = deeplink_booking_card(
            precise_ss=precise_ss,
            lat=lat, lon=lon,
            checkin=start, checkout=end, adults=adults
        )

        out.append({
            "title": a["title"],
            "why": " • ".join(why),
            "tags": a["tags"],
            "link": link
        })
    return out

# =========================================================
# Itinerary builder
# =========================================================
def assemble_itinerary(lat: float, lon: float, city: str, area: str,
                       start_date: date, end_date: date,
                       interests: List[str], amount: int):
    days = max((end_date - start_date).days, 1)
    profile = budget_profile(amount)
    museums_per_day = profile["museums_per_day"]
    food_style = profile["food_style"]

    pools: Dict[str, List[Dict[str, Any]]] = {}
    for k in ["landmark", "museum", "park", "cafe", "restaurant", "viewpoint"]:
        pools[k] = fetch_pois(lat, lon, radius_m=3000, kind=k, limit=50)

    def pick_food(used, origin):
        if food_style == "cheap":
            picks = pick_unique(pools["cafe"], 1, used, origin) or pick_unique(pools["restaurant"], 1, used, origin)
        elif food_style == "mid":
            picks = pick_unique(pools["restaurant"], 1, used, origin) or pick_unique(pools["cafe"], 1, used, origin)
        else:
            picks = pick_unique(pools["restaurant"], 1, used, origin) or pick_unique(pools["cafe"], 1, used, origin)
        return picks

    used_names: Set[str] = set()
    origin = (lat, lon)
    days_out: List[Dict[str, Any]] = []
    for _ in range(days):
        morning: List[Dict[str, Any]] = []
        afternoon: List[Dict[str, Any]] = []
        evening: List[Dict[str, Any]] = []

        morning += pick_unique(pools["landmark"], 1, used_names, origin)
        if museums_per_day >= 1:
            morning += pick_unique(pools["museum"], 1, used_names, origin)

        afternoon += pick_unique(pools["park"], 1, used_names, origin)
        afternoon += pick_food(used_names, origin)

        ev_view = pick_unique(pools["viewpoint"], 1, used_names, origin) or pick_unique(pools["landmark"], 1, used_names, origin)
        evening += ev_view
        evening += pick_food(used_names, origin)

        days_out.append({"Morning": morning, "Afternoon": afternoon, "Evening": evening})

    header = f"## {city}" + (f" ({area})" if area else "") + f" — {days}-Day Itinerary"
    return header, days_out

def render_itinerary_markdown(header: str, days_plan: List[Dict[str, Any]]) -> str:
    lines = [header]
    for idx, slots in enumerate(days_plan, start=1):
        lines.append(f"\n### Day {idx}")
        for part in ["Morning", "Afternoon", "Evening"]:
            items = slots.get(part, [])
            if not items: continue
            names = ", ".join([i["name"] for i in items])
            lines.append(f"- **{part}**: {names}")
    return "\n".join(lines)

# =========================================================
# History (session)
# =========================================================
def get_user_id() -> str:
    try:
        user = st.experimental_user or {}
    except Exception:
        user = {}
    return str(user.get("id") or "guest")

def get_user_history(uid: str) -> List[Dict[str, Any]]:
    if "history" not in st.session_state: st.session_state["history"] = {}
    if uid not in st.session_state["history"]: st.session_state["history"][uid] = []
    return st.session_state["history"][uid]

def add_history(uid: str, record: dict):
    hist = get_user_history(uid); hist.append(record); st.session_state["history"][uid] = hist

def derive_interest_bias(uid: str) -> Set[str]:
    hist = get_user_history(uid)
    freq: Dict[str, int] = {}
    for trip in hist:
        for i in trip.get("interests", []):
            k = i.lower(); freq[k] = freq.get(k, 0) + 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
    return set(k for k, _ in top)

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="MonTravels — Personalized Planner", page_icon="🧭", layout="wide")
apply_pokemon_theme()

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
    if not city:
        st.error("Please enter a destination city."); st.stop()
    if end_date <= start_date:
        st.error("End date must be after Start date."); st.stop()

    # Geocode city/area, then build precise search string for Booking
    query = f"{area}, {city}" if area else city
    geo = geocode_osm(query) or geocode_osm(city)
    if not geo:
        st.error("Could not geolocate that place. Try a simpler query (just the city)."); st.stop()

    st.caption(f"📍 {geo['name']}  ({geo['lat']:.4f}, {geo['lon']:.4f})")

    precise_ss = build_precise_ss(city, area or None, geo)

    with st.status("Finding nearby places & crafting itinerary...", expanded=False):
        header, days_plan = assemble_itinerary(
            geo["lat"], geo["lon"], city, (area or "").strip(),
            start_date, end_date, interests, int(budget_amount)
        )

    st.subheader("🗓️ Your Itinerary")
    st.markdown(render_itinerary_markdown(header, days_plan))

    st.markdown(budget_notes(int(budget_amount)))
    st.caption("*Note: flight & visa costs are not included.*")

    user_bias = derive_interest_bias(uid)
    st.subheader("🏨 Recommended Places to Stay (Personalized)")
    hotel_cards = synthesize_hotel_cards(
        city=city,
        area=(area or None),
        start=start_date,
        end=end_date,
        adults=adults,
        interests=interests,
        amount=int(budget_amount),
        user_interest_bias=user_bias,
        precise_ss=precise_ss,
        lat=geo["lat"], lon=geo["lon"],
        k=8
    )
    for c in hotel_cards:
        with st.container(border=True):
            st.markdown('<div class="pokecard">', unsafe_allow_html=True)
            st.markdown(f"**{c['title']}**")
            st.caption(c["why"])
            st.write("Tags:", ", ".join(c["tags"]))
            external_link_button("Open on Booking.com", c["link"])
            st.markdown('</div>', unsafe_allow_html=True)

    # Global link anchored to the exact place
    external_link_button(
        "🔗 See full results on Booking.com",
        deeplink_booking_city(precise_ss, geo["lat"], geo["lon"], start_date, end_date, adults)
    )

    add_history(uid, {
        "city": city, "area": (area or "").strip(),
        "start": f"{start_date}", "end": f"{end_date}",
        "adults": adults, "budget": int(budget_amount), "interests": interests
    })

    pkg = {
        "itinerary_header": header,
        "itinerary": days_plan,
        "budget_per_day": int(budget_amount),
        "budget_notes": budget_notes(int(budget_amount)),
        "stay_recommendations": hotel_cards,
        "deeplink_city": deeplink_booking_city(precise_ss, geo["lat"], geo["lon"], start_date, end_date, adults)
    }
    st.download_button("⬇️ Download Plan (JSON)",
                       data=json.dumps(pkg, ensure_ascii=False, indent=2),
                       file_name=f"montravels_{city.lower().replace(' ','_')}.json",
                       mime="application/json")

    st.subheader("🧠 Your Saved History (Private to this user)")
    st.json(get_user_history(uid))

else:
    st.info("Enter a city (and optional area), pick dates & budget, select interests, then click **Build Personalized Plan**.")
