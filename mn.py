import os
import re
import math
import json
import time
import hashlib
from datetime import date, timedelta
from typing import Optional, List, Set, Dict, Tuple
import urllib.parse
import requests
import streamlit as st
from textwrap import shorten

# ================================
# Page + Light Theme
# ================================
st.set_page_config(page_title="MonTravels", page_icon="🧭", layout="wide")

def apply_pokemon_theme():
    st.markdown("""
        <style>
        .stApp { background-color: #F5F7FA; font-family: 'Trebuchet MS', sans-serif; color: #2C2C2C; }
        h1 { color: #FFCC00; text-shadow: 2px 2px 0px #3B4CCA; }
        h2, h3 { color: #3B4CCA; }
        section[data-testid="stSidebar"] { background-color: #3B4CCA; color: white; }
        section[data-testid="stSidebar"] * { color: white !important; }
        div.stButton > button {
            background-color: #FF1C1C; color: white;
            border-radius: 12px; border: 2px solid #3B4CCA; font-weight: bold; transition: 0.2s;
        }
        div.stButton > button:hover {
            background-color: #FFCC00; color: #2C2C2C; border: 2px solid #FF1C1C;
        }
        .card { background-color: #FFFFFF; border-radius: 14px; padding: 12px; margin-bottom: 10px;
                border: 1px solid #FFD84D; box-shadow: 1px 2px 5px rgba(0,0,0,0.06); }
        a { color: #3B4CCA; text-decoration: none; font-weight: bold; }
        a:hover { color: #FF1C1C; }
        </style>
    """, unsafe_allow_html=True)

apply_pokemon_theme()
st.title("🧭 MonTravels")

# ================================
# OpenAI (modern SDK)
# ================================
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

@st.cache_resource(show_spinner=False)
def get_openai_client() -> Optional["OpenAI"]:
    if OpenAI is None:
        return None
    key = (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)

# ================================
# Utilities
# ================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def combined_query(city: str, area: Optional[str]) -> str:
    return (f"{city} {area}".strip() if area else city).strip()

def _cachebuster(seed: str) -> str:
    raw = f"{seed}-{time.time_ns()}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]

def external_link_button(label: str, url: str):
    st.markdown(
        f'<a target="_blank" rel="noopener" href="{url}" style="text-decoration:none;"><button class="stButton">{label}</button></a>',
        unsafe_allow_html=True
    )

# ================================
# Booking deeplinks
# ================================
def deeplink_booking_city(city_or_area: str, checkin: date, checkout: date, adults: int = 2) -> str:
    q = urllib.parse.quote(city_or_area)
    return ("https://www.booking.com/searchresults.html"
            f"?ss={q}&ssne={q}&ssne_untouched=1"
            f"&checkin={checkin:%Y-%m-%d}&checkout={checkout:%Y-%m-%d}"
            f"&group_adults={adults}&no_rooms=1&group_children=0"
            f"&lang=en-us&src=index&sb=1&_mtu={_cachebuster(city_or_area)}")

def deeplink_booking_with_keywords(city: str, area: Optional[str], keywords: str,
                                   checkin: date, checkout: date, adults: int = 2) -> str:
    parts = [city]; 
    if area: parts.append(area)
    if keywords: parts.append(keywords)
    ss_raw = " ".join(parts).strip()
    ss = urllib.parse.quote(ss_raw)
    return ("https://www.booking.com/searchresults.html"
            f"?ss={ss}&ssne={ss}&ssne_untouched=1"
            f"&checkin={checkin:%Y-%m-%d}&checkout={checkout:%Y-%m-%d}"
            f"&group_adults={adults}&no_rooms=1&group_children=0"
            f"&lang=en-us&src=index&sb=1&_mtu={_cachebuster(ss_raw)}")

# ================================
# Geocoding + POIs (OSM)
# ================================
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_osm(query: str) -> Optional[Dict]:
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "MonTravels/1.0"},
            timeout=10
        )
        r.raise_for_status()
        js = r.json() or []
        if js:
            return {"lat": float(js[0]["lat"]), "lon": float(js[0]["lon"]), "name": js[0]["display_name"]}
    except Exception:
        return None
    return None

OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
OSM_TARGETS = {
    "landmark":   [("tourism","attraction"), ("historic","~.*"), ("building","cathedral"), ("amenity","place_of_worship")],
    "museum":     [("tourism","museum")],
    "park":       [("leisure","park"), ("leisure","garden")],
    "cafe":       [("amenity","cafe"), ("amenity","fast_food")],
    "restaurant": [("amenity","restaurant")],
    "viewpoint":  [("tourism","viewpoint")]
}

def build_overpass_query(lat: float, lon: float, radius_m: int, kv_pairs: List[Tuple[str,str]]) -> str:
    parts = []
    for k, v in kv_pairs:
        if str(v).startswith("~"):
            parts += [f'node["{k}"{v}](around:{radius_m},{lat},{lon});']
        else:
            parts += [f'node["{k}"="{v}"](around:{radius_m},{lat},{lon});']
    return f"[out:json][timeout:20];({''.join(parts)});out center 40;"

@st.cache_data(ttl=600, show_spinner=False)
def fetch_pois(lat: float, lon: float, radius_m: int = 2500, kind: str = "landmark", limit: int = 30) -> List[Dict]:
    # smaller radius + limits -> faster & more walkable
    try:
        kv = OSM_TARGETS.get(kind, [])
        if not kv:
            return []
        q = build_overpass_query(lat, lon, radius_m, kv)
        r = requests.post(OVERPASS_ENDPOINT, data={"data": q}, timeout=20)
        r.raise_for_status()
        elements = r.json().get("elements", [])
        out, seen = [], set()
        for e in elements:
            tags = e.get("tags", {})
            name = tags.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            lat_, lon_ = e.get("lat") or e.get("center",{}).get("lat"), e.get("lon") or e.get("center",{}).get("lon")
            if lat_ and lon_:
                out.append({"name": name, "lat": float(lat_), "lon": float(lon_), "tags": tags})
        # Prefer closest (walkable)
        out.sort(key=lambda p: haversine_km(lat, lon, p["lat"], p["lon"]))
        return out[:limit]
    except Exception:
        return []

def pick_unique(pois: List[Dict], n: int, used: Set[str], origin: Tuple[float,float]) -> List[Dict]:
    if not pois: return []
    olat, olon = origin
    enriched = [(haversine_km(olat,olon,p["lat"],p["lon"]),p) for p in pois if p["name"] not in used]
    enriched.sort(key=lambda x: x[0])
    chosen = [p for _,p in enriched[:n]]
    for c in chosen: used.add(c["name"])
    return chosen

# ================================
# Budget helpers
# ================================
def budget_notes(amount: int) -> str:
    if amount < 50:
        return "**Budget (~${}/day)**\n- Street food, public transport, free sights.".format(amount)
    elif amount < 150:
        return "**Budget (~${}/day)**\n- Mix of free & paid attractions, casual dining.".format(amount)
    else:
        return "**Budget (~${}/day)**\n- Premium tours, fine dining, upscale stays.".format(amount)

def budget_band(amount: int) -> str:
    return "shoestring" if amount < 50 else ("moderate" if amount < 150 else "premium")

# ================================
# Hotels (simple archetypes)
# ================================
ARCHETYPES = [
    {"key":"historic-boutique","title":"Boutique near Old Town","tags":["walkable"],"good_for":["history","museums"]},
    {"key":"central-midscale","title":"Midscale City Center","tags":["convenient"],"good_for":["shopping","food"]},
    {"key":"waterfront-view","title":"Waterfront Hotel","tags":["views"],"good_for":["nature","family"]}
]

def score_archetype(arch: Dict, interests: List[str], amount: int, area: Optional[str], bias: Set[str]) -> float:
    s = len(set(i.lower() for i in interests) & set(arch["good_for"]))
    s += len(bias & set(arch["good_for"]))
    if amount < 50 and arch["key"] == "central-midscale": s += 1
    if amount >= 150 and arch["key"] == "waterfront-view": s += 2
    return s

def synthesize_hotel_cards(city: str, area: Optional[str], start: date, end: date,
                           adults: int, interests: List[str], amount: int,
                           bias: Set[str], k:int=3)->List[Dict]:
    ranked = sorted(ARCHETYPES, key=lambda a: score_archetype(a, interests, amount, area, bias), reverse=True)
    out=[]
    for a in ranked[:k]:
        link = deeplink_booking_with_keywords(city, area, a["title"], start, end, adults)
        out.append({"title":a["title"],"why":", ".join(a["good_for"]),"tags":a["tags"] , "link":link})
    return out

# ================================
# History (fixed bug)
# ================================
def get_user_id():
    try:
        return str((st.experimental_user or {}).get("id") or "guest")
    except Exception:
        return "guest"

def get_user_history(uid): 
    return st.session_state.setdefault("history",{}).setdefault(uid,[])

def add_history(uid, rec): 
    get_user_history(uid).append(rec)

def derive_interest_bias(uid) -> Set[str]:
    freq = {}
    for trip in get_user_history(uid):
        for i in trip.get("interests", []):
            freq[i.lower()] = freq.get(i.lower(), 0) + 1
    return set([k for k,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:3]])

# ================================
# Heuristic fallback itinerary
# ================================
def assemble_itinerary(lat,lon,city,area,start,end,interests,amount):
    days = max((end-start).days, 1)
    pools={k:fetch_pois(lat,lon,2500 if k!="viewpoint" else 5000,k,20 if k!="restaurant" else 40) 
           for k in ["landmark","museum","park","cafe","restaurant","viewpoint"]}
    used=set(); origin=(lat,lon); out=[]
    # simple morning/afternoon/evening with short hops
    for _ in range(days):
        out.append({
            "Morning":  pick_unique(pools["landmark"] or pools["museum"], 1, used, origin),
            "Afternoon":pick_unique(pools["park"] or pools["viewpoint"], 1, used, origin),
            "Evening":  pick_unique(pools["restaurant"] or pools["cafe"], 1, used, origin)
        })
    return f"## {city} Itinerary ({days} days)", out

# ================================
# Grounding helpers
# ================================
def _osm_catalog(lat: float, lon: float) -> Dict[str, List[Dict]]:
    # smaller sets -> faster
    return {
        "landmarks":  fetch_pois(lat, lon, 2500, "landmark", 40),
        "museums":    fetch_pois(lat, lon, 2500, "museum", 20),
        "parks":      fetch_pois(lat, lon, 2500, "park", 20),
        "restaurants":fetch_pois(lat, lon, 2500, "restaurant", 40),
        "viewpoints": fetch_pois(lat, lon, 4000, "viewpoint", 20),
        "cafes":      fetch_pois(lat, lon, 2000, "cafe", 25),
    }

def _as_name_set(catalog: Dict[str, List[Dict]]) -> Set[str]:
    names = set()
    for arr in catalog.values():
        for x in arr:
            names.add(x["name"])
    return names

def _ground_and_prune(model_days: List[Dict], osm_names: Set[str]) -> List[Dict]:
    pruned = []
    for day in model_days:
        clean = {}
        for slot, items in day.items():
            keep = []
            for it in items:
                name = (it.get("name") or "").strip()
                if name and name in osm_names:
                    keep.append({"name": name})
            clean[slot] = keep[:1]  # keep first only -> concise
        pruned.append(clean)
    return pruned

def _extract_json(text: str) -> Dict:
    if not text: return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL|re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        s = m.group(1).split("```")[0]
        try: return json.loads(s)
        except Exception: pass
    return {}

def _coerce_to_schema(data: Dict) -> Dict:
    out = {"days": [], "notes": ""}
    if not isinstance(data, dict): return out
    if isinstance(data.get("notes"), str): out["notes"] = data["notes"]
    days = data.get("days", [])
    if not isinstance(days, list): return out
    def _norm_slot(items):
        norm=[]
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and isinstance(it.get("name"), str) and it["name"].strip():
                    o={"name": it["name"].strip()}
                    if isinstance(it.get("why"), str): o["why"]=it["why"]
                    if isinstance(it.get("est_cost_usd"), (int,float)): o["est_cost_usd"]=float(it["est_cost_usd"])
                    norm.append(o)
        return norm
    for d in days:
        if not isinstance(d, dict): continue
        out["days"].append({
            "Morning":  _norm_slot(d.get("Morning", [])),
            "Afternoon":_norm_slot(d.get("Afternoon", [])),
            "Evening":  _norm_slot(d.get("Evening", [])),
        })
    return out

# ================================
# Itinerary with OpenAI (concise + budget-aware)
# ================================
def generate_itinerary_with_openai(city: str, area: Optional[str], start: date, end: date,
                                   lat: float, lon: float, interests: List[str], amount: int) -> Tuple[str, List[Dict]]:
    client = get_openai_client()
    days = max((end - start).days, 1)

    if client is None:
        st.info("OpenAI key not found — using local heuristic itinerary.")
        return assemble_itinerary(lat, lon, city, area, start, end, interests, amount)

    catalog = _osm_catalog(lat, lon)
    osm_names = sorted(list(_as_name_set(catalog)))[:500]  # smaller grounding set -> faster
    band = budget_band(amount)

    system_msg = (
        "You are a practical travel planner. Build a realistic, concise, walkable itinerary "
        "using ONLY places from 'allowed_place_names'. Respect 'budget_band' (shoestring/moderate/premium). "
        "One activity per slot (Morning/Afternoon/Evening). Prefer short travel hops. No duplicates. "
        "Output ONLY valid JSON: {\"days\": [{\"Morning\": [{\"name\": str}], \"Afternoon\": [...], \"Evening\": [...]}], \"notes\": str}."
    )
    user_payload = {
        "city": city, "area": area, "days": days,
        "budget_band": band,
        "interests": interests,
        "allowed_place_names": osm_names
    }

    MODEL = os.getenv("OPENAI_TRAVEL_MODEL", "gpt-4o-mini")

    try:
        comp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
            temperature=0.4,
            max_tokens=1200,
        )
        raw = comp.choices[0].message.content or ""
        data = _coerce_to_schema(_extract_json(raw))
        grounded = _ground_and_prune(data.get("days", []), set(osm_names))
        if not grounded or all(not any(v for v in d.values()) for d in grounded):
            return assemble_itinerary(lat, lon, city, area, start, end, interests, amount)
        header = f"## {city} Itinerary ({days} days)"
        return header, grounded
    except Exception as e:
        st.warning(f"OpenAI plan generation failed ({e}); using local fallback.")
        return assemble_itinerary(lat, lon, city, area, start, end, interests, amount)

# ================================
# Local Guide (unique tips, no repetition)
# ================================
def _narrative_template(city: str, start: date, end: date, interests: List[str], budget: int, days_plan: List[Dict]) -> str:
    lines = [f"### Your {city} trip ({start}–{end})",
             f"_Budget: ~${budget}/day · Interests: {', '.join(interests) or '—'}_",
             ""]
    for i, day in enumerate(days_plan, 1):
        lines.append(f"**Day {i}**")
        for slot in ["Morning","Afternoon","Evening"]:
            names = [p["name"] for p in day.get(slot, [])]
            if names:
                lines.append(f"- {slot}: {names[0]} — short walk, easy pace.")
        lines.append("")
    lines.append("_Tip: Check opening hours and carry water._")
    return "\n".join(lines)

def generate_local_guide_tips(city: str, start: date, end: date, interests: List[str],
                              budget: int, days_plan: List[Dict]) -> str:
    client = get_openai_client()
    # Build a set of already-mentioned places to AVOID repetition in local tips
    mentioned = set()
    for d in days_plan:
        for s in ["Morning","Afternoon","Evening"]:
            for it in d.get(s, []):
                mentioned.add(it["name"])

    prompt = {
        "instruction": (
            "Write a short 'Your Local Guide Says' section with 6–10 bullet tips. "
            "Be realistic, budget-aware, and specific (timings, neighborhoods, transit/ride-hailing norms, scams to avoid, "
            "tipping, ticket hacks, where to find local snacks/markets). "
            "Include 2–3 lesser-known suggestions (e.g., a typical street or market style) WITHOUT naming any new specific venue. "
            "ABSOLUTE RULES: Do NOT repeat the day-by-day places, don't list attraction names, and keep it crisp."
        ),
        "context": {
            "city": city,
            "dates": {"start": str(start), "end": str(end)},
            "budget_per_day_usd": budget,
            "interests": interests,
            "already_mentioned_places": sorted(list(mentioned))
        }
    }

    if client is None:
        # Minimal, generic but useful tips
        return (
            "- Start mornings early to beat crowds and heat.\n"
            "- Use ride-hailing or metro for longer hops; walk short segments.\n"
            "- Carry small change; card acceptance varies at kiosks.\n"
            "- For budget eats, try busy local food streets at lunch.\n"
            "- Book popular tickets online the night before.\n"
            "- Evenings: choose compact neighborhoods to minimize transit.\n"
            "- Respect local dress norms at religious sites.\n"
            "- Avoid unmetered taxis; confirm price before rides.\n"
            "- Keep a reusable water bottle; many parks have fountains.\n"
            "- Sundays/holidays can shift opening hours — double check."
        )

    MODEL = os.getenv("OPENAI_TIPS_MODEL", "gpt-4o-mini")
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":"You are a savvy local who gives concise, actionable travel tips."},
                {"role":"user","content": json.dumps(prompt, ensure_ascii=False)}
            ],
            temperature=0.6,
            max_tokens=500,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return _narrative_template(city, start, end, interests, budget, days_plan)

# ================================
# Reviews (optional toggle, faster defaults)
# ================================
GOOGLE_PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
GOOGLE_PLACES_DETAILS_URL_TMPL = "https://places.googleapis.com/v1/places/{place_id}"

def _google_headers(field_mask: str) -> Dict[str, str]:
    key = (st.secrets.get("GOOGLE_MAPS_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("GOOGLE_MAPS_API_KEY")
    if not key: return {}
    return {"X-Goog-Api-Key": key, "X-Goog-FieldMask": field_mask, "Content-Type": "application/json"}

@st.cache_data(ttl=24*3600, show_spinner=False)
def google_place_id_by_text(name: str, city: str, lat: float, lon: float) -> Optional[str]:
    headers = _google_headers("places.id,places.displayName")
    if not headers: return None
    try:
        payload = {"textQuery": f"{name}, {city}",
                   "locationBias": {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": 5000.0}}}
        r = requests.post(GOOGLE_PLACES_SEARCH_URL, headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        places = (r.json() or {}).get("places", [])
        return places[0].get("id") if places else None
    except Exception:
        return None

@st.cache_data(ttl=24*3600, show_spinner=False)
def google_place_details_reviews(place_id: str) -> Optional[Dict]:
    headers = _google_headers("id,displayName,rating,userRatingCount,reviews")
    if not headers: return None
    try:
        url = GOOGLE_PLACES_DETAILS_URL_TMPL.format(place_id=place_id)
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json() or {}
        out = {"source":"google","name": data.get("displayName",{}).get("text"),
               "rating": data.get("rating"), "reviews_count": data.get("userRatingCount"), "reviews":[]}
        for rv in (data.get("reviews") or [])[:3]:
            out["reviews"].append({
                "author": (rv.get("authorAttribution") or {}).get("displayName"),
                "rating": rv.get("rating"),
                "text": rv.get("text", ""),
            })
        return out
    except Exception:
        return None

def render_reviews_block(name: str, city: str, lat: float, lon: float):
    pid = google_place_id_by_text(name, city, lat, lon)
    if not pid: 
        st.caption("No online reviews available."); 
        return
    data = google_place_details_reviews(pid)
    if not data:
        st.caption("No online reviews available.")
        return
    header_bits = []
    if data.get("rating") is not None: header_bits.append(f"⭐ {data['rating']:.1f}")
    if data.get("reviews_count") is not None: header_bits.append(f"({int(data['reviews_count'])} reviews)")
    hdr = " ".join(header_bits) or "Reviews"
    with st.container():
        st.caption("Google reviews")
        if hdr: st.write(hdr)
        for rv in (data.get("reviews") or [])[:2]:
            txt = shorten((rv.get("text","") or "").strip().replace("\n"," "), width=220, placeholder="…")
            who = rv.get("author") or "Visitor"
            stars = f"{rv.get('rating')}★" if rv.get("rating") is not None else ""
            st.markdown(f"> _{txt}_ — **{who}** {stars}")

# ================================
# UI — Sidebar (generate ONLY on click)
# ================================
with st.sidebar:
    city = st.text_input("Destination*").strip()
    area = st.text_input("Area (optional)").strip()
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("Start", date.today() + timedelta(days=7))
    with c2: end_date   = st.date_input("End",   date.today() + timedelta(days=10))
    adults  = st.number_input("Adults", 1, 10, 2)
    budget  = st.number_input("Budget ($/day)", 10, 1000, 100)
    interests = st.multiselect("Interests", ["food","history","museums","nature","nightlife"], default=["food","history"])
    st.markdown("---")
    tone = st.selectbox("Tone", ["Friendly", "Crisp", "Excited"], index=0)
    fetch_reviews = st.checkbox("Fetch online reviews (slower)", value=False)
    go = st.button("✨ Build Plan")

uid = get_user_id()
st.caption(f"User: `{uid}`")

# ================================
# ACTION
# ================================
if go:
    if not city:
        st.error("Enter a city."); st.stop()

    with st.spinner("Finding places and building your plan…"):
        geo = geocode_osm(combined_query(city,area) or city)
        if not geo:
            st.error("Could not find that destination."); st.stop()

        # Model-assisted plan (grounded on OSM & budget-aware)
        header, days_plan = generate_itinerary_with_openai(
            city=city, area=area, start=start_date, end=end_date,
            lat=geo["lat"], lon=geo["lon"],
            interests=interests, amount=budget
        )

    # ===== Quick View (concise) =====
    st.subheader("🗓️ Itinerary (Quick View)")
    st.markdown(header)
    for i, slots in enumerate(days_plan, 1):
        st.markdown(f"### Day {i}")
        for part in ["Morning", "Afternoon", "Evening"]:
            items = slots.get(part, [])
            if not items: 
                continue
            names = ", ".join(p["name"] for p in items)
            st.markdown(f"- **{part}**: {names}")
            if fetch_reviews:
                with st.expander("See quick reviews"):
                    render_reviews_block(items[0]["name"], city or "", geo["lat"], geo["lon"])

    st.markdown(budget_notes(budget))

    # ===== Local Guide (unique tips, no repeats) =====
    st.subheader("🗣️ Your Local Guide Says…")
    with st.spinner("Asking your local guide…"):
        tips = generate_local_guide_tips(
            city=city, start=start_date, end=end_date,
            interests=interests, budget=budget, days_plan=days_plan
        )
    st.markdown(tips)

    # ---- Stay options ----
    st.subheader("🏨 Places to Stay")
    hotel_cards = synthesize_hotel_cards(city, area, start_date, end_date, adults, interests, budget, derive_interest_bias(uid))
    for c in hotel_cards:
        with st.container():
            st.markdown(f"**{c['title']}**")
            st.caption(c["why"])
            st.write("Tags:", ", ".join(c["tags"]))
            external_link_button("Open on Booking.com", c["link"])
    external_link_button("🔗 See all on Booking.com", deeplink_booking_city(city or area or "", start_date, end_date, adults))

    # ---- History ----
    add_history(uid, {"city":city,"area":area,"start":str(start_date),"end":str(end_date),"budget":budget,"interests":interests})
    with st.expander("🧠 History"):
        st.json(get_user_history(uid))
else:
    st.info("Enter details and click **Build Plan**.")
