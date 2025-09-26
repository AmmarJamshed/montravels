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
# Page + Theme
# ================================
st.set_page_config(page_title="MonTravels", page_icon="🧭", layout="wide")

def apply_pokemon_theme():
    st.markdown("""
        <style>
        .stApp { background-color: #F5F7FA; font-family: 'Trebuchet MS', sans-serif; color: #2C2C2C; }
        h1 { color: #FFCC00; text-shadow: 2px 2px 0px #3B4CCA; }
        h2, h3 { color: #3B4CCA; }

        /* Sidebar */
        section[data-testid="stSidebar"] { background-color: #3B4CCA; color: white; }
        section[data-testid="stSidebar"] * { color: white !important; }

        /* Readable inputs in sidebar */
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] select,
        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] div,
        section[data-testid="stSidebar"] .stNumberInput input,
        section[data-testid="stSidebar"] .stDateInput input {
            color: #0f172a !important;
            background-color: #eef2ff !important;
            border-radius: 10px !important;
        }
        section[data-testid="stSidebar"] .stMultiSelect div[role="listbox"] * { color: #0f172a !important; }

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

# ==========================================
# OPTIONAL Hugging Face Inference (no OpenAI)
# ==========================================
try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

@st.cache_resource(show_spinner=False)
def get_hf_client() -> Optional["InferenceClient"]:
    """
    Returns a Hugging Face serverless Inference client if HF_TOKEN is set,
    otherwise None (we'll use the heuristic-only path).
    """
    if InferenceClient is None:
        return None
    token = (st.secrets.get("HF_TOKEN") if hasattr(st, "secrets") else None) or os.getenv("HF_TOKEN")
    if not token:
        return None
    model = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    try:
        return InferenceClient(model=model, token=token, timeout=30)
    except Exception:
        return None

def hf_generate_json(prompt: str, max_new_tokens: int = 800, temperature: float = 0.4) -> str:
    """
    Single-call JSON generator. We keep formatting strict: ask model to output ONLY JSON.
    """
    client = get_hf_client()
    if client is None:
        return ""
    try:
        # Serverless text-generation; returns a string
        return client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=1.05,
            stop_sequences=["\n\n\n", "\n```", "\n</s>", "</s>"]
        )
    except Exception:
        return ""

# ================================
# Utils
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
# Multi-provider Geocoder (reliable)
# ================================
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city(query: str) -> Optional[Dict]:
    q = (query or "").strip()
    if not q:
        return None

    # 1) Nominatim
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": 1},
            headers={"User-Agent": "MonTravels/1.0"},
            timeout=8,
        )
        if r.ok:
            js = r.json() or []
            if js:
                return {"lat": float(js[0]["lat"]), "lon": float(js[0]["lon"]), "name": js[0]["display_name"]}
    except Exception:
        pass

    # 2) Maps.co mirror
    try:
        r = requests.get(
            "https://geocode.maps.co/search",
            params={"q": q, "limit": 1},
            headers={"User-Agent": "MonTravels/1.0"},
            timeout=8,
        )
        if r.ok:
            js = r.json() or []
            if js:
                return {"lat": float(js[0]["lat"]), "lon": float(js[0]["lon"]), "name": js[0].get("display_name") or q}
    except Exception:
        pass

    # 3) Photon/Komoot
    try:
        r = requests.get(
            "https://photon.komoot.io/api/",
            params={"q": q, "limit": 1},
            headers={"User-Agent": "MonTravels/1.0"},
            timeout=8,
        )
        if r.ok:
            js = r.json() or {}
            feats = (js.get("features") or [])
            if feats:
                coords = feats[0]["geometry"]["coordinates"]  # [lon, lat]
                props = feats[0].get("properties", {})
                name = props.get("name") or props.get("city") or props.get("country") or q
                return {"lat": float(coords[1]), "lon": float(coords[0]), "name": name}
    except Exception:
        pass

    return None

# ================================
# OSM POIs via Overpass
# ================================
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
# History
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
            i = (i or "").lower()
            freq[i] = freq.get(i, 0) + 1
    return set([k for k,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:3]])

# ================================
# Heuristic fallback itinerary (with rough costs)
# ================================
def _rough_cost(slot_name: str, band: str) -> float:
    table = {
        "shoestring": {"Morning": 0, "Afternoon": 5, "Evening": 8},
        "moderate":   {"Morning": 5, "Afternoon": 15, "Evening": 20},
        "premium":    {"Morning": 20,"Afternoon": 35, "Evening": 40},
    }
    return float(table.get(band, table["moderate"]).get(slot_name, 10))

def assemble_itinerary(lat,lon,city,area,start,end,interests,amount):
    days = max((end-start).days, 1)
    band = budget_band(amount)
    pools={k:fetch_pois(lat,lon,2500 if k!="viewpoint" else 5000,k,20 if k!="restaurant" else 40) 
           for k in ["landmark","museum","park","cafe","restaurant","viewpoint"]}
    used=set(); origin=(lat,lon); out=[]
    for _ in range(days):
        morning = pick_unique(pools["landmark"] or pools["museum"], 1, used, origin)
        afternoon = pick_unique(pools["park"] or pools["viewpoint"], 1, used, origin)
        evening = pick_unique(pools["restaurant"] or pools["cafe"], 1, used, origin)
        for slot, arr in (("Morning", morning), ("Afternoon", afternoon), ("Evening", evening)):
            if arr:
                arr[0]["est_cost_usd"] = _rough_cost(slot, band)
        out.append({"Morning": morning, "Afternoon": afternoon, "Evening": evening})
    header = f"## {city} Itinerary ({days} days)"
    return header, out

# ================================
# JSON helpers
# ================================
def _extract_json(text: str) -> Dict:
    if not text: return {}
    # First, try raw JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Look for fenced JSON
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL|re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    # Greedy last resort
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
        rec = {
            "Morning":  _norm_slot(d.get("Morning", []))[:1],
            "Afternoon":_norm_slot(d.get("Afternoon", []))[:1],
            "Evening":  _norm_slot(d.get("Evening", []))[:1],
            "daily_notes": d.get("daily_notes", "") if isinstance(d.get("daily_notes",""), str) else "",
            "daily_estimated_cost_usd": float(d.get("daily_estimated_cost_usd", 0)) if isinstance(d.get("daily_estimated_cost_usd", None),(int,float)) else None
        }
        out["days"].append(rec)
    return out

# ================================
# HF-enhanced itinerary (costed, concise, grounded)
# ================================
def _osm_catalog(lat: float, lon: float) -> Dict[str, List[Dict]]:
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
        for slot in ["Morning","Afternoon","Evening"]:
            keep = []
            for it in day.get(slot, []):
                name = (it.get("name") or "").strip()
                if name and name in osm_names:
                    keep.append({"name": name, "est_cost_usd": it.get("est_cost_usd")})
            clean[slot] = keep[:1]
        clean["daily_notes"] = day.get("daily_notes","")
        clean["daily_estimated_cost_usd"] = day.get("daily_estimated_cost_usd")
        pruned.append(clean)
    return pruned

def generate_itinerary_hf(city: str, area: Optional[str], start: date, end: date,
                          lat: float, lon: float, interests: List[str], amount: int) -> Tuple[str, List[Dict]]:
    client = get_hf_client()
    days = max((end - start).days, 1)
    catalog = _osm_catalog(lat, lon)
    osm_names = sorted(list(_as_name_set(catalog)))[:500]
    band = budget_band(amount)

    if client is None:
        st.info("HF token not set — using local heuristic itinerary with rough costs.")
        return assemble_itinerary(lat, lon, city, area, start, end, interests, amount)

    # Instruction-style prompt for JSON output
    prompt = (
        "You are a practical travel planner.\n"
        "TASK: Produce a realistic, concise, walkable itinerary with EXACTLY one activity per slot (Morning/Afternoon/Evening) "
        "for the given number of days. Use ONLY places from the 'ALLOWED_PLACE_NAMES' list. Prefer short travel hops and avoid duplicates.\n"
        "Budget band: {band}. Budget per day: {budget} USD. Interests: {interests}.\n"
        "For each slot item, include a short 'why' and an 'est_cost_usd' (0 allowed for free sights). "
        "Provide a 'daily_estimated_cost_usd' that sums the three slots, and a brief 'daily_notes'.\n"
        "Return ONLY valid JSON with this exact schema (no prose):\n"
        "{{\"days\":[{{\"Morning\":[{{\"name\":str,\"why\":str,\"est_cost_usd\":number}}],"
        "\"Afternoon\":[{{\"name\":str,\"why\":str,\"est_cost_usd\":number}}],"
        "\"Evening\":[{{\"name\":str,\"why\":str,\"est_cost_usd\":number}}],"
        "\"daily_notes\":str,\"daily_estimated_cost_usd\":number}}],\"notes\":str}}\n\n"
        "CITY: {city}\nAREA: {area}\nDAYS: {days}\nALLOWED_PLACE_NAMES: {names}\n"
    ).format(
        band=band,
        budget=amount,
        interests=", ".join(interests) if interests else "—",
        city=city, area=area or "—", days=days,
        names=osm_names
    )

    raw = hf_generate_json(prompt, max_new_tokens=900, temperature=0.4)
    data = _coerce_to_schema(_extract_json(raw))
    grounded = _ground_and_prune(data.get("days", []), set(osm_names))
    if not grounded or all(not any(v for v in d.values()) for d in grounded):
        return assemble_itinerary(lat, lon, city, area, start, end, interests, amount)
    header = f"## {city} Itinerary ({days} days)"
    return header, grounded

# ================================
# Local Guide (unique tips, no repetition) via HF or fallback
# ================================
def generate_local_guide_tips(city: str, start: date, end: date, interests: List[str],
                              budget: int, days_plan: List[Dict]) -> str:
    client = get_hf_client()
    mentioned = set()
    for d in days_plan:
        for s in ["Morning","Afternoon","Evening"]:
            for it in d.get(s, []):
                if it.get("name"):
                    mentioned.add(it["name"])

    if client is None:
        return (
            "- Start early to beat crowds and heat.\n"
            "- Use ride-hailing or metro for longer hops; walk short segments.\n"
            "- Carry small change; card acceptance varies at kiosks.\n"
            "- For budget eats, pick busy local food streets at lunch.\n"
            "- Book popular tickets online the night before.\n"
            "- Evenings: stick to compact neighborhoods to minimize transit.\n"
            "- Respect dress norms at religious sites.\n"
            "- Confirm taxi prices before rides; avoid unmetered cabs.\n"
            "- Keep a reusable bottle; many parks have fountains.\n"
            "- Sundays/holidays can change hours — double check."
        )

    prompt = (
        "Write a short section titled 'Your Local Guide Says' with 6–10 bullet tips for {city}.\n"
        "Rules:\n"
        "- Be realistic, budget-aware, and specific (timings, neighborhoods, transit/ride-hailing norms, scams to avoid, tipping, ticket hacks, market/street-food etiquette).\n"
        "- Include 2–3 lesser-known suggestions WITHOUT naming any specific venue.\n"
        "- ABSOLUTE: Do NOT repeat these place names: {avoid}.\n"
        "- Keep it crisp; bullets only.\n"
        "Return plain text bullets (no JSON).\n"
    ).format(city=city, avoid=", ".join(sorted(list(mentioned))) or "—")

    text = hf_generate_json(prompt, max_new_tokens=300, temperature=0.6)
    return (text or "").strip() or (
        "- Start early to beat crowds and heat.\n"
        "- Use ride-hailing or metro for longer hops; walk short segments."
    )

# ================================
# Reviews (optional toggle)
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
# Sidebar (generate ONLY on click)
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
    if len(city) < 3:
        st.error("Destination name looks too short. Try 'Karachi, Pakistan' etc."); st.stop()

    with st.spinner("Finding places and building your plan…"):
        search_q = combined_query(city, area) or city
        geo = geocode_city(search_q)
        if not geo:
            st.error(f"Could not find that destination. (Tried: “{search_q}”)")
            st.caption("Tip: Try a broader name like 'City, Country'.")
            st.stop()

        # HF-enhanced plan (or heuristic fallback)
        header, days_plan = generate_itinerary_hf(
            city=city, area=area, start=start_date, end=end_date,
            lat=geo["lat"], lon=geo["lon"],
            interests=interests, amount=budget
        )

    # ===== Quick View (with per-slot costs) =====
    st.subheader("🗓️ Itinerary (Quick View)")
    st.markdown(header)

    trip_cost = 0.0
    planned_days = 0

    for i, slots in enumerate(days_plan, 1):
        st.markdown(f"### Day {i}")
        day_cost = 0.0
        for part in ["Morning", "Afternoon", "Evening"]:
            items = slots.get(part, [])
            if not items: 
                continue
            first = items[0]
            nm = first.get("name","")
            cst = first.get("est_cost_usd")
            cost_txt = f" (~${cst:.0f})" if isinstance(cst,(int,float)) else ""
            st.markdown(f"- **{part}**: {nm}{cost_txt}")
            if isinstance(cst,(int,float)): 
                day_cost += float(cst)
            if fetch_reviews:
                with st.expander("See quick reviews"):
                    render_reviews_block(nm, city or "", geo["lat"], geo["lon"])

        dn = slots.get("daily_notes","")
        dsum = slots.get("daily_estimated_cost_usd")
        if isinstance(dsum,(int,float)):
            day_cost = float(dsum)
        if dn:
            st.caption(dn)
        st.write(f"**Estimated day spend:** ${day_cost:.0f}")
        trip_cost += day_cost
        planned_days += 1

    target = float(budget) * max(planned_days, 1)
    delta = trip_cost - target
    st.markdown("### 💰 Budget Use")
    st.write(f"- **Planned total:** ${trip_cost:.0f}")
    st.write(f"- **Your budget ({planned_days} × ${budget:.0f}/day):** ${target:.0f}")
    if delta > 5:
        st.warning(f"Over budget by ~${delta:.0f}. Consider swapping one paid slot for a free landmark/park.")
    else:
        st.success("Within your budget 👍")

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

    # ---- Travel partners (agent booking) ----
    st.subheader("✈️ Travel Partners (Booking Help)")
    agents = [
        {"name":"GlobeTrek Tours","desc":"Cultural & family packages","email":"info@globetrek.com","link":"https://globetrek.example.com"},
        {"name":"SkyHigh Travels","desc":"Custom itineraries & visa support","email":"bookings@skyhigh.example.com","link":"https://skyhigh.example.com"}
    ]
    summary = f"Destination: {city}\nDates: {start_date}→{end_date}\nBudget: ${budget}/day\nAdults: {adults}\nInterests: {', '.join(interests)}"
    for a in agents:
        with st.container():
            st.markdown(f"**{a['name']}**")
            st.caption(a["desc"])
            external_link_button("🌍 Visit Website", a["link"])
            subject = urllib.parse.quote(f"MonTravels Plan — {city} {start_date}→{end_date}")
            body    = urllib.parse.quote(f"Hello {a['name']},\n\nHere is my MonTravels plan:\n{summary}\n\nPlease help me book this trip.")
            external_link_button("📧 Send My Plan", f"mailto:{a['email']}?subject={subject}&body={body}")

    # ---- History ----
    add_history(uid, {"city":city,"area":area,"start":str(start_date),"end":str(end_date),"budget":budget,"interests":interests})
    with st.expander("🧠 History"):
        st.json(get_user_history(uid))
else:
    st.info("Enter details and click **Build Plan**.")
