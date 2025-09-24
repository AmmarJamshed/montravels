import os
import math
import json
import time
import hashlib
from datetime import date, timedelta
from typing import Optional, List, Set, Dict, Tuple
import urllib.parse
import requests
import streamlit as st

# === OpenAI client (modern SDK) ===
try:
    from openai import OpenAI
except ImportError:
    # If the SDK isn't installed yet, we'll warn at runtime.
    OpenAI = None

# =========================================================
# THEME (Pokémon-inspired Travel Guide)
# =========================================================
def apply_pokemon_theme():
    st.markdown("""
        <style>
        .stApp { background-color: #F5F7FA; font-family: 'Trebuchet MS', sans-serif; color: #2C2C2C; }
        h1 { color: #FFCC00; text-shadow: 2px 2px 0px #3B4CCA; }
        h2, h3 { color: #3B4CCA; }
        section[data-testid="stSidebar"] { background-color: #3B4CCA; color: white; }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3, 
        section[data-testid="stSidebar"] label, 
        section[data-testid="stSidebar"] span { color: white !important; }
        div.stButton > button {
            background-color: #FF1C1C; color: white;
            border-radius: 12px; border: 2px solid #3B4CCA;
            font-weight: bold; transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #FFCC00; color: #2C2C2C; border: 2px solid #FF1C1C;
        }
        .stContainer {
            background-color: #FFFFFF; border-radius: 16px; padding: 12px; margin-bottom: 12px;
            border: 2px solid #FFCC00; box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        }
        a { color: #3B4CCA; text-decoration: none; font-weight: bold; }
        a:hover { color: #FF1C1C; }
        .stCaption { color: #4CAF50 !important; }
        </style>
    """, unsafe_allow_html=True)

# =========================================================
# Utilities
# =========================================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def combined_query(city: str, area: Optional[str]) -> str:
    return (f"{city} {area}".strip() if area else city).strip()

def _cachebuster(seed: str) -> str:
    raw = f"{seed}-{time.time_ns()}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]

def deeplink_booking_city(city_or_area: str, checkin: date, checkout: date, adults: int = 2) -> str:
    q = urllib.parse.quote(city_or_area)
    u = (
        "https://www.booking.com/searchresults.html"
        f"?ss={q}"
        f"&ssne={q}&ssne_untouched=1"
        f"&checkin={checkin:%Y-%m-%d}"
        f"&checkout={checkout:%Y-%m-%d}"
        f"&group_adults={adults}&no_rooms=1&group_children=0"
        f"&lang=en-us&src=index&sb=1&nflt="
        f"&_mtu={_cachebuster(city_or_area)}"
    )
    return u

def deeplink_booking_with_keywords(city: str, area: Optional[str], keywords: str,
                                   checkin: date, checkout: date, adults: int = 2) -> str:
    parts = [city]
    if area: parts.append(area)
    if keywords: parts.append(keywords)
    ss_raw = " ".join(parts).strip()
    ss = urllib.parse.quote(ss_raw)
    u = (
        "https://www.booking.com/searchresults.html"
        f"?ss={ss}"
        f"&ssne={ss}&ssne_untouched=1"
        f"&checkin={checkin:%Y-%m-%d}"
        f"&checkout={checkout:%Y-%m-%d}"
        f"&group_adults={adults}&no_rooms=1&group_children=0"
        f"&lang=en-us&src=index&sb=1&nflt="
        f"&_mtu={_cachebuster(ss_raw)}"
    )
    return u

def external_link_button(label: str, url: str):
    st.markdown(
        f'<a target="_blank" rel="noopener" href="{url}" '
        f'style="text-decoration:none;"><button class="stButton">{label}</button></a>',
        unsafe_allow_html=True
    )

# =========================================================
# Geocoding & POIs (OpenStreetMap)
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_osm(query: str) -> Optional[Dict]:
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": query, "format": "json", "limit": 1}
        headers = {"User-Agent": "MonTravels/1.0"}
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
    "landmark": [("tourism","attraction"),("historic","~.*"),("building","cathedral"),("amenity","place_of_worship")],
    "museum": [("tourism","museum")],
    "park": [("leisure","park"),("leisure","garden")],
    "cafe": [("amenity","cafe"),("amenity","fast_food")],
    "restaurant": [("amenity","restaurant")],
    "viewpoint": [("tourism","viewpoint")]
}

def build_overpass_query(lat: float, lon: float, radius_m: int, kv_pairs: List[Tuple[str,str]]) -> str:
    parts = []
    for k, v in kv_pairs:
        if str(v).startswith("~"):
            parts += [f'node["{k}"{v}](around:{radius_m},{lat},{lon});']
        else:
            parts += [f'node["{k}"="{v}"](around:{radius_m},{lat},{lon});']
    return f"[out:json][timeout:30];({''.join(parts)});out center 60;"

@st.cache_data(ttl=900, show_spinner=False)
def fetch_pois(lat: float, lon: float, radius_m: int = 3000, kind: str = "landmark", limit: int = 50) -> List[Dict]:
    try:
        kv = OSM_TARGETS.get(kind, [])
        if not kv: return []
        q = build_overpass_query(lat, lon, radius_m, kv)
        r = requests.post(OVERPASS_ENDPOINT, data={"data": q}, timeout=40)
        r.raise_for_status()
        elements = r.json().get("elements", [])
        out, seen = [], set()
        for e in elements:
            tags = e.get("tags", {})
            name = tags.get("name")
            if not name or name in seen: continue
            seen.add(name)
            lat_, lon_ = e.get("lat") or e.get("center",{}).get("lat"), e.get("lon") or e.get("center",{}).get("lon")
            if lat_ and lon_:
                out.append({"name": name, "lat": float(lat_), "lon": float(lon_), "tags": tags})
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

# =========================================================
# Budget helpers
# =========================================================
def budget_notes(amount: int) -> str:
    if amount < 50: return f"**Budget (~${amount}/day)**\n- Street food, public transport, free sights."
    elif amount < 150: return f"**Budget (~${amount}/day)**\n- Mix of free & paid attractions, casual dining."
    else: return f"**Budget (~${amount}/day)**\n- Premium tours, fine dining, upscale stays."

def budget_profile(amount: int) -> Dict:
    return {"museums_per_day": (0 if amount<50 else 1 if amount<150 else 2)}

# =========================================================
# Hotels (offline archetypes)
# =========================================================
ARCHETYPES=[{"key":"historic-boutique","title":"Boutique near Old Town","tags":["walkable"],"good_for":["history","museums"]},
{"key":"central-midscale","title":"Midscale City Center","tags":["convenient"],"good_for":["shopping","food"]},
{"key":"waterfront-view","title":"Waterfront Hotel","tags":["views"],"good_for":["nature","family"]}]

def score_archetype(arch: Dict, interests: List[str], amount: int, area: Optional[str], bias: Set[str]) -> float:
    score=len(set(i.lower() for i in interests)&set(arch["good_for"]))
    score+=len(bias&set(arch["good_for"]))
    if amount<50 and arch["key"]=="central-midscale": score+=1
    if amount>=150 and arch["key"]=="waterfront-view": score+=2
    return score

def synthesize_hotel_cards(city: str, area: Optional[str], start: date, end: date,
                           adults: int, interests: List[str], amount: int,
                           bias: Set[str], k:int=5)->List[Dict]:
    ranked=sorted(ARCHETYPES,key=lambda a:score_archetype(a,interests,amount,area,bias),reverse=True)
    out=[]
    for a in ranked[:k]:
        link=deeplink_booking_with_keywords(city,area,a["title"],start,end,adults)
        out.append({"title":a["title"],"why":", ".join(a["good_for"]),"tags":a["tags"],"link":link})
    return out

# =========================================================
# Local heuristic itinerary (fallback)
# =========================================================
def assemble_itinerary(lat,lon,city,area,start,end,interests,amount):
    days=max((end-start).days,1)
    pools={k:fetch_pois(lat,lon,3000,k,50) for k in ["landmark","museum","park","cafe","restaurant","viewpoint"]}
    used=set(); origin=(lat,lon); out=[]
    for _ in range(days):
        out.append({
            "Morning":pick_unique(pools["landmark"],1,used,origin),
            "Afternoon":pick_unique(pools["park"],1,used,origin),
            "Evening":pick_unique(pools["restaurant"],1,used,origin)
        })
    return f"## {city} Itinerary ({days} days)",out

def render_itinerary(header,days_plan):
    lines=[header]
    for i,slots in enumerate(days_plan,1):
        lines.append(f"### Day {i}")
        for part,items in slots.items():
            if items: lines.append(f"- **{part}**: {', '.join(p['name'] for p in items)}")
    return "\n".join(lines)

# =========================================================
# User history
# =========================================================
def get_user_id():
    try: return str((st.experimental_user or {}).get("id") or "guest")
    except: return "guest"

def get_user_history(uid): return st.session_state.setdefault("history",{}).setdefault(uid,[])
def add_history(uid,rec): get_user_history(uid).append(rec)
def derive_interest_bias(uid):
    freq={}; 
    for trip in get_user_history(uid):
        for i in trip.get("interests",[]): freq[i]=freq.get(i,0)+1
    return set(sorted(freq,key=freq.get,reverse=True)[:3])

# =========================================================
# OpenAI helpers (Chat Completions + JSON Schema)
# =========================================================
def get_openai_client() -> Optional[OpenAI]:
    if OpenAI is None:
        st.warning("OpenAI SDK not installed. Run:  pip install openai")
        return None
    key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    key = key or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    os.environ["OPENAI_API_KEY"] = key  # SDK reads from env
    try:
        return OpenAI()
    except Exception as e:
        st.error(f"OpenAI client error: {e}")
        return None

def _osm_catalog(lat: float, lon: float) -> Dict[str, List[Dict]]:
    # Gather a pool near the center to ground the model
    pools = {
        "landmarks": fetch_pois(lat, lon, 4000, "landmark", 80),
        "museums":   fetch_pois(lat, lon, 4000, "museum", 40),
        "parks":     fetch_pois(lat, lon, 4000, "park", 40),
        "restaurants": fetch_pois(lat, lon, 4000, "restaurant", 100),
        "viewpoints":  fetch_pois(lat, lon, 6000, "viewpoint", 40),
        "cafes":       fetch_pois(lat, lon, 3000, "cafe", 60),
    }
    return pools

def _as_name_set(catalog: Dict[str, List[Dict]]) -> Set[str]:
    names = set()
    for arr in catalog.values():
        for x in arr:
            names.add(x["name"])
    return names

def _ground_and_prune(model_days: List[Dict], osm_names: Set[str]) -> List[Dict]:
    # Keep only items that appear in OSM name set (simple grounding)
    pruned = []
    for day in model_days:
        clean = {}
        for slot, items in day.items():
            keep = []
            for it in items:
                name = it.get("name","").strip()
                if name and name in osm_names:
                    keep.append(it)
            clean[slot] = keep
        pruned.append(clean)
    return pruned

def generate_itinerary_with_openai(city: str, area: Optional[str], start: date, end: date,
                                   lat: float, lon: float, interests: List[str], amount: int) -> Tuple[str, List[Dict]]:
    client = get_openai_client()
    days = max((end - start).days, 1)

    # Fallback to local assembly if no key available
    if client is None:
        st.info("OpenAI key not found — using local heuristic itinerary.")
        return assemble_itinerary(lat, lon, city, area, start, end, interests, amount)

    catalog = _osm_catalog(lat, lon)
    osm_names = sorted(list(_as_name_set(catalog)))[:800]  # cap tokens

    # Structured output schema the model must follow
    schema = {
        "type": "object",
        "properties": {
            "days": {
                "type": "array",
                "items": {
                    "type":"object",
                    "properties":{
                        "Morning":{"type":"array","items":{"type":"object","properties":{
                            "name":{"type":"string"},"why":{"type":"string"},"est_cost_usd":{"type":"number"}}, "required":["name"]}},
                        "Afternoon":{"type":"array","items":{"type":"object","properties":{
                            "name":{"type":"string"},"why":{"type":"string"},"est_cost_usd":{"type":"number"}}, "required":["name"]}},
                        "Evening":{"type":"array","items":{"type":"object","properties":{
                            "name":{"type":"string"},"why":{"type":"string"},"est_cost_usd":{"type":"number"}}, "required":["name"]}}
                    },
                    "required":["Morning","Afternoon","Evening"],
                    "additionalProperties": False
                }
            },
            "notes":{"type":"string"}
        },
        "required":["days"],
        "additionalProperties": False
    }

    budget_band = (
        "shoestring" if amount < 50 else
        "moderate" if amount < 150 else
        "premium"
    )

    try:
        comp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a practical travel planner. Build a realistic, walkable itinerary using ONLY places "
                    "from the provided list. Match the user's budget band and interests. Allocate free/low-cost items "
                    "for 'shoestring', balanced picks for 'moderate', and paid experiences for 'premium'. "
                    "Prefer short travel hops. Avoid duplicates across days."
                )},
                {"role": "user", "content": json.dumps({
                    "city": city, "area": area, "days": days,
                    "budget_band": budget_band,
                    "interests": interests,
                    "allowed_place_names": osm_names
                }, ensure_ascii=False)}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "itinerary_schema",
                    "strict": True,
                    "schema": schema
                }
            },
            temperature=0.4,
            max_tokens=2000,
        )

        # With json_schema, we should receive valid JSON
        raw = comp.choices[0].message.content
        data = json.loads(raw)
        model_days = data.get("days", [])

        grounded = _ground_and_prune(model_days, set(osm_names))
        if not grounded or all(not any(v for v in d.values()) for d in grounded):
            return assemble_itinerary(lat, lon, city, area, start, end, interests, amount)

        # Convert to your existing rendering format
        out = []
        for d in grounded:
            out.append({
                "Morning": [{"name": x["name"]} for x in d.get("Morning", [])][:1],
                "Afternoon": [{"name": x["name"]} for x in d.get("Afternoon", [])][:1],
                "Evening": [{"name": x["name"]} for x in d.get("Evening", [])][:1],
            })
        header = f"## {city} Itinerary ({days} days)"
        return header, out

    except Exception as e:
        st.warning(f"OpenAI plan generation failed ({e}); using local fallback.")
        return assemble_itinerary(lat, lon, city, area, start, end, interests, amount)

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="MonTravels", page_icon="🧭", layout="wide")
apply_pokemon_theme()
st.title("🧭 MonTravels")

with st.sidebar:
    city=st.text_input("Destination*").strip()
    area=st.text_input("Area (optional)").strip()
    c1,c2=st.columns(2)
    with c1: start_date=st.date_input("Start",date.today()+timedelta(days=7))
    with c2: end_date=st.date_input("End",date.today()+timedelta(days=10))
    adults=st.number_input("Adults",1,10,2)
    budget=st.number_input("Budget ($/day)",10,1000,100)
    interests=st.multiselect("Interests",["food","history","museums","nature","nightlife"],default=["food","history"])
    go=st.button("✨ Build Plan")

uid=get_user_id()
st.caption(f"User: `{uid}`")

if go:
    if not city:
        st.error("Enter a city."); st.stop()
    geo=geocode_osm(combined_query(city,area) or city)
    if not geo:
        st.error("Could not geocode."); st.stop()

    # Model-assisted plan (grounded on OSM & budget-aware)
    header,days_plan=generate_itinerary_with_openai(
        city=city, area=area, start=start_date, end=end_date,
        lat=geo["lat"], lon=geo["lon"],
        interests=interests, amount=budget
    )

    st.subheader("🗓️ Itinerary")
    st.markdown(render_itinerary(header,days_plan))
    st.markdown(budget_notes(budget))

    # ---- Side by side layout ----
    st.subheader("🏨 Places to Stay & ✈️ Travel Partners")
    col1,col2=st.columns([2,1])

    with col1:
        hotel_cards=synthesize_hotel_cards(city,area,start_date,end_date,adults,interests,budget,derive_interest_bias(uid))
        for c in hotel_cards:
            with st.container(border=True):
                st.markdown(f"**{c['title']}**")
                st.caption(c["why"])
                st.write("Tags:",", ".join(c["tags"]))
                external_link_button("Open on Booking.com",c["link"])
        external_link_button("🔗 See all on Booking.com",deeplink_booking_city(city,start_date,end_date,adults))

    with col2:
        agents=[{"name":"GlobeTrek Tours","desc":"Cultural & family packages","email":"info@globetrek.com","link":"https://globetrek.example.com"},
                {"name":"SkyHigh Travels","desc":"Custom itineraries & visa support","email":"bookings@skyhigh.example.com","link":"https://skyhigh.example.com"}]
        summary=f"Destination: {city}\nDates: {start_date}→{end_date}\nBudget:${budget}/day\nAdults:{adults}\nInterests:{', '.join(interests)}"
        for a in agents:
            with st.container(border=True):
                st.markdown(f"**{a['name']}**")
                st.caption(a["desc"])
                external_link_button("🌍 Visit Website",a["link"])
                subject=urllib.parse.quote(f"MonTravels Plan — {city} {start_date}→{end_date}")
                body=urllib.parse.quote(f"Hello {a['name']},\n\nHere is my MonTravels plan:\n{summary}\n\nPlease help me book this trip.")
                external_link_button("📧 Send My Plan",f"mailto:{a['email']}?subject={subject}&body={body}")

    add_history(uid,{"city":city,"area":area,"start":str(start_date),"end":str(end_date),"budget":budget,"interests":interests})
    st.subheader("🧠 History")
    st.json(get_user_history(uid))
else:
    st.info("Enter details and click Build Plan.")
