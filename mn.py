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

# LangChain Groq
from langchain_openai import ChatOpenAI

# ================================
# Page + Theme
# ================================
st.set_page_config(page_title="MonTravels", page_icon="🧭", layout="wide")

def apply_theme():
    st.markdown("""
        <style>
        .stApp { background-color: #F5F7FA; font-family: 'Trebuchet MS', sans-serif; color: #2C2C2C; }
        h1 { color: #FFCC00; text-shadow: 2px 2px 0px #3B4CCA; }
        h2, h3 { color: #3B4CCA; }

        section[data-testid="stSidebar"] { background-color: #3B4CCA; color: white; }
        section[data-testid="stSidebar"] * { color: white !important; }

        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] select,
        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] div,
        section[data-testid="stSidebar"] .stNumberInput input,
        section[data-testid="stSidebar"] .stDateInput input {
            color: #000000 !important;          /* black text */
            background-color: #eef2ff !important;
            border-radius: 10px !important;
        }
        section[data-testid="stSidebar"] input::placeholder,
        section[data-testid="stSidebar"] textarea::placeholder {
            color: #555555 !important;   /* gray placeholder */
        }

        div.stButton > button {
            background-color: #FF1C1C; color: white;
            border-radius: 12px; border: 2px solid #3B4CCA; font-weight: bold; transition: 0.2s;
        }
        div.stButton > button:hover {
            background-color: #FFCC00; color: #2C2C2C; border: 2px solid #FF1C1C;
        }
        </style>
    """, unsafe_allow_html=True)

apply_theme()
st.title("🧭 MonTravels")

# ================================
# Groq LLM
# ================================
def groq_generate_text(prompt: str, max_new_tokens: int = 600, temperature: float = 0.4) -> str:
    """Generate text using Groq API via LangChain."""
    llm = ChatOpenAI(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    try:
        result = llm.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        st.error(f"Groq error: {e}")
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

# ================================
# Geocoding with fallbacks
# ================================
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city(query: str) -> Optional[Dict]:
    q = (query or "").strip()
    if not q: return None
    headers = {"User-Agent": "MonTravelsApp/1.0"}

    # 1. Nominatim
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": q, "format": "json", "limit": 1},
                         headers=headers, timeout=8)
        if r.ok:
            js = r.json() or []
            if js:
                return {"lat": float(js[0]["lat"]), "lon": float(js[0]["lon"]), "name": js[0]["display_name"]}
    except Exception:
        pass

    # 2. Maps.co
    try:
        r = requests.get("https://geocode.maps.co/search",
                         params={"q": q, "limit": 1},
                         headers=headers, timeout=8)
        if r.ok:
            js = r.json() or []
            if js:
                return {"lat": float(js[0]["lat"]), "lon": float(js[0]["lon"]), "name": js[0].get("display_name") or q}
    except Exception:
        pass

    # 3. Photon (Komoot)
    try:
        r = requests.get("https://photon.komoot.io/api/",
                         params={"q": q, "limit": 1},
                         headers=headers, timeout=8)
        if r.ok:
            js = r.json() or {}
            feats = (js.get("features") or [])
            if feats:
                coords = feats[0]["geometry"]["coordinates"]
                props = feats[0].get("properties", {})
                name = props.get("name") or props.get("city") or props.get("country") or q
                return {"lat": float(coords[1]), "lon": float(coords[0]), "name": name}
    except Exception:
        pass

    return None

# ================================
# OSM POIs (basic)
# ================================
OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"

@st.cache_data(ttl=600, show_spinner=False)
def fetch_pois(lat: float, lon: float, radius_m: int = 2500, kind: str = "landmark", limit: int = 30) -> List[Dict]:
    try:
        q = f"[out:json][timeout:20];node[tourism=attraction](around:{radius_m},{lat},{lon});out center {limit};"
        r = requests.post(OVERPASS_ENDPOINT, data={"data": q}, timeout=20)
        r.raise_for_status()
        elements = r.json().get("elements", [])
        out, seen = [], set()
        for e in elements:
            name = e.get("tags", {}).get("name")
            if not name or name in seen: continue
            seen.add(name)
            lat_, lon_ = e.get("lat"), e.get("lon")
            if lat_ and lon_: out.append({"name": name, "lat": float(lat_), "lon": float(lon_)})
        return out[:limit]
    except Exception:
        return []

# ================================
# Safe JSON parsing
# ================================
def safe_json_parse(text: str) -> Dict:
    try:
        return json.loads(text)
    except:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    return {}

# ================================
# Itinerary with Groq
# ================================
def generate_itinerary_groq(city: str, area: Optional[str], start: date, end: date,
                          lat: float, lon: float, interests: List[str], amount: int) -> Tuple[str, List[Dict]]:
    days = max((end - start).days, 1)
    osm_places = fetch_pois(lat, lon, 3000, "landmark", 40)
    place_names = [p["name"] for p in osm_places]

    prompt = (
        "You are a travel planner.\n"
        "TASK: Create an itinerary with exactly one activity for Morning, Afternoon, Evening per day.\n"
        f"City: {city}, Area: {area or '—'}, Days: {days}, Interests: {', '.join(interests)}.\n"
        f"Budget per day: {amount} USD.\n"
        f"Allowed places: {place_names}.\n"
        "Return valid JSON in this schema:\n"
        "{\"days\":[{\"Morning\":[{\"name\":str}],\"Afternoon\":[{\"name\":str}],\"Evening\":[{\"name\":str}],\"daily_notes\":str}],\"notes\":str}\n"
        "ABSOLUTE RULES: Output JSON only. No text, no explanations, no markdown. If unsure, output {}."
    )

    raw = groq_generate_text(prompt, max_new_tokens=600, temperature=0.4)
    data = safe_json_parse(raw)

    if not data or "days" not in data:
        st.warning("Could not parse model output, falling back to simple itinerary.")
        return f"## {city} Itinerary ({days} days)", [
            {"Morning": osm_places[:1], "Afternoon": osm_places[1:2], "Evening": osm_places[2:3]}
        ]

    return f"## {city} Itinerary ({days} days)", data.get("days", [])

# ================================
# Sidebar
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
    show_debug = st.checkbox("Show debug")
    go = st.button("✨ Build Plan")

if show_debug and st.button("🧪 Test Groq"):
    test = groq_generate_text("Return ONLY this JSON: {\"ok\": true}", max_new_tokens=40, temperature=0.01)
    st.code(test or "(empty)", language="json")

# ================================
# ACTION
# ================================
if go:
    if not city:
        st.error("Enter a city."); st.stop()
        with st.spinner("Building your plan…"):
            geo = geocode_city(city)
            if not geo:
                st.error("Could not find that destination.")
                st.stop()

            header, days_plan = generate_itinerary_groq(
                city=city, area=area, start=start_date, end=end_date,
                lat=geo["lat"], lon=geo["lon"],
                interests=interests, amount=budget
        )

            st.subheader("🗓️ Itinerary")
            if header:
                st.markdown(header)
            if days_plan:
                for i, day in enumerate(days_plan, 1):
                    st.markdown(f"### Day {i}")
                    for slot in ["Morning", "Afternoon", "Evening"]:
                        items = day.get(slot, [])
                        if items:
                            name = items[0].get("name", "")
                            st.markdown(f"- **{slot}**: {name}")
                    if "daily_notes" in day and day["daily_notes"]:
                        st.caption(day["daily_notes"])
else:
    st.info("Enter details and click **Build Plan**.")
