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

        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] select,
        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] div,
        section[data-testid="stSidebar"] .stNumberInput input,
        section[data-testid="stSidebar"] .stDateInput input {
        color: #000000 !important;          /* make text always black */
        background-color: #eef2ff !important;
        border-radius: 10px !important;
}
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

apply_theme()
st.title("🧭 MonTravels")

# ================================
# Groq LLM
# ================================
def groq_generate_text(prompt: str, max_new_tokens: int = 600, temperature: float = 0.4) -> str:
    """Generate text using Groq API via LangChain."""
    llm = ChatOpenAI(
        model="llama-3.1-8b-instant",   # ✅ working Groq model
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
            f"?ss={q}&checkin={checkin:%Y-%m-%d}&checkout={checkout:%Y-%m-%d}"
            f"&group_adults={adults}&no_rooms=1&group_children=0&lang=en-us&_mtu={_cachebuster(city_or_area)}")

# ================================
# Geocoding
# ================================
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city(query: str) -> Optional[Dict]:
    q = (query or "").strip()
    if not q: return None
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": q, "format": "json", "limit": 1},
                         headers={"User-Agent": "MonTravels/1.0"}, timeout=8)
        if r.ok:
            js = r.json() or []
            if js:
                return {"lat": float(js[0]["lat"]), "lon": float(js[0]["lon"]), "name": js[0]["display_name"]}
    except Exception:
        pass
    return None

# ================================
# OSM POIs
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
# Itinerary Generation with Groq
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
        "{\"days\":[{\"Morning\":[{\"name\":str}],\"Afternoon\":[{\"name\":str}],\"Evening\":[{\"name\":str}],\"daily_notes\":str}],\"notes\":str}"
    )

    raw = groq_generate_text(prompt, max_new_tokens=600, temperature=0.4)

    try:
        data = json.loads(raw)
    except Exception:
        st.warning("Could not parse model output, falling back to simple itinerary.")
        return f"## {city} Itinerary ({days} days)", [{"Morning":osm_places[:1], "Afternoon":osm_places[1:2], "Evening":osm_places[2:3]}]

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
    st.markdown(header)
    st.json(days_plan)
else:
    st.info("Enter details and click **Build Plan**.")
