
import os
import re
import json
import time
import hashlib
import urllib.parse
from datetime import date, timedelta
from typing import Optional, List, Dict, Tuple

import requests
import streamlit as st
from bs4 import BeautifulSoup
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
            color: #000000 !important;
            background-color: #eef2ff !important;
            border-radius: 10px !important;
        }
        section[data-testid="stSidebar"] input::placeholder,
        section[data-testid="stSidebar"] textarea::placeholder {
            color: #555555 !important;
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
def _cachebuster(seed: str) -> str:
    raw = f"{seed}-{time.time_ns()}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]

# ================================
# Geocoding with fallbacks
# ================================
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city(city: str, area: Optional[str] = None) -> Optional[Dict]:
    q = f"{area}, {city}" if area else city
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": q, "format": "json", "limit": 1},
                         headers={"User-Agent": "MonTravels/1.0"}, timeout=8)
        if r.ok:
            js = r.json() or []
            if js:
                return {
                    "lat": float(js[0]["lat"]),
                    "lon": float(js[0]["lon"]),
                    "name": js[0]["display_name"]
                }
    except Exception:
        return None
    return None

# ================================
# OSM POIs
# ================================
OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"

@st.cache_data(ttl=600, show_spinner=False)
def fetch_pois(lat: float, lon: float, radius_m: int = 2500, limit: int = 30) -> List[Dict]:
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
# TripAdvisor scraping
# ================================
@st.cache_data(ttl=3600, show_spinner=False)
def scrape_tripadvisor(city: str) -> List[str]:
    try:
        city_q = urllib.parse.quote_plus(city)
        url = f"https://www.tripadvisor.com/Search?q={city_q}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if not r.ok: return []
        soup = BeautifulSoup(r.text, "html.parser")
        places = [tag.get_text(strip=True) for tag in soup.select("div.result-title")]
        return places[:15]
    except Exception as e:
        st.warning(f"TripAdvisor scraping failed: {e}")
        return []

# ================================
# JSON parsing
# ================================
def safe_json_parse(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
        text = text.rstrip("```").strip()

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
# Itinerary generation
# ================================
def generate_itinerary_groq(city: str, area: Optional[str], start: date, end: date,
                          lat: float, lon: float, interests: List[str], amount: int) -> Tuple[str, List[Dict]]:
    days = max((end - start).days, 1)

    osm_places = fetch_pois(lat, lon, 3000, 40)
    trip_places = scrape_tripadvisor(city)
    place_names = [p["name"] for p in osm_places] + trip_places

    prompt = (
    "You are an expert travel planner.\n"
    f"Destination: {city}, Area: {area or '—'}\n"
    f"Trip Length: {days} days, Adults: {adults}\n"
    f"Interests: {', '.join(interests)}\n"
    f"Budget per day: ${amount}\n\n"
    "Candidate Places (from OpenStreetMap + TripAdvisor near the given Area):\n"
    f"{place_names}\n\n"
    "TASK:\n"
    "1. Build a full travel itinerary based on this specific Area.\n"
    "2. Each day must have Morning, Afternoon, Evening.\n"
    "3. Use places close to the Area for walkability.\n"
    "4. Ensure variety (no repeating the same spots).\n"
    "5. Add a 'daily_notes' about budget/travel tips.\n"
    "OUTPUT JSON ONLY.\n"
)                        

    raw = groq_generate_text(prompt, max_new_tokens=600, temperature=0.4)
    data = safe_json_parse(raw)

    if not data or "days" not in data:
        st.warning("Could not parse model output, falling back to simple itinerary.")
        return f"## {city} Itinerary ({days} days)", [
            {"Morning": osm_places[:1], "Afternoon": osm_places[1:2], "Evening": osm_places[2:3], "daily_notes": ""}
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

    if not days_plan:
        st.warning("No plan generated. Try again or change inputs.")
    else:
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
