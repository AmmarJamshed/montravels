import os
import re
import json
import requests
import urllib.parse
import streamlit as st
from datetime import date, timedelta
from typing import Optional, List, Dict
from bs4 import BeautifulSoup
from groq import Groq

# ================================
# Page Config + Theme
# ================================
st.set_page_config(page_title="MonTravels", page_icon="🧭", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F5F7FA; font-family: 'Trebuchet MS', sans-serif; }
    h1 { color: #FFCC00; text-shadow: 2px 2px 0px #3B4CCA; }
    h2, h3 { color: #3B4CCA; }
    section[data-testid="stSidebar"] { background-color: #3B4CCA; color: white; }
    section[data-testid="stSidebar"] * { color: white !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🧭 MonTravels")

# ================================
# Groq Client
# ================================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================================
# Geocoding (city + area)
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
                return {"lat": float(js[0]["lat"]), "lon": float(js[0]["lon"]), "name": js[0]["display_name"]}
    except Exception:
        return None
    return None

# ================================
# Overpass API (OSM POIs)
# ================================
OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"

def fetch_pois(lat: float, lon: float, radius_m: int = 3000, limit: int = 40) -> List[Dict]:
    q = f"""
    [out:json][timeout:25];
    (
      node["tourism"](around:{radius_m},{lat},{lon});
      node["amenity"](around:{radius_m},{lat},{lon});
      node["historic"](around:{radius_m},{lat},{lon});
    );
    out center {limit};
    """
    try:
        r = requests.post(OVERPASS_ENDPOINT, data={"data": q}, timeout=25)
        r.raise_for_status()
        elements = r.json().get("elements", [])
        out = []
        for e in elements:
            tags = e.get("tags", {})
            name = tags.get("name")
            if name:
                out.append({"name": name, "lat": e.get("lat"), "lon": e.get("lon")})
        return out[:limit]
    except Exception:
        return []

# ================================
# TripAdvisor Scraper
# ================================
@st.cache_data(ttl=3600, show_spinner=False)
def scrape_tripadvisor(city: str, area: Optional[str] = None) -> List[str]:
    try:
        query = f"{area} {city}" if area else city
        city_q = urllib.parse.quote_plus(query)
        url = f"https://www.tripadvisor.com/Search?q={city_q}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if not r.ok:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        places = [tag.get_text(strip=True) for tag in soup.select("div.result-title")]
        return places[:15]
    except Exception:
        return []

# ================================
# Safe JSON Parse
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
# Generate Itinerary with Groq
# ================================
def generate_itinerary_groq(city: str, area: Optional[str], start: date, end: date,
                            lat: float, lon: float, interests: List[str], amount: int, adults: int) -> Dict:
    days = max((end - start).days, 1)

    osm_places = fetch_pois(lat, lon, 3000, 40)
    trip_places = scrape_tripadvisor(city, area)
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
        "1. Build a full travel itinerary for the given number of days.\n"
        "2. Each day must include exactly one activity for Morning, Afternoon, and Evening.\n"
        "3. Prioritize places close to the given Area for walkability.\n"
        "4. Ensure variety across days (no repeating the same spots).\n"
        "5. Add a short 'daily_notes' about budget or travel tips.\n\n"
        "OUTPUT RULES:\n"
        "- Return ONLY valid JSON.\n"
        "- Schema:\n"
        "{\n"
        "  \"days\": [\n"
        "    {\"Morning\": [{\"name\": str}], \"Afternoon\": [{\"name\": str}], \"Evening\": [{\"name\": str}], \"daily_notes\": str}\n"
        "  ],\n"
        "  \"notes\": str\n"
        "}\n"
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=900,
    )

    raw = response.choices[0].message.content
    return safe_json_parse(raw)

# ================================
# Sidebar UI
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
    go = st.button("✨ Build Plan")

# ================================
# Action
# ================================
if go:
    if not city:
        st.error("Enter a city."); st.stop()
    geo = geocode_city(city, area)
    if not geo:
        st.error(f"Could not find that destination: {city} {area}"); st.stop()

    with st.spinner("Building itinerary..."):
        data = generate_itinerary_groq(city, area, start_date, end_date,
                                       geo["lat"], geo["lon"], interests, budget, adults)

    if not data or "days" not in data:
        st.error("Model did not return a valid itinerary.")
    else:
        st.subheader("🗓️ Your Itinerary")
        for i, d in enumerate(data["days"], 1):
            st.markdown(f"### Day {i}")
            for part in ["Morning","Afternoon","Evening"]:
                items = d.get(part, [])
                if items:
                    st.write(f"- **{part}**: {items[0]['name']}")
            st.caption(d.get("daily_notes", ""))
        st.subheader("📝 Notes")
        st.write(data.get("notes",""))
else:
    st.info("Enter details and click **Build Plan**.")
