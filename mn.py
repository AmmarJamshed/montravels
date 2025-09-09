import math
import json
import time
import hashlib
from datetime import date, timedelta
import urllib.parse
import requests
import streamlit as st
import uuid
from typing import Optional

# =========================================================
# THEME (Pokémon-inspired Travel Guide)
# =========================================================
def apply_pokemon_theme():
    st.markdown("""
        <style>
        .stApp {
            background-color: #F5F7FA;
            font-family: 'Trebuchet MS', sans-serif;
            color: #2C2C2C;
        }
        h1 {
            color: #FFCC00;
            text-shadow: 2px 2px 0px #3B4CCA;
        }
        h2, h3 {
            color: #3B4CCA;
        }
        section[data-testid="stSidebar"] {
            background-color: #3B4CCA;
            color: white;
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3, 
        section[data-testid="stSidebar"] label, 
        section[data-testid="stSidebar"] span {
            color: white !important;
        }
        div.stButton > button {
            background-color: #FF1C1C;
            color: white;
            border-radius: 12px;
            border: 2px solid #3B4CCA;
            font-weight: bold;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #FFCC00;
            color: #2C2C2C;
            border: 2px solid #FF1C1C;
        }
        .stContainer {
            background-color: #FFFFFF;
            border-radius: 16px;
            padding: 12px;
            margin-bottom: 12px;
            border: 2px solid #FFCC00;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        }
        a {
            color: #3B4CCA;
            text-decoration: none;
            font-weight: bold;
        }
        a:hover {
            color: #FF1C1C;
        }
        .stCaption {
            color: #4CAF50 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# =========================================================
# Utilities
# =========================================================

def haversine_km(lat1, lon1, lat2, lon2):
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

def deeplink_booking_city(city_or_area: str, checkin: date, checkout: date, adults: int = 2):
    q = urllib.parse.quote(city_or_area)
    return (
        "https://www.booking.com/searchresults.html"
        f"?ss={q}"
        f"&checkin={checkin:%Y-%m-%d}"
        f"&checkout={checkout:%Y-%m-%d}"
        f"&group_adults={adults}&no_rooms=1&group_children=0"
        f"&lang=en-us&_mtu={_cachebuster(city_or_area)}"
    )

def deeplink_booking_with_keywords(city: str, area: Optional[str], keywords: str,
                                   checkin: date, checkout: date, adults: int = 2):
    parts = [city]
    if area:
        parts.append(area)
    if keywords:
        parts.append(keywords)
    ss_raw = " ".join(parts).strip()
    ss = urllib.parse.quote(ss_raw)
    return (
        "https://www.booking.com/searchresults.html"
        f"?ss={ss}"
        f"&checkin={checkin:%Y-%m-%d}"
        f"&checkout={checkout:%Y-%m-%d}"
        f"&group_adults={adults}&no_rooms=1&group_children=0"
        f"&lang=en-us&_mtu={_cachebuster(ss_raw)}"
    )

def external_link_button(label: str, url: str):
    st.markdown(
        f'<a target="_blank" rel="noopener" href="{url}" '
        f'style="text-decoration:none;"><button class="stButton">{label}</button></a>',
        unsafe_allow_html=True
    )

# =========================================================
# Session helpers (user id & history)
# =========================================================

def get_user_id() -> str:
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid.uuid4())[:8]
    return st.session_state["user_id"]

def get_user_history(uid: str):
    return st.session_state.get(f"history_{uid}", [])

def add_history(uid: str, trip: dict):
    key = f"history_{uid}"
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].append(trip)
    st.session_state[key] = st.session_state[key][-10:]

# =========================================================
# Stubs for missing functions (replace with real logic)
# =========================================================

def geocode_osm(query: str):
    # Dummy geocode
    return {"name": query.title(), "lat": 40.0, "lon": 30.0}

def assemble_itinerary(lat, lon, city, area, start_date, end_date, interests, budget):
    header = f"Trip to {city}"
    days_plan = {f"{start_date}": [f"Explore {city} center"]}
    return header, days_plan

def render_itinerary_markdown(header, days_plan):
    return f"### {header}\n\n{json.dumps(days_plan, indent=2)}"

def budget_notes(budget: int):
    if budget < 50:
        return "💸 Budget-friendly trip — expect hostels and street food."
    elif budget < 200:
        return "💰 Mid-range trip — mix of hotels & restaurants."
    else:
        return "🏰 Luxury trip — premium stays & fine dining."

def derive_interest_bias(uid: str):
    return {"food": 1.0, "history": 0.8}

def synthesize_hotel_cards(city, area, start, end, adults, interests, budget, bias, k=5):
    return [
        {
            "title": f"{city} Stay {i+1}",
            "why": f"Great for {', '.join(interests)}",
            "tags": interests,
            "link": deeplink_booking_city(city, start, end, adults)
        }
        for i in range(k)
    ]

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
        st.error("Please enter a destination city.")
        st.stop()
    if end_date <= start_date:
        st.error("End date must be after Start date.")
        st.stop()

    q = combined_query(city, area or None)
    geo = geocode_osm(q) or geocode_osm(city)
    if not geo:
        st.error("Could not geolocate that place. Try a simpler query (just the city).")
        st.stop()

    st.caption(f"📍 {geo['name']}  ({geo['lat']:.4f}, {geo['lon']:.4f})")

    with st.spinner("Finding nearby places & crafting itinerary..."):
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
        city, (area or None), start_date, end_date, adults,
        interests, int(budget_amount), user_bias, k=5
    )
    for c in hotel_cards:
        with st.container():
            st.markdown(f"**{c['title']}**")
            st.caption(c["why"])
            st.write("Tags:", ", ".join(c["tags"]))
            external_link_button("Open on Booking.com", c["link"])

    external_link_button(
        "🔗 See full results on Booking.com",
        deeplink_booking_city(q, start_date, end_date, adults)
    )

    add_history(uid, {
        "city": city,
        "area": (area or "").strip(),
        "start": f"{start_date}",
        "end": f"{end_date}",
        "adults": adults,
        "budget": int(budget_amount),
        "interests": interests
    })

    pkg = {
        "itinerary_header": header,
        "itinerary": days_plan,
        "budget_per_day": int(budget_amount),
        "budget_notes": budget_notes(int(budget_amount)),
        "stay_recommendations": hotel_cards,
        "deeplink_city": deeplink_booking_city(q, start_date, end_date, adults)
    }
    st.download_button("⬇️ Download Plan (JSON)",
                       data=json.dumps(pkg, ensure_ascii=False, indent=2),
                       file_name=f"montravels_{city.lower().replace(' ','_')}.json",
                       mime="application/json")

    st.subheader("🧠 Your Saved History (Private to this user)")
    st.json(get_user_history(uid))

else:
    st.info("Enter a city (and optional area), pick dates & budget, select interests, then click **Build Personalized Plan**.")
