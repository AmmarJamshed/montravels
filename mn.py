import os
from datetime import date, timedelta
import requests
import streamlit as st
from langchain_openai import ChatOpenAI

# ================================
# Page Config + Theme
# ================================
st.set_page_config(page_title="MonTravels", page_icon="🧭", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F5F7FA; font-family: 'Trebuchet MS', sans-serif; }
    h1 { color: #FFCC00; text-shadow: 2px 2px 0px #3B4CCA; }
    h2, h3 { color: #3B4CCA; }
    div.stButton > button {
        background-color: #FF1C1C; color: white;
        border-radius: 8px; border: 2px solid #3B4CCA; font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #FFCC00; color: #2C2C2C; border: 2px solid #FF1C1C;
    }
    section[data-testid="stSidebar"] { background-color: #3B4CCA; color: white; }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div[role="button"] {
        color: white !important;
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] select {
        color: #0f172a !important;
        background-color: #eef2ff !important;
        border-radius: 6px !important;
    }
    .agent-card {
        background-color: white; padding: 15px; margin: 10px 0;
        border-radius: 10px; box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
    .agent-card h4 { color: #3B4CCA; margin-bottom: 5px; }
    .agent-card p { margin: 2px 0; }
    </style>
""", unsafe_allow_html=True)

st.title("🧭 MonTravels – Travel with Wisdom")

# ================================
# Keys
# ================================
RAPIDAPI_KEY = st.secrets["RAPIDAPI_KEY"]

# ================================
# Initialize Groq (via LangChain)
# ================================
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.4,
    max_tokens=2800,
)

# ================================
# RapidAPI Hotel Functions
# ================================
def get_dest_id(city, area=""):
    """
    Fetch dest_id for Booking.com searches
    """
    url = "https://booking-com.p.rapidapi.com/v1/hotels/locations"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
    }
    query = f"{city} {area}".strip()
    resp = requests.get(url, headers=headers, params={"name": query, "locale": "en-us"})
    
    if resp.status_code != 200:
        return None
    
    data = resp.json()
    # Prefer city-type destinations
    for d in data:
        if d.get("dest_type") == "city":
            return d.get("dest_id")
    # fallback: take first available
    if data:
        return data[0].get("dest_id")
    return None


RAPID_API_KEY = st.secrets["RAPIDAPI_KEY"]

def fetch_hotels(city, area, checkin, checkout, adults):
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
    }

    # Step 1: Get destination ID
    query = f"{city}, {area}" if area else city
    loc_url = "https://booking-com.p.rapidapi.com/v1/hotels/locations"
    loc_params = {"name": query, "locale": "en-us"}
    loc_resp = requests.get(loc_url, headers=headers, params=loc_params)
    if not loc_resp.ok:
        return []
    locations = loc_resp.json()
    if not locations:
        return []

    dest_id = locations[0].get("dest_id")

    # Step 2: Search hotels
    search_url = "https://booking-com.p.rapidapi.com/v1/hotels/search"
    params = {
        "checkin_date": checkin,
        "checkout_date": checkout,
        "dest_id": dest_id,
        "dest_type": "city",
        "adults_number": adults,
        "order_by": "price",
        "locale": "en-us",
        "units": "metric",
        "room_number": 1,
        "filter_by_currency": "USD"
    }

    resp = requests.get(search_url, headers=headers, params=params)
    if not resp.ok:
        return []
    results = resp.json().get("result", [])

    hotels = []
    for r in results[:5]:  # show top 5
        hotels.append({
            "name": r.get("hotel_name"),
            "price": r.get("price_breakdown", {}).get("all_inclusive_price"),
            "rating": r.get("review_score"),
            "link": f"https://www.booking.com/hotel/{r.get('hotel_id_encrypted')}.html"
        })
    return hotels


# ================================
# Function to generate itinerary
# ================================
def generate_itinerary(city, area, start, end, interests, budget, adults):
    days = max((end - start).days, 1)
    prompt = f"""
    You are an expert travel planner.

    Create a detailed {days}-day travel itinerary for {city}, {area or ''}.
    It MUST include exactly {days} full days, clearly labeled as:
    Day 1, Day 2, ..., Day {days}.
    
    For each day:
    - Morning, Afternoon, and Evening activities
    - Meals with budget-friendly suggestions
    - Practical tips
    - Daily notes

    Make sure the plan is COMPLETE and does not stop mid-way.
    If you run out of space, summarize but cover all {days} days.
    
    Focus on interests: {', '.join(interests)}.
    Budget: ${budget} per day for {adults} adults.
    """
    resp = llm.invoke(prompt)
    return resp.content

# ================================
# Sidebar Inputs
# ================================
with st.sidebar:
    city = st.text_input("Destination*").strip()
    area = st.text_input("Area (optional)").strip()
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("Start", date.today() + timedelta(days=7))
    with c2: end_date   = st.date_input("End",   date.today() + timedelta(days=10))
    adults  = st.number_input("Adults", 1, 10, 2)
    budget  = st.number_input("Budget ($/day)", 10, 1000, 100)
    interests = st.multiselect(
        "Interests",
        ["food","history","museums","nature","nightlife"],
        default=["food","history"]
    )
    go = st.button("✨ Build Plan")

# ================================
# Main Action
# ================================
if go:
    if not city:
        st.error("Please enter a destination.")
        st.stop()

    with st.spinner("Building your personalized itinerary..."):
        itinerary = generate_itinerary(city, area, start_date, end_date, interests, budget, adults)

    # Two-column layout
    col1, col2 = st.columns([2, 1])

    # --- LEFT: Itinerary ---
    with col1:
        st.subheader("🗓️ Your Itinerary")
        st.write(itinerary)

    # --- RIGHT: Hotels + Agents ---
    with col2:
        st.subheader("🏨 Suggested Hotels & Lodges")
        checkin = start_date.strftime("%Y-%m-%d")
        checkout = end_date.strftime("%Y-%m-%d")
        hotels = fetch_hotels(city, area, checkin, checkout, adults)

        if not hotels:
            st.caption("No hotels found.")
        else:
            for h in hotels:
                if h["link"]:
                    st.markdown(f"""
**[{h['name']}]({h['link']})**  
💵 Price: {h['price']} USD  
⭐ Rating: {h['rating']}
""")
                else:
                    st.write(h["name"])

        # Travel Agents block
        st.subheader("✈️ Travel Agents")
        agents = [
            {"name": "GlobeTrek Tours", "desc": "Cultural & family packages", "email": "info@globetrek.com"},
            {"name": "SkyHigh Travels", "desc": "Custom itineraries & visa support", "email": "bookings@skyhigh.com"}
        ]
        for a in agents:
            st.markdown(f"""
            <div class="agent-card">
                <h4>{a['name']}</h4>
                <p>{a['desc']}</p>
                <p><a href="mailto:{a['email']}?subject=MonTravels {city} Trip">📧 Contact</a></p>
            </div>
            """, unsafe_allow_html=True)
