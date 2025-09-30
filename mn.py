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
GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]

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
# Google Maps Places API
# ================================
def fetch_google_places(city, place_type="lodging"):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{place_type} in {city}",
        "key": GOOGLE_MAPS_API_KEY
    }
    response = requests.get(url, params=params)
    if not response.ok:
        return []

    data = response.json()
    places = []
    for item in data.get("results", [])[:5]:  # Top 5 results
        maps_url = f"https://www.google.com/maps/place/?q=place_id:{item['place_id']}"
        places.append({
            "name": item.get("name"),
            "address": item.get("formatted_address"),
            "rating": item.get("rating"),
            "link": maps_url
        })
    return places

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
    lodging_choice = st.selectbox(
        "Lodging Type",
        ["All", "Hotels", "Residences", "Motels"]
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

    # --- RIGHT: Lodging Options ---
    with col2:
        st.subheader("🏨 Lodging Suggestions")

        if lodging_choice in ["All", "Hotels"]:
            st.markdown("### 🏨 Hotels")
            hotels = fetch_google_places(city, "hotel")
            if not hotels:
                st.caption("No hotels found.")
            else:
                for h in hotels:
                    st.markdown(f"""
**[{h['name']}]({h['link']})**  
📍 {h['address']}  
⭐ Rating: {h.get('rating', 'N/A')}
""")

        if lodging_choice in ["All", "Residences"]:
            st.markdown("### 🏡 Residences & Apartments")
            residences = fetch_google_places(city, "residence")
            if not residences:
                st.caption("No residences found.")
            else:
                for r in residences:
                    st.markdown(f"""
**[{r['name']}]({r['link']})**  
📍 {r['address']}  
⭐ Rating: {r.get('rating', 'N/A')}
""")

        if lodging_choice in ["All", "Motels"]:
            st.markdown("### 🛏️ Motels")
            motels = fetch_google_places(city, "motel")
            if not motels:
                st.caption("No motels found.")
            else:
                for m in motels:
                    st.markdown(f"""
**[{m['name']}]({m['link']})**  
📍 {m['address']}  
⭐ Rating: {m.get('rating', 'N/A')}
""")

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
