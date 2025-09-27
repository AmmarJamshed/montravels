import os
from datetime import date, timedelta
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
    /* Sidebar Styling */
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
    </style>
""", unsafe_allow_html=True)

st.title("🧭 MonTravels – Travel with Wisdom")

# ================================
# Initialize Groq (via LangChain)
# ================================
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",   # or try "llama-3.1-70b-versatile"
    api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.4,
    max_tokens=800,
)

# ================================
# Function to generate itinerary
# ================================
def generate_itinerary(city, area, start, end, interests, budget, adults):
    days = max((end - start).days, 1)
    prompt = f"""
    You are an expert travel planner.

    Make me a {days}-day itinerary for {city}, {area or ''}.
    Focus on interests: {', '.join(interests)}.
    Budget: ${budget} per day.
    Adults traveling: {adults}.

    Include:
    - Morning, Afternoon, and Evening activities each day.
    - Mix of food, culture, and history (based on interests).
    - Stay within budget with practical tips.
    - Add short daily notes at the end of each day.
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
    col1, col2 = st.columns([2,1])

    # --- LEFT: Itinerary ---
    with col1:
        st.subheader("🗓️ Your Itinerary")
        st.write(itinerary)

    # --- RIGHT: Booking & Agents ---
    with col2:
        st.subheader("🏨 Book Your Stay")

        # Booking.com deeplink
        checkin = start_date.strftime("%Y-%m-%d")
        checkout = end_date.strftime("%Y-%m-%d")
        booking_url = f"https://www.booking.com/searchresults.html?ss={city}&checkin={checkin}&checkout={checkout}&group_adults={adults}"
        st.markdown(f"[🔗 Booking.com Hotels in {city}]({booking_url})")

        # Airbnb deeplink
        airbnb_url = f"https://www.airbnb.com/s/{city}/homes?checkin={checkin}&checkout={checkout}&adults={adults}"
        st.markdown(f"[🔗 Airbnb Stays in {city}]({airbnb_url})")

        st.subheader("✈️ Travel Agents")
        agents = [
            {"name": "GlobeTrek Tours", "desc": "Cultural & family packages", "email": "info@globetrek.com"},
            {"name": "SkyHigh Travels", "desc": "Custom itineraries & visa support", "email": "bookings@skyhigh.com"}
        ]
        for a in agents:
            st.markdown(f"**{a['name']}** – {a['desc']}")
            st.markdown(f"[📧 Contact {a['name']}](mailto:{a['email']}?subject=MonTravels {city} Trip)")
            st.markdown("---")

else:
    st.info("Enter details in the sidebar and click **Build Plan**.")
