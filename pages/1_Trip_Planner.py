import os
import re
from datetime import date, timedelta, datetime
import requests
import streamlit as st
from bs4 import BeautifulSoup
from supabase import create_client
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ================================
# 🌍 ENV & SUPABASE INIT
# ================================
load_dotenv()
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.4,
    max_tokens=2800,
)

# ================================
# 🌍 PAGE CONFIG
# ================================
st.set_page_config(page_title="✈️ Trip Planner", page_icon="🌍", layout="wide")
st.title("✈️ AI Trip Planner – Plan Your Perfect Adventure")

# Get logged-in user from session
if "user" not in st.session_state:
    st.error("Please log in from the home page first.")
    st.stop()

user = st.session_state["user"]
current_user_id = user["id"]

# ================================
# 📍 Helper Functions
# ================================
def geocode_city(city: str):
    url = "https://nominatim.openstreetmap.org/search"
    r = requests.get(url, params={"q": city, "format": "json", "limit": 1})
    data = r.json()
    return (float(data[0]["lat"]), float(data[0]["lon"])) if data else (None, None)

def fetch_osm_places(city: str, place_type="hotel", limit=5):
    lat, lon = geocode_city(city)
    if not lat or not lon:
        return []
    query = f"""
    [out:json][timeout:25];
    (
      node(around:5000,{lat},{lon})[tourism={place_type}];
      way(around:5000,{lat},{lon})[tourism={place_type}];
      relation(around:5000,{lat},{lon})[tourism={place_type}];
    );
    out center {limit};
    """
    resp = requests.post("https://overpass-api.de/api/interpreter", data={"data": query})
    if not resp.ok: return []
    data = resp.json()
    return [
        {
            "name": el.get("tags", {}).get("name", "Unnamed"),
            "link": f"https://www.openstreetmap.org/{el['type']}/{el['id']}",
            "address": el.get("tags", {}).get("addr:street", ""),
            "rating": el.get("tags", {}).get("stars", "N/A"),
        }
        for el in data.get("elements", [])[:limit]
    ]

def generate_itinerary(city, start, end, interests, budget, adults):
    days = max((end - start).days, 1)
    prompt = f"""
    You are a world-class travel planner. Create a {days}-day itinerary for {city}.
    Include morning, afternoon, and evening activities, meal suggestions, and daily travel tips.
    - Focus on: {', '.join(interests)}
    - Budget: ${budget}/day for {adults} adults
    - Show day-by-day plan with recommended times.
    """
    return llm.invoke(prompt).content

def scrape_news(city: str, max_articles=6):
    query = f"{city} travel OR safety OR protests OR incidents site:bbc.com OR site:cnn.com OR site:reuters.com"
    url = f"https://duckduckgo.com/html/?q={query}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "lxml")
    return [{"title": a.get_text(strip=True), "url": a.get("href")} for a in soup.select("a.result__a")[:max_articles]]

def summarize_news(city, articles):
    if not articles:
        return "No recent news found. Likely stable and safe."
    headlines = "\n".join([f"- {a['title']}" for a in articles])
    prompt = f"Summarize the current safety situation in {city} based on these headlines:\n{headlines}"
    return llm.invoke(prompt).content

# ================================
# 📍 Trip Planner Form
# ================================
st.subheader("🌎 Plan a New Trip")

with st.form("trip_form"):
    city = st.text_input("Destination City *")
    start_date = st.date_input("Start Date", date.today() + timedelta(days=7))
    end_date = st.date_input("End Date", date.today() + timedelta(days=10))
    adults = st.number_input("Number of Adults", 1, 10, 2)
    budget = st.number_input("Daily Budget (USD)", 50, 1000, 200)
    interests = st.multiselect(
        "Travel Interests",
        ["food", "history", "nature", "museums", "nightlife", "shopping"],
        default=["food", "history"]
    )
    submitted = st.form_submit_button("✨ Generate Plan")

if submitted:
    if not city:
        st.error("Please enter a destination city.")
        st.stop()

    with st.spinner("✈️ Generating your itinerary..."):
        itinerary = generate_itinerary(city, start_date, end_date, interests, budget, adults)
        supabase.table("trips").insert({
            "creator_id": current_user_id,
            "city": city,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "itinerary": itinerary
        }).execute()

    st.success("✅ Trip created and saved!")
    st.markdown("### 🗺️ Your AI-Powered Itinerary")
    st.write(itinerary)

    # ================================
    # 🏨 Lodging
    # ================================
    st.subheader("🏨 Lodging Options Near Your Destination")
    hotels = fetch_osm_places(city, "hotel")
    if hotels:
        for h in hotels:
            st.markdown(f"**[{h['name']}]({h['link']})**  \n📍 {h['address']}  \n⭐ Rating: {h['rating']}")
    else:
        st.info("No hotels found in a 5km radius.")

    # ================================
    # 🛡️ Safety Analysis
    # ================================
    st.subheader("🛡️ Safety Insights")
    news = scrape_news(city)
    summary = summarize_news(city, news)
    st.info(summary)
    if news:
        st.markdown("**Latest Headlines:**")
        for n in news:
            st.markdown(f"- [{n['title']}]({n['url']})")

# ================================
# 📊 Past Trips
# ================================
st.subheader("📜 Your Planned Trips")
past_trips = supabase.table("trips").select("*").eq("creator_id", current_user_id).order("start_date", desc=True).execute().data
if past_trips:
    for t in past_trips:
        st.markdown(f"**{t['city']}** – {t['start_date']} → {t['end_date']}")
        st.caption(t['itinerary'][:150] + "...")
else:
    st.info("You haven’t planned any trips yet.")
