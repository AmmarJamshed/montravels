import os
from datetime import date, timedelta, datetime
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI

# ================================
# PAGE CONFIG & THEME
# ================================
st.set_page_config(page_title="MonTravels – Smart Travel Planner", page_icon="🧭", layout="wide")

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
# INITIALIZE LLM (Groq)
# ================================
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.4,
    max_tokens=2800,
)

# ================================
# GEOCODING + OSM HELPERS
# ================================
def geocode_city(city: str):
    """Get latitude/longitude of a city using Nominatim (OpenStreetMap)."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city, "format": "json", "limit": 1, "accept-language": "en"}
    r = requests.get(url, params=params, headers={"User-Agent": "montravels-app"})
    data = r.json()
    if not data:
        return None, None
    return float(data[0]["lat"]), float(data[0]["lon"])

def fetch_osm_places(city: str, place_type="hotel", limit=5):
    """Search for places (hotels, residences, motels) near a city using Overpass API."""
    lat, lon = geocode_city(city)
    if not lat or not lon:
        return []
    radius = 5000  # 5 km
    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lon})[tourism={place_type}];
      way(around:{radius},{lat},{lon})[tourism={place_type}];
      relation(around:{radius},{lat},{lon})[tourism={place_type}];
    );
    out center {limit};
    """
    resp = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=30)
    if not resp.ok:
        return []
    data = resp.json()
    places = []
    for el in data.get("elements", [])[:limit]:
        tags = el.get("tags", {})
        name = tags.get("name", f"Unnamed {place_type.title()}")
        addr = tags.get("addr:full") or tags.get("addr:street") or ""
        maps_url = f"https://www.openstreetmap.org/{el['type']}/{el['id']}"
        places.append({"name": name, "address": addr, "rating": tags.get("stars", "N/A"), "link": maps_url})
    return places

# ================================
# ITINERARY GENERATOR
# ================================
def generate_itinerary(city, area, start, end, interests, budget, adults):
    """Generate a detailed day-by-day itinerary with LLM based on preferences."""
    days = max((end - start).days, 1)
    prompt = f"""
    You are an expert travel planner. Create a {days}-day travel itinerary for {city}, {area or ''}.
    - Include Morning, Afternoon, and Evening activities
    - Include budget-friendly meal suggestions
    - Include daily tips
    - Focus on interests: {', '.join(interests)}
    - Budget: ${budget} per day for {adults} adults
    - Label days as Day 1, Day 2, ...
    """
    resp = llm.invoke(prompt)
    return resp.content

# ================================
# NEWS SCRAPER & SAFETY ANALYSIS
# ================================
def scrape_news(city: str, max_articles=6):
    """Scrape latest English-language news articles about the city from DuckDuckGo."""
    query = f"{city} travel OR safety OR protests OR incidents site:bbc.com OR site:cnn.com OR site:reuters.com"
    url = f"https://duckduckgo.com/html/?q={query}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "lxml")

    articles = []
    for link in soup.select("a.result__a")[:max_articles]:
        articles.append({"title": link.get_text(strip=True), "url": link.get("href")})
    return articles

def summarize_safety_from_news(city: str, articles: list):
    """Summarize city safety conditions from recent headlines using LLM."""
    if not articles:
        return "No recent news found. Likely stable and safe."
    headlines = "\n".join([f"- {a['title']}" for a in articles])
    prompt = f"""
    You are a global travel safety analyst. Based on these recent headlines about {city}, 
    summarize the current safety situation in 4-5 sentences. Mention protests, disasters, 
    crime spikes, or health issues if relevant:
    {headlines}
    """
    return llm.invoke(prompt).content

# ================================
# GOVERNMENT TRAVEL ADVISORY SCRAPER
# ================================
def fetch_us_advisory(city: str):
    """Scrape the U.S. State Dept. travel advisory page for city/country info."""
    search_url = f"https://travel.state.gov/content/travel/en/traveladvisories/traveladvisories.html?search={city}"
    resp = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "lxml")
    text = " ".join([p.get_text(" ", strip=True) for p in soup.select("div.tsg-rwd-advisories p")])
    return text or "No U.S. advisory available."

def fetch_uk_advisory(city: str):
    """Scrape UK Foreign Office travel advisory for the destination."""
    search_url = f"https://www.gov.uk/foreign-travel-advice/{city.lower().replace(' ', '-')}"
    resp = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})
    if not resp.ok:
        return "No UK advisory found."
    soup = BeautifulSoup(resp.text, "lxml")
    paragraphs = soup.select("div.govspeak p")
    return " ".join([p.get_text(" ", strip=True) for p in paragraphs]) or "No UK advisory available."

def summarize_advisories(city: str):
    """Merge and summarize US + UK advisories into a single digest."""
    us = fetch_us_advisory(city)
    uk = fetch_uk_advisory(city)
    combined = f"U.S. Advisory:\n{us}\n\nUK Advisory:\n{uk}"
    prompt = f"""
    Summarize the combined U.S. and UK travel advisories for {city} in 4-6 sentences. 
    Include main risks (crime, terrorism, natural disasters, etc.) and provide a 
    recommended safety level (Low, Moderate, High).
    {combined}
    """
    return llm.invoke(prompt).content

# ================================
# SAFETY SCORE CALCULATOR
# ================================
def calculate_safety_score(news_text: str, advisory_text: str):
    """
    Use LLM to assign a safety score (1-10) based on both news and official advisories.
    """
    prompt = f"""
    Based on the following two reports about a city's safety:
    NEWS:\n{news_text}\n\nADVISORY:\n{advisory_text}\n
    Assign a safety score from 1 to 10 (1 = extremely risky, 10 = very safe).
    Provide only the number.
    """
    score_text = llm.invoke(prompt).content.strip()
    try:
        score = int("".join(ch for ch in score_text if ch.isdigit()))
        return max(1, min(score, 10))
    except:
        return 5  # fallback default
# ================================
# TRAVEL AGENCY SCRAPER
# ================================
def scrape_travel_agencies(city: str, max_results=10):
    """Scrape travel agencies for the destination using DuckDuckGo search."""
    query = f"travel agencies in {city}"
    search_url = f"https://duckduckgo.com/html/?q={query}"
    resp = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})
    if not resp.ok:
        return []
    soup = BeautifulSoup(resp.text, "lxml")
    results = []
    for a in soup.select("a.result__a")[:max_results]:
        name = a.get_text(strip=True)
        url = a.get("href")
        results.append({"name": name, "url": url})
    return results

# ================================
# STREAMLIT SIDEBAR INPUTS
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
    lodging_choice = st.selectbox("Lodging Type", ["All", "Hotels", "Residences", "Motels"])
    go = st.button("✨ Build Plan")

# ================================
# MAIN ACTION LOGIC
# ================================
if go:
    if not city:
        st.error("Please enter a destination.")
        st.stop()

    with st.spinner("🧠 Building your personalized itinerary..."):
        itinerary = generate_itinerary(city, area, start_date, end_date, interests, budget, adults)

    # ----------------------------
    # 2-COLUMN LAYOUT
    # ----------------------------
    col1, col2 = st.columns([2, 1])

    # --- LEFT COLUMN: ITINERARY ---
    with col1:
        st.subheader("🗓️ Your Travel Itinerary")
        st.markdown(f"**📅 Travel Period:** {start_date} → {end_date}")
        st.write(itinerary)

    # --- RIGHT COLUMN: Lodging ---
    with col2:
        st.subheader("🏨 Lodging Options")
        if lodging_choice in ["All", "Hotels"]:
            st.markdown("### 🏨 Hotels")
            hotels = fetch_osm_places(city, "hotel")
            if not hotels:
                st.caption("No hotels found.")
            else:
                for h in hotels:
                    st.markdown(f"**[{h['name']}]({h['link']})**  \n📍 {h['address']}  \n⭐ Rating: {h.get('rating','N/A')}")

        if lodging_choice in ["All", "Residences"]:
            st.markdown("### 🏡 Residences & Apartments")
            residences = fetch_osm_places(city, "guest_house")
            if not residences:
                st.caption("No residences found.")
            else:
                for r in residences:
                    st.markdown(f"**[{r['name']}]({r['link']})**  \n📍 {r['address']}  \n⭐ Rating: {r.get('rating','N/A')}")

        if lodging_choice in ["All", "Motels"]:
            st.markdown("### 🛏️ Motels")
            motels = fetch_osm_places(city, "motel")
            if not motels:
                st.caption("No motels found.")
            else:
                for m in motels:
                    st.markdown(f"**[{m['name']}]({m['link']})**  \n📍 {m['address']}  \n⭐ Rating: {m.get('rating','N/A')}")

    # ================================
    # 🛡️ SAFETY & NEWS SECTION
    # ================================
    st.subheader("🛡️ Safety & Travel Insights")
    with st.spinner("🔎 Fetching latest news and advisories..."):
        # 1. News scraping and summary
        news_articles = scrape_news(city)
        news_summary = summarize_safety_from_news(city, news_articles)

        # 2. Official advisories
        advisory_summary = summarize_advisories(city)

        # 3. Safety score calculation
        safety_score = calculate_safety_score(news_summary, advisory_summary)
        if safety_score >= 8:
            color, status = "🟢", "Safe"
        elif 5 <= safety_score < 8:
            color, status = "🟡", "Caution"
        else:
            color, status = "🔴", "Risky"

    # Display results
    st.markdown(f"### 📊 Safety Score: **{safety_score}/10** {color} – *{status}*")
    st.caption(f"🕒 Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    st.markdown("### 📰 News-Based Safety Summary")
    st.info(news_summary)

    if news_articles:
        st.markdown("**Latest Headlines:**")
        for art in news_articles:
            st.markdown(f"- [{art['title']}]({art['url']})")
    else:
        st.caption("No recent news found for this destination.")

    st.markdown("### 🏛️ Government Travel Advisory Summary")
    st.warning(advisory_summary)

    # ================================
    # ✈️ TRAVEL AGENCIES
    # ================================
    st.subheader("🌍 Local Travel Agencies")
    with st.spinner("🔍 Searching local agencies..."):
        agencies = scrape_travel_agencies(city)

    if agencies:
        for ag in agencies:
            st.markdown(f"🔗 **[{ag['name']}]({ag['url']})**")
    else:
        st.caption("No travel agencies found online for this location.")

    # ================================
    # 📌 Tips Section
    # ================================
    st.info("""
    💡 **Tips:**  
    - Always verify advisory updates closer to your travel date.  
    - Register with your local embassy if traveling to high-risk areas.  
    - Combine both news trends and official advisories before making final decisions.
    """)
