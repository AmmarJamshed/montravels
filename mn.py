import os
import re
from datetime import date, timedelta, datetime
import requests
import streamlit as st
from bs4 import BeautifulSoup
from supabase import create_client
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# ================================
# 📁 ENVIRONMENT VARIABLES
# ================================
load_dotenv()
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================================
# 🌍 PAGE STYLING
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
    </style>
""", unsafe_allow_html=True)

st.title("🧭 MonTravels – Plan Your Journey Solo or With Friends")
st.caption("✨ Plan your trips, invite friends, explore reviews, and connect with fellow travelers — all in one place!")

# ================================
# 🔑 LOGIN / SIGNUP
# ================================
st.sidebar.subheader("Login or Sign Up")
mode = st.sidebar.radio("", ["Login", "Sign Up"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button(mode):
    if not email or not password:
        st.sidebar.error("Please enter email and password.")
    else:
        profile = supabase.table("users").select("*").eq("email", email).execute()

        if mode == "Sign Up":
            if profile.data:
                st.sidebar.error("User already exists. Please login.")
            else:
                # ✅ Insert safely with default values for nullable fields
                supabase.table("users").insert({
                    "email": email,
                    "name": "",
                    "bio": "",
                    "country": ""
                }).execute()
                st.sidebar.success("✅ Account created. You can log in now.")

        elif mode == "Login":
            if not profile.data:
                st.sidebar.error("No account found. Please sign up first.")
            else:
                st.session_state["user"] = profile.data[0]
                st.sidebar.success("✅ Logged in!")

# Stop here if not logged in
if "user" not in st.session_state:
    st.stop()

current_user = st.session_state["user"]
current_user_id = current_user["id"]
st.success(f"Welcome back, {current_user.get('name', 'Traveler')}! 🌍")

# ================================
# 🤖 INITIALIZE LLM
# ================================
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.4,
    max_tokens=2800,
)

# ================================
# 📍 HELPER FUNCTIONS
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
    (node(around:5000,{lat},{lon})[tourism={place_type}];);
    out center {limit};
    """
    resp = requests.post("https://overpass-api.de/api/interpreter", data={"data": query})
    if not resp.ok: return []
    data = resp.json()
    return [{"name": el["tags"].get("name", "Unknown"), "link": f"https://www.openstreetmap.org/{el['type']}/{el['id']}"} for el in data.get("elements", [])]

def generate_itinerary(city, start, end, interests, budget, adults):
    days = max((end - start).days, 1)
    prompt = f"Create a {days}-day itinerary for {city} focused on {', '.join(interests)} for {adults} adults on ${budget}/day."
    return llm.invoke(prompt).content

# ================================
# 📍 TRIP PLANNER
# ================================
with st.sidebar:
    city = st.text_input("Destination*").strip()
    start_date = st.date_input("Start", date.today() + timedelta(days=7))
    end_date = st.date_input("End", date.today() + timedelta(days=10))
    adults = st.number_input("Adults", 1, 10, 2)
    budget = st.number_input("Budget ($/day)", 10, 1000, 100)
    interests = st.multiselect("Interests", ["food", "history", "nature", "nightlife"], default=["food", "history"])
    go = st.button("✨ Build Trip Plan")

if go and city:
    st.subheader("🗺️ Itinerary")
    itinerary = generate_itinerary(city, start_date, end_date, interests, budget, adults)
    st.write(itinerary)

    supabase.table("trips").insert({
        "creator_id": current_user_id,
        "city": city,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "itinerary": itinerary
    }).execute()
    st.success("✅ Trip saved to your profile!")

# ================================
# ✍️ COMMUNITY REVIEWS
# ================================
st.header("✍️ Traveler Reviews")
with st.form("review_form"):
    city_reviewed = st.text_input("City")
    review_text = st.text_area("Your review")
    if st.form_submit_button("Post Review"):
        supabase.table("reviews").insert({
            "user_id": current_user_id,
            "city": city_reviewed,
            "review": review_text,
            "date": datetime.utcnow().isoformat()
        }).execute()
        st.success("✅ Review posted!")

reviews = supabase.table("reviews").select("*").order("date", desc=True).limit(30).execute().data
for r in reviews:
    st.markdown(f"**{r['city']}** — {r['review']}")

# ================================
# 🤝 FRIENDS & CONNECTIONS
# ================================
st.header("🤝 Connect with Travelers")
others = supabase.table("users").select("id,email").neq("id", current_user_id).execute().data
for u in others:
    st.write(f"📧 {u['email']}")
    if st.button(f"Connect with {u['email']}", key=u['id']):
        supabase.table("friends").insert({
            "sender_id": current_user_id,
            "receiver_id": u["id"],
            "status": "pending"
        }).execute()
        st.success("✅ Request sent!")

# ================================
# 💬 Messaging (Simple Version)
# ================================
st.header("💬 Chat with Friends")
friends = supabase.table("friends").select("*").or_(
    f"(sender_id.eq.{current_user_id},status.eq.accepted)",
    f"(receiver_id.eq.{current_user_id},status.eq.accepted)"
).execute().data

friend_ids = [f["sender_id"] if f["sender_id"] != current_user_id else f["receiver_id"] for f in friends]
friend_profiles = supabase.table("users").select("id,email").in_("id", friend_ids).execute().data

friend_map = {f["email"]: f["id"] for f in friend_profiles}
friend_choice = st.selectbox("Select friend", list(friend_map.keys()) if friend_map else ["No friends yet"])

if friend_choice != "No friends yet":
    fid = friend_map[friend_choice]
    chat = supabase.table("messages").select("*").or_(
        f"(sender_id.eq.{current_user_id},receiver_id.eq.{fid})",
        f"(sender_id.eq.{fid},receiver_id.eq.{current_user_id})"
    ).order("timestamp", desc=False).execute().data

    for m in chat:
        sender = "You" if m["sender_id"] == current_user_id else friend_choice
        st.write(f"**{sender}:** {m['content']}")

    new_msg = st.text_input("Type a message:")
    if st.button("Send"):
        supabase.table("messages").insert({
            "sender_id": current_user_id,
            "receiver_id": fid,
            "content": new_msg,
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
        st.experimental_rerun()
