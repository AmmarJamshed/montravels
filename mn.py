import os
import re
from datetime import date, timedelta, datetime
import requests
import streamlit as st
from bs4 import BeautifulSoup
from supabase import create_client
from langchain_openai import ChatOpenAI

# ================================
# 🌍 PAGE CONFIG & STYLING
# ================================
st.set_page_config(page_title="MonTravels – Plan & Explore", page_icon="🧭", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #F5F7FA; font-family: 'Trebuchet MS', sans-serif; }
    h1 { color: #FFCC00; text-shadow: 2px 2px 0px #3B4CCA; }
    h2, h3 { color: #3B4CCA; }
</style>
""", unsafe_allow_html=True)
st.title("🧭 MonTravels – Plan Your Journey Solo or With Friends")

st.markdown("""
> ✈️ **Welcome to MonTravels** — Plan your trip, invite friends, and explore together.  
Whether you're traveling **solo** or building unforgettable memories with **friends**, this is your travel hub.
""")

# ================================
# 🔌 SUPABASE CONNECTION
# ================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================================
# 🔐 AUTHENTICATION
# ================================
if "user" not in st.session_state:
    st.session_state.user = None

auth_mode = st.radio("Login or Sign Up", ["Login", "Sign Up"])
email = st.text_input("Email")
password = st.text_input("Password", type="password")

if auth_mode == "Sign Up":
    if st.button("Create Account"):
        try:
            supabase.auth.sign_up({"email": email, "password": password})
            st.success("✅ Account created! Please verify your email and log in.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

elif auth_mode == "Login":
    if st.button("Login"):
        try:
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            st.session_state.user = response.user
            st.success("✅ Logged in!")
        except Exception as e:
            st.error(f"❌ Login failed: {e}")

if not st.session_state.user:
    st.stop()

user_email = st.session_state.user.email

# Get or create profile
profile = supabase.table("users").select("*").eq("email", user_email).execute()
if not profile.data:
    supabase.table("users").insert({"email": user_email}).execute()
    profile = supabase.table("users").select("*").eq("email", user_email).execute()
current_user = profile.data[0]
current_user_id = current_user["id"]

# ================================
# 👤 PROFILE MANAGEMENT
# ================================
st.subheader("👤 My Profile")
with st.form("profile_form"):
    name = st.text_input("Name", current_user.get("name", ""))
    bio = st.text_area("Bio", current_user.get("bio", ""))
    country = st.text_input("Country", current_user.get("country", ""))
    if st.form_submit_button("💾 Save Profile"):
        supabase.table("users").update({
            "name": name,
            "bio": bio,
            "country": country
        }).eq("id", current_user_id).execute()
        st.success("✅ Profile updated!")
        st.experimental_rerun()

# ================================
# 🤖 LLM INITIALIZATION
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
    params = {"q": city, "format": "json", "limit": 1}
    r = requests.get(url, params=params, headers={"User-Agent": "montravels"})
    data = r.json()
    return (float(data[0]["lat"]), float(data[0]["lon"])) if data else (None, None)

def fetch_osm_places(city: str, place_type="hotel", limit=5):
    lat, lon = geocode_city(city)
    if not lat or not lon: return []
    q = f"""
    [out:json][timeout:25];
    (node(around:5000,{lat},{lon})[tourism={place_type}];
     way(around:5000,{lat},{lon})[tourism={place_type}];
     relation(around:5000,{lat},{lon})[tourism={place_type}];);
    out center {limit};
    """
    resp = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=30)
    return [{"name": el.get("tags", {}).get("name", "Unnamed"),
             "address": el.get("tags", {}).get("addr:street", ""),
             "link": f"https://www.openstreetmap.org/{el['type']}/{el['id']}"} 
            for el in resp.json().get("elements", [])[:limit]] if resp.ok else []

def generate_itinerary(city, area, start, end, interests, budget, adults):
    days = max((end - start).days, 1)
    prompt = f"""
    You are a travel expert. Create a {days}-day plan for {city}, {area or ''}:
    - Daily plan (morning/afternoon/evening)
    - Budget: ${budget}/day for {adults} adults
    - Interests: {', '.join(interests)}
    """
    return llm.invoke(prompt).content

# ================================
# 🧭 TRAVEL PLANNER TAB
# ================================
st.header("🧭 Travel Planner")
with st.sidebar:
    city = st.text_input("Destination*")
    area = st.text_input("Area (optional)")
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("Start", date.today() + timedelta(days=7))
    with c2: end_date = st.date_input("End", date.today() + timedelta(days=10))
    adults = st.number_input("Adults", 1, 10, 2)
    budget = st.number_input("Budget ($/day)", 10, 1000, 100)
    interests = st.multiselect("Interests", ["food","history","museums","nature","nightlife"], default=["food","history"])
    plan_trip = st.button("✨ Build Plan")

if plan_trip and city:
    with st.spinner("🧠 Generating your itinerary..."):
        itinerary = generate_itinerary(city, area, start_date, end_date, interests, budget, adults)

    st.subheader("📜 Itinerary")
    st.write(itinerary)

    # Save trip
    trip_insert = supabase.table("trips").insert({
        "creator_id": current_user_id,
        "city": city,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "itinerary": itinerary
    }).execute()
    trip_id = trip_insert.data[0]["id"]
    st.success("✅ Trip saved!")

    # ================================
    # ✈️ INVITE FRIENDS SECTION
    # ================================
    st.subheader("🤝 Invite Friends to Join This Trip")

    # Fetch all friends
    friends_data = supabase.table("friends").select("*").or_(
        f"(sender_id.eq.{current_user_id},status.eq.accepted)",
        f"(receiver_id.eq.{current_user_id},status.eq.accepted)"
    ).execute().data

    friend_ids = [f["sender_id"] if f["sender_id"] != current_user_id else f["receiver_id"] for f in friends_data]
    all_friends = supabase.table("users").select("id,name").in_("id", friend_ids).execute().data

    # ✅ Auto-suggest: users who reviewed this city
    past_visitors = supabase.table("reviews").select("user_id").eq("city", city).execute().data
    suggested_ids = list({p["user_id"] for p in past_visitors if p["user_id"] in friend_ids})
    suggested_friends = [f for f in all_friends if f["id"] in suggested_ids]

    if suggested_friends:
        st.markdown("👀 Suggested friends who visited this city:")
        for sf in suggested_friends:
            if st.button(f"Invite {sf['name']}"):
                supabase.table("trip_members").insert({
                    "trip_id": trip_id,
                    "user_id": sf["id"],
                    "invited_by": current_user_id
                }).execute()
                st.success(f"✅ Invited {sf['name']}")

    st.markdown("📋 Or manually invite:")
    for friend in all_friends:
        if st.button(f"Invite {friend['name']}", key=f"invite_{friend['id']}"):
            supabase.table("trip_members").insert({
                "trip_id": trip_id,
                "user_id": friend["id"],
                "invited_by": current_user_id
            }).execute()
            st.success(f"✅ Invited {friend['name']}")

    # ================================
    # 👀 PEOPLE WHO VISITED THIS CITY
    # ================================
    st.subheader("🌍 People Who Visited This Destination")

    visitors = supabase.table("reviews").select("user_id").eq("city", city).execute().data
    unique_visitors = list({v["user_id"] for v in visitors})
    visitor_profiles = supabase.table("users").select("name,country,bio").in_("id", unique_visitors).execute().data

    if visitor_profiles:
        for vp in visitor_profiles:
            st.markdown(f"**{vp['name']}** – {vp.get('country','Unknown')}  \n🧭 {vp.get('bio','No bio provided.')}")
    else:
        st.info("No one has posted about this destination yet — be the first!")
# ================================
# 📍 COMMUNITY TAB – Reviews & Discovery
# ================================
st.header("👥 Travel Community")

st.subheader("✍️ Share Your Travel Experience")
with st.form("review_form"):
    city_reviewed = st.text_input("City Visited")
    review_text = st.text_area("Your Review")
    if st.form_submit_button("📤 Post Review"):
        if city_reviewed and review_text:
            supabase.table("reviews").insert({
                "user_id": current_user_id,
                "city": city_reviewed,
                "review": review_text,
                "date": datetime.utcnow().isoformat()
            }).execute()
            st.success("✅ Review posted!")
        else:
            st.error("Please fill in both city and review.")

st.subheader("🌍 Recent Traveler Reviews")
reviews = supabase.table("reviews").select("*").order("date", desc=True).limit(30).execute().data

for rev in reviews:
    u = supabase.table("users").select("name,country").eq("id", rev["user_id"]).execute().data
    uname = u[0]["name"] if u else "Traveler"
    country = u[0]["country"] if u else "Unknown"
    st.markdown(f"**✈️ {uname}** from *{country}* visited **{rev['city']}**")
    st.write(f"🗣️ {rev['review']}")
    st.caption(f"📅 {rev['date'].split('T')[0]}")
    st.markdown("---")

# ================================
# 🤝 FRIENDS TAB – Connect & Accept Requests
# ================================
st.header("🤝 Friends & Connections")

# Show all other users
all_users = supabase.table("users").select("id,name,country,bio").neq("id", current_user_id).execute().data
for u in all_users:
    st.markdown(f"**{u['name']}** – {u.get('country','Unknown')}  \n🧭 {u.get('bio','No bio available')}")

    # Check friendship status
    existing = supabase.table("friends").select("*").or_(
        f"(sender_id.eq.{current_user_id},receiver_id.eq.{u['id']})",
        f"(sender_id.eq.{u['id']},receiver_id.eq.{current_user_id})"
    ).execute().data

    if not existing:
        if st.button(f"➕ Connect with {u['name']}", key=f"connect_{u['id']}"):
            supabase.table("friends").insert({
                "sender_id": current_user_id,
                "receiver_id": u["id"],
                "status": "pending"
            }).execute()
            st.success("✅ Request sent!")
            st.experimental_rerun()
    else:
        status = existing[0]["status"]
        if status == "pending" and existing[0]["receiver_id"] == current_user_id:
            if st.button(f"✅ Accept request from {u['name']}", key=f"accept_{u['id']}"):
                supabase.table("friends").update({"status": "accepted"}).eq("id", existing[0]["id"]).execute()
                st.success("🤝 Connection accepted!")
                st.experimental_rerun()
        elif status == "pending":
            st.info("📤 Request Pending...")
        elif status == "accepted":
            st.success("👥 Connected")

st.markdown("---")

# ================================
# 📬 MESSAGING TAB – Private & Group Chat
# ================================
st.header("📬 Messaging")

# 1️⃣ Private chat with friends
st.subheader("💬 Chat with a Friend")

friends = supabase.table("friends").select("*").or_(
    f"(sender_id.eq.{current_user_id},status.eq.accepted)",
    f"(receiver_id.eq.{current_user_id},status.eq.accepted)"
).execute().data

friend_ids = [f["sender_id"] if f["sender_id"] != current_user_id else f["receiver_id"] for f in friends]
friend_profiles = supabase.table("users").select("id,name").in_("id", friend_ids).execute().data

friend_names = {f["name"]: f["id"] for f in friend_profiles}
friend_choice = st.selectbox("Choose a friend:", list(friend_names.keys()) if friend_names else ["No friends"])

if friend_choice != "No friends":
    fid = friend_names[friend_choice]

    messages = supabase.table("messages").select("*").or_(
        f"(sender_id.eq.{current_user_id},receiver_id.eq.{fid})",
        f"(sender_id.eq.{fid},receiver_id.eq.{current_user_id})"
    ).order("timestamp", desc=False).execute().data

    st.markdown("### 📜 Conversation")
    for msg in messages:
        sender = "You" if msg["sender_id"] == current_user_id else friend_choice
        st.write(f"**{sender}:** {msg['content']}  \n📅 {msg['timestamp'].split('T')[0]}")

    new_msg = st.text_input("✍️ Type a message:")
    if st.button("📨 Send Message"):
        if new_msg.strip():
            supabase.table("messages").insert({
                "sender_id": current_user_id,
                "receiver_id": fid,
                "content": new_msg,
                "timestamp": datetime.utcnow().isoformat()
            }).execute()
            st.experimental_rerun()

# 2️⃣ Group chat – shared trip messages
st.subheader("👥 Group Trip Chats")

# Get trips where current user is a member
trip_memberships = supabase.table("trip_members").select("trip_id").eq("user_id", current_user_id).execute().data
trip_ids = [tm["trip_id"] for tm in trip_memberships]
trips = supabase.table("trips").select("id,city,start_date,end_date").in_("id", trip_ids).execute().data

trip_map = {f"{t['city']} ({t['start_date']} → {t['end_date']})": t["id"] for t in trips}
trip_choice = st.selectbox("Select a trip group:", list(trip_map.keys()) if trip_map else ["No group trips"])

if trip_choice != "No group trips":
    trip_id = trip_map[trip_choice]

    group_messages = supabase.table("messages").select("*").eq("trip_id", trip_id).order("timestamp", desc=False).execute().data

    st.markdown("### 🧭 Group Chat")
    for gm in group_messages:
        sender_info = supabase.table("users").select("name").eq("id", gm["sender_id"]).execute().data
        sender_name = sender_info[0]["name"] if sender_info else "Traveler"
        st.write(f"**{sender_name}:** {gm['content']}  \n📅 {gm['timestamp'].split('T')[0]}")

    group_msg = st.text_input("✍️ Send a message to group:")
    if st.button("📨 Send to Group"):
        if group_msg.strip():
            supabase.table("messages").insert({
                "sender_id": current_user_id,
                "trip_id": trip_id,
                "content": group_msg,
                "timestamp": datetime.utcnow().isoformat()
            }).execute()
            st.experimental_rerun()
