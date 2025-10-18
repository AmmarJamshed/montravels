import os
from datetime import datetime
import streamlit as st
from supabase import create_client
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ================================
# 🌍 ENV & INIT
# ================================
load_dotenv()
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

st.set_page_config(page_title="🧠 Trip Discovery AI", page_icon="🤖", layout="wide")
st.title("🧠 Trip Discovery AI – Your Personalized Travel Guide")

if "user" not in st.session_state:
    st.error("Please log in from the home page first.")
    st.stop()

user = st.session_state["user"]
uid = user["id"]

# ================================
# 🤖 LLM Setup
# ================================
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.6,
    max_tokens=2800,
)

# ================================
# 🔍 Fetch user profile & history
# ================================
past_trips = supabase.table("trips").select("*").eq("creator_id", uid).execute().data
reviews = supabase.table("reviews").select("*").eq("user_id", uid).execute().data
friendships = supabase.table("friends").select("*").or_(
    f"(sender_id.eq.{uid},status.eq.accepted)", f"(receiver_id.eq.{uid},status.eq.accepted)"
).execute().data
friend_ids = [f["sender_id"] if f["sender_id"] != uid else f["receiver_id"] for f in friendships]

friend_trips = supabase.table("trips").select("*").in_("creator_id", friend_ids).gte(
    "start_date", datetime.utcnow().date().isoformat()
).execute().data

# ================================
# 🧠 User Input
# ================================
st.subheader("✍️ Tell us a bit about your next trip")
col1, col2 = st.columns(2)
with col1:
    budget = st.number_input("💰 Budget per day (USD)", min_value=50, max_value=2000, value=300)
    travel_month = st.selectbox("🗓️ Preferred Travel Month", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])
with col2:
    days = st.slider("🧭 Duration (days)", 3, 21, 7)
    style = st.multiselect("🎯 Travel Style", ["Nature", "Adventure", "Food", "Luxury", "History", "Nightlife"], default=["Nature","Food"])

st.markdown("---")

# ================================
# 🤝 Friend-based suggestions
# ================================
st.subheader("👥 Where Your Friends Are Headed")
if friend_trips:
    for ft in friend_trips:
        friend_info = supabase.table("users").select("email").eq("id", ft["creator_id"]).execute().data
        fname = friend_info[0]["email"] if friend_info else "Traveler"
        st.markdown(f"🧑‍🤝‍🧑 **{fname}** is going to **{ft['city']}** ({ft['start_date']} → {ft['end_date']})")
else:
    st.caption("No upcoming friend trips — be the first to plan!")

st.markdown("---")

# ================================
# 🔮 AI Recommendations
# ================================
if st.button("✨ Generate Personalized Recommendations"):
    st.info("⏳ Analyzing your travel profile, friends’ plans, and global trends...")

    user_history = ", ".join([t["city"] for t in past_trips]) if past_trips else "No previous trips"
    friends_destinations = ", ".join([t["city"] for t in friend_trips]) if friend_trips else "None"

    prompt = f"""
    You are a world-class travel recommendation AI.

    Analyze the following details:
    - User's previous trips: {user_history}
    - User's reviews & preferences: {[r['review_text'][:100] for r in reviews]}
    - Travel style: {', '.join(style)}
    - Preferred month: {travel_month}
    - Budget: {budget} USD/day
    - Duration: {days} days
    - Friends' planned destinations: {friends_destinations}

    Generate a **balanced list of 5 full trip recommendations**.  
    For each destination, include:

    1. 📍 **Destination name & country**
    2. 💡 Why it’s a great fit (personalized reasoning)
    3. 🧭 Top 3 activities or experiences
    4. 🗓️ Best time to visit (mention if it matches user's month)
    5. 🛡️ Safety rating (1-10 with 🟢🟡🔴 indicator)
    6. ✈️ Whether any friends are going there
    7. 🧳 Estimated daily cost

    Recommendations must be ranked from most to least suitable.
    """

    resp = llm.invoke(prompt)
    st.success("✅ Here are your AI-powered trip recommendations:")
    st.markdown(resp.content)

    st.markdown("---")
    st.info("💡 Tip: Click 'Plan This Trip' on any suggestion to instantly create a custom itinerary in the Trip Planner.")

# ================================
# 🔁 Quick Shortcuts
# ================================
st.markdown("---")
st.subheader("⚡ Quick Actions")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("✈️ Go to Trip Planner"):
        st.switch_page("pages/1_Trip_Planner.py")
with c2:
    if st.button("🧑‍🤝‍🧑 Invite a Friend"):
        st.switch_page("pages/3_Friends.py")
with c3:
    if st.button("🏠 Back to Dashboard"):
        st.switch_page("pages/4_Dashboard.py")
