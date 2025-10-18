import os
from datetime import datetime
import streamlit as st
from supabase import create_client
from dotenv import load_dotenv

# ================================
# 🌍 ENV & INIT
# ================================
load_dotenv()
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

st.set_page_config(page_title="📊 Dashboard – MonTravels", page_icon="📊", layout="wide")
st.title("📊 MonTravels – Your Travel Hub")

if "user" not in st.session_state:
    st.error("Please log in from the home page first.")
    st.stop()

user = st.session_state["user"]
uid = user["id"]

# ================================
# 📆 Overview Section
# ================================
st.header("📅 Your Trips Overview")

upcoming_trips = supabase.table("trips").select("*").eq("creator_id", uid).gte("start_date", datetime.utcnow().date().isoformat()).execute().data
past_trips = supabase.table("trips").select("*").eq("creator_id", uid).lt("start_date", datetime.utcnow().date().isoformat()).execute().data

c1, c2, c3 = st.columns(3)
c1.metric("🗺️ Upcoming Trips", len(upcoming_trips))
c2.metric("🌍 Trips Completed", len(past_trips))
c3.metric("👥 Friends", len(supabase.table("friends").select("*").or_(
    f"(sender_id.eq.{uid},status.eq.accepted)", f"(receiver_id.eq.{uid},status.eq.accepted)"
).execute().data))

st.markdown("---")

# ================================
# 🧭 Upcoming Trips
# ================================
st.subheader("✈️ Your Upcoming Trips")
if upcoming_trips:
    for t in upcoming_trips:
        st.markdown(f"**📍 {t['city']}** – {t['start_date']} → {t['end_date']}")
else:
    st.caption("No upcoming trips yet.")

st.markdown("---")

# ================================
# 🔥 Combined Leaderboard
# ================================
st.header("🏆 Top Travelers Leaderboard")

users = supabase.table("users").select("id, email").execute().data
leaderboard = []

for u in users:
    uid_ = u["id"]
    trips = supabase.table("trips").select("city").eq("creator_id", uid_).execute().data
    reviews = supabase.table("reviews").select("*").eq("user_id", uid_).execute().data

    total_trips = len(trips)
    unique_cities = len(set([t["city"] for t in trips]))
    total_reviews = len(reviews)

    score = (total_trips * 2) + (unique_cities * 3) + (total_reviews * 1.5)
    leaderboard.append({
        "email": u["email"],
        "trips": total_trips,
        "cities": unique_cities,
        "reviews": total_reviews,
        "score": score
    })

leaderboard = sorted(leaderboard, key=lambda x: x["score"], reverse=True)[:5]

st.markdown("### 🥇 Top 5 Most Active Travelers")
for i, lb in enumerate(leaderboard, start=1):
    st.markdown(f"**#{i} – {lb['email']}**  \n🌍 Trips: {lb['trips']} | 🏙️ Cities: {lb['cities']} | 📝 Reviews: {lb['reviews']} | 🔥 Score: {lb['score']}")

st.markdown("---")

# ================================
# 🔔 Friend Activity Feed
# ================================
st.header("🪩 Friends’ Recent Activity")

friendships = supabase.table("friends").select("*").or_(
    f"(sender_id.eq.{uid},status.eq.accepted)", f"(receiver_id.eq.{uid},status.eq.accepted)"
).execute().data

friend_ids = [f["sender_id"] if f["sender_id"] != uid else f["receiver_id"] for f in friendships]

activities = []

for fid in friend_ids:
    fuser = supabase.table("users").select("email").eq("id", fid).execute().data
    fname = fuser[0]["email"] if fuser else "Traveler"

    trips = supabase.table("trips").select("*").eq("creator_id", fid).order("start_date", desc=True).limit(2).execute().data
    reviews = supabase.table("reviews").select("*").eq("user_id", fid).order("created_at", desc=True).limit(2).execute().data

    for t in trips:
        activities.append(f"✈️ **{fname}** planned a trip to **{t['city']}** from {t['start_date']} → {t['end_date']}")
    for r in reviews:
        activities.append(f"⭐ **{fname}** reviewed **{r['city']}** – \"{r['review_text'][:80]}...\"")

if activities:
    for act in activities[:10]:
        st.markdown(act)
else:
    st.caption("No recent activity from your friends.")

st.markdown("---")

# ================================
# 🧭 Quick Actions
# ================================
st.header("⚡ Quick Actions")

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("✍️ Create a New Trip"):
        st.switch_page("pages/1_Trip_Planner.py")
with c2:
    if st.button("🧑‍🤝‍🧑 Find New Friends"):
        st.switch_page("pages/3_Friends.py")
with c3:
    if st.button("🧠 Discover New Destinations"):
        st.switch_page("pages/5_Trip_Discovery_AI.py")

st.markdown("---")

# ================================
# 📊 Analytics Summary
# ================================
st.header("📊 Your Analytics")

col1, col2 = st.columns(2)
with col1:
    total_reviews = supabase.table("reviews").select("*").eq("user_id", uid).execute().data
    st.metric("📝 Total Reviews Written", len(total_reviews))

with col2:
    total_destinations = len(set([t["city"] for t in past_trips + upcoming_trips]))
    st.metric("🌍 Total Destinations Visited", total_destinations)
