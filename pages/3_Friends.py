import os
from datetime import datetime
import streamlit as st
from supabase import create_client
from dotenv import load_dotenv

# ================================
# 🌍 ENV & SUPABASE INIT
# ================================
load_dotenv()
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

st.set_page_config(page_title="🤝 Friends & Messaging", page_icon="💬", layout="wide")
st.title("🤝 Travel Friends & Messaging")

# Ensure user is logged in
if "user" not in st.session_state:
    st.error("Please log in from the home page first.")
    st.stop()

user = st.session_state["user"]
current_user_id = user["id"]

# ================================
# 🤝 Send Friend Requests
# ================================
st.header("👥 Find and Connect with Travelers")

all_users = supabase.table("users").select("id, email").neq("id", current_user_id).execute().data
for u in all_users:
    existing = supabase.table("friends").select("*").or_(
        f"(sender_id.eq.{current_user_id},receiver_id.eq.{u['id']})",
        f"(sender_id.eq.{u['id']},receiver_id.eq.{current_user_id})"
    ).execute().data

    if not existing:
        if st.button(f"➕ Connect with {u['email']}", key=f"connect_{u['id']}"):
            supabase.table("friends").insert({
                "sender_id": current_user_id,
                "receiver_id": u["id"],
                "status": "pending",
                "timestamp": datetime.utcnow().isoformat()
            }).execute()
            st.success("✅ Friend request sent!")
            st.experimental_rerun()
    else:
        status = existing[0]["status"]
        if status == "pending" and existing[0]["receiver_id"] == current_user_id:
            if st.button(f"✅ Accept request from {u['email']}", key=f"accept_{u['id']}"):
                supabase.table("friends").update({"status": "accepted"}).eq("id", existing[0]["id"]).execute()
                st.success("🤝 Connection accepted!")
                st.experimental_rerun()
        elif status == "pending":
            st.info("📩 Request pending...")
        elif status == "accepted":
            st.success("👥 Connected")

st.markdown("---")

# ================================
# 🌟 Recommended Friends (Similar Destinations)
# ================================
st.header("🌟 Recommended Friends (Similar Travel Interests)")

# Fetch all trips by current user
user_trips = supabase.table("trips").select("city").eq("creator_id", current_user_id).execute().data
visited_cities = {t["city"] for t in user_trips}

if visited_cities:
    # Find users with overlapping trips
    recommended = supabase.table("trips").select("creator_id, city").neq("creator_id", current_user_id).execute().data
    match_counts = {}
    for trip in recommended:
        if trip["city"] in visited_cities:
            match_counts[trip["creator_id"]] = match_counts.get(trip["creator_id"], 0) + 1

    sorted_recs = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for uid, count in sorted_recs:
        user_info = supabase.table("users").select("email").eq("id", uid).execute().data
        if user_info:
            st.write(f"💡 **{user_info[0]['email']}** – visited {count} similar destinations")
else:
    st.info("📍 Plan at least one trip to see friend recommendations.")

st.markdown("---")

# ================================
# 📜 Your Friends
# ================================
st.header("📜 Your Friends List")

friends = supabase.table("friends").select("*").or_(
    f"(sender_id.eq.{current_user_id},status.eq.accepted)",
    f"(receiver_id.eq.{current_user_id},status.eq.accepted)"
).execute().data

if friends:
    for f in friends:
        friend_id = f["sender_id"] if f["sender_id"] != current_user_id else f["receiver_id"]
        friend_info = supabase.table("users").select("email").eq("id", friend_id).execute().data
        st.write(f"👤 {friend_info[0]['email'] if friend_info else 'Unknown'}")
else:
    st.caption("You have no friends yet.")

st.markdown("---")

# ================================
# 💬 Messaging – Direct Chat
# ================================
st.header("💬 Chat with a Friend")

friend_ids = [f["sender_id"] if f["sender_id"] != current_user_id else f["receiver_id"] for f in friends]
friend_profiles = supabase.table("users").select("id,email").in_("id", friend_ids).execute().data
friend_map = {f["email"]: f["id"] for f in friend_profiles}

if friend_map:
    friend_choice = st.selectbox("Select a friend to chat with", list(friend_map.keys()))
    fid = friend_map[friend_choice]

    messages = supabase.table("messages").select("*").or_(
        f"(sender_id.eq.{current_user_id},receiver_id.eq.{fid})",
        f"(sender_id.eq.{fid},receiver_id.eq.{current_user_id})"
    ).order("timestamp", desc=False).execute().data

    st.markdown("### 💬 Conversation")
    for msg in messages:
        sender = "You" if msg["sender_id"] == current_user_id else friend_choice
        st.write(f"**{sender}:** {msg['content']}  \n🕒 {msg['timestamp'].split('T')[0]}")

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
else:
    st.info("No friends yet. Make connections to start chatting.")

st.markdown("---")

# ================================
# 👥 Group Chat – Shared Trips
# ================================
st.header("👥 Group Chats (Shared Trips)")

# Trips where user is a member
trip_memberships = supabase.table("trip_members").select("trip_id").eq("user_id", current_user_id).execute().data
trip_ids = [tm["trip_id"] for tm in trip_memberships]
trips = supabase.table("trips").select("id, city, start_date, end_date").in_("id", trip_ids).execute().data

trip_map = {f"{t['city']} ({t['start_date']} → {t['end_date']})": t["id"] for t in trips}
if trip_map:
    trip_choice = st.selectbox("Select a trip group", list(trip_map.keys()))
    trip_id = trip_map[trip_choice]

    group_messages = supabase.table("messages").select("*").eq("trip_id", trip_id).order("timestamp", desc=False).execute().data

    st.markdown("### 🧭 Group Conversation")
    for gm in group_messages:
        sender_info = supabase.table("users").select("email").eq("id", gm["sender_id"]).execute().data
        sender_name = sender_info[0]["email"] if sender_info else "Traveler"
        st.write(f"**{sender_name}:** {gm['content']}")

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
else:
    st.info("📍 Join or create a trip to start a group chat!")
