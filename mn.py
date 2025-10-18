import streamlit as st
from supabase import create_client
from datetime import datetime

# ================================
# 🔌 CONNECT TO SUPABASE
# ================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

st.set_page_config(page_title="MonTravels Community", layout="wide")
st.title("🌍 MonTravels – Travel Community Hub")

# ================================
# 🔐 AUTHENTICATION
# ================================
if "user" not in st.session_state:
    st.session_state.user = None

# Redirect to montravels_social.py login if not authenticated
if not st.session_state.user:
    st.warning("🔐 Please login or sign up first in `montravels_social.py`")
    st.stop()

user_email = st.session_state.user.email

# Fetch user row from `users` table
user_data = supabase.table("users").select("*").eq("email", user_email).execute()
if len(user_data.data) == 0:
    # Create a new blank profile if not exists
    supabase.table("users").insert({"email": user_email}).execute()
    user_data = supabase.table("users").select("*").eq("email", user_email).execute()

current_user = user_data.data[0]
current_user_id = current_user["id"]

# ================================
# 👤 PROFILE MANAGEMENT
# ================================
st.subheader("👤 My Profile")

with st.form("profile_form"):
    name = st.text_input("Full Name", current_user.get("name", ""))
    bio = st.text_area("Short Bio", current_user.get("bio", ""))
    country = st.text_input("Country", current_user.get("country", ""))
    interests = st.text_input("Travel Interests (comma separated)", ", ".join(current_user.get("interests", []) if current_user.get("interests") else []))

    if st.form_submit_button("💾 Save Profile"):
        interests_list = [i.strip() for i in interests.split(",") if i.strip()]
        supabase.table("users").update({
            "name": name,
            "bio": bio,
            "country": country,
            "interests": interests_list
        }).eq("id", current_user_id).execute()
        st.success("✅ Profile updated successfully!")
        st.experimental_rerun()

st.markdown("---")

# ================================
# 🗺️ TRIP REVIEWS
# ================================
st.subheader("🗺️ Share Your Travel Review")

with st.form("review_form"):
    city = st.text_input("City Visited")
    review_text = st.text_area("Your Review")
    if st.form_submit_button("📤 Post Review"):
        if city and review_text:
            supabase.table("reviews").insert({
                "user_id": current_user_id,
                "city": city,
                "review": review_text,
                "date": datetime.utcnow().isoformat()
            }).execute()
            st.success("✅ Review posted!")
        else:
            st.error("Please fill in both city and review.")

st.markdown("---")

# ================================
# 🌟 COMMUNITY FEED
# ================================
st.subheader("🌟 Traveler Reviews & Experiences")

reviews = supabase.table("reviews").select("*").order("date", desc=True).limit(50).execute().data
for rev in reviews:
    user_info = supabase.table("users").select("name, country").eq("id", rev["user_id"]).execute().data
    name = user_info[0]["name"] if user_info else "Unknown User"
    country = user_info[0]["country"] if user_info else ""
    st.markdown(f"""
    **✈️ {name}** from *{country}* visited **{rev['city']}**  
    🗣️ "{rev['review']}"  
    📅 *{rev['date'].split('T')[0]}*
    """)
    st.markdown("---")

# ================================
# 🤝 FRIEND REQUESTS
# ================================
st.subheader("🤝 Connect with Other Travelers")

all_users = supabase.table("users").select("id, name, country, bio").neq("id", current_user_id).execute().data

for u in all_users:
    st.markdown(f"**{u['name']}** ({u.get('country','Unknown')}) – {u.get('bio','No bio')}")

    # Check if already friends or request sent
    friendship = supabase.table("friends").select("*").or_(
        f"(sender_id.eq.{current_user_id},receiver_id.eq.{u['id']})",
        f"(sender_id.eq.{u['id']},receiver_id.eq.{current_user_id})"
    ).execute().data

    if not friendship:
        if st.button(f"➕ Send Friend Request to {u['name']}", key=f"fr_{u['id']}"):
            supabase.table("friends").insert({
                "sender_id": current_user_id,
                "receiver_id": u["id"],
                "status": "pending"
            }).execute()
            st.success("✅ Friend request sent!")
            st.experimental_rerun()
    else:
        status = friendship[0]["status"]
        if status == "pending" and friendship[0]["receiver_id"] == current_user_id:
            if st.button(f"✅ Accept Friend Request from {u['name']}", key=f"accept_{u['id']}"):
                supabase.table("friends").update({"status": "accepted"}).eq("id", friendship[0]["id"]).execute()
                st.success("🤝 Friend request accepted!")
                st.experimental_rerun()
        elif status == "pending":
            st.info("📤 Request pending...")
        elif status == "accepted":
            st.success("👥 Connected")

st.markdown("---")

# ================================
# 📬 MESSAGING SYSTEM
# ================================
st.subheader("📬 Private Messaging")

# Fetch all accepted friends
friends_list = supabase.table("friends").select("*").or_(
    f"(sender_id.eq.{current_user_id},status.eq.accepted)",
    f"(receiver_id.eq.{current_user_id},status.eq.accepted)"
).execute().data

friend_ids = [f["sender_id"] if f["sender_id"] != current_user_id else f["receiver_id"] for f in friends_list]
friend_profiles = supabase.table("users").select("id,name").in_("id", friend_ids).execute().data

friend_map = {fp["name"]: fp["id"] for fp in friend_profiles}
chat_friend = st.selectbox("💬 Choose a friend to chat with:", list(friend_map.keys()) if friend_map else ["No friends connected yet"])

if chat_friend != "No friends connected yet":
    friend_id = friend_map[chat_friend]

    # Display chat history
    messages = supabase.table("messages").select("*").or_(
        f"(sender_id.eq.{current_user_id},receiver_id.eq.{friend_id})",
        f"(sender_id.eq.{friend_id},receiver_id.eq.{current_user_id})"
    ).order("timestamp", desc=False).limit(50).execute().data

    st.markdown("### 💌 Conversation History")
    for msg in messages:
        sender_name = "You" if msg["sender_id"] == current_user_id else chat_friend
        st.markdown(f"**{sender_name}:** {msg['content']}  \n📅 *{msg['timestamp'].split('T')[0]}*")

    # New message box
    new_msg = st.text_input("✍️ Type your message:")
    if st.button("📨 Send"):
        if new_msg.strip():
            supabase.table("messages").insert({
                "sender_id": current_user_id,
                "receiver_id": friend_id,
                "content": new_msg,
                "timestamp": datetime.utcnow().isoformat()
            }).execute()
            st.experimental_rerun()
        else:
            st.error("Message cannot be empty!")
