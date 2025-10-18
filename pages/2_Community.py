import os
from datetime import datetime, timedelta
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

st.set_page_config(page_title="🌍 Community Reviews", page_icon="🧑‍🤝‍🧑", layout="wide")
st.title("🧑‍🤝‍🧑 MonTravels Community – Share & Explore Experiences")

# Ensure user is logged in
if "user" not in st.session_state:
    st.error("Please log in from the home page first.")
    st.stop()

user = st.session_state["user"]
current_user_id = user["id"]

# ================================
# ✍️ Submit a New Review
# ================================
st.subheader("✍️ Share Your Travel Experience")

with st.form("review_form"):
    city = st.text_input("🌆 City you visited *")
    review_text = st.text_area("📝 Write your review *")
    rating = st.slider("⭐ Rate your experience", 1, 5, 4)
    visit_date = st.date_input("📅 When did you visit?", datetime.today().date())

    submit_review = st.form_submit_button("📤 Post Review")

    if submit_review:
        if not city or not review_text:
            st.error("Please fill in all fields before submitting.")
        else:
            supabase.table("reviews").insert({
                "user_id": current_user_id,
                "city": city,
                "review": review_text,
                "rating": rating,
                "visit_date": str(visit_date),
                "date": datetime.utcnow().isoformat()
            }).execute()
            st.success("✅ Your review has been posted!")

st.markdown("---")

# ================================
# 🔎 Search & Filter Reviews
# ================================
st.subheader("🔍 Explore Reviews")
search_city = st.text_input("Search by city", "")

sort_by = st.selectbox(
    "Sort reviews by:",
    ["Most Recent", "Highest Rated", "Lowest Rated"]
)

# Build query
query = supabase.table("reviews").select("*")

if search_city:
    query = query.ilike("city", f"%{search_city}%")

if sort_by == "Most Recent":
    query = query.order("date", desc=True)
elif sort_by == "Highest Rated":
    query = query.order("rating", desc=True)
elif sort_by == "Lowest Rated":
    query = query.order("rating", desc=False)

reviews = query.limit(50).execute().data

if not reviews:
    st.info("No reviews found yet. Be the first to post one!")
else:
    for r in reviews:
        # Fetch user name
        user_info = supabase.table("users").select("email").eq("id", r["user_id"]).execute().data
        username = user_info[0]["email"] if user_info else "Anonymous"

        stars = "⭐" * r.get("rating", 0)
        st.markdown(f"""
        ### 🌆 {r['city']}  
        👤 **{username}** visited on **{r.get('visit_date', 'Unknown')}**  
        ⭐ **Rating:** {stars}  
        📝 {r['review']}  
        📅 Posted: {r['date'].split('T')[0]}
        """)
        st.markdown("---")

# ================================
# 📈 Weekly Highlights
# ================================
st.subheader("🔥 Trending This Week")

seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
recent_reviews = supabase.table("reviews").select("city").gte("date", seven_days_ago).execute().data

if recent_reviews:
    city_count = {}
    for r in recent_reviews:
        city_name = r["city"]
        city_count[city_name] = city_count.get(city_name, 0) + 1

    top_cities = sorted(city_count.items(), key=lambda x: x[1], reverse=True)[:5]
    for city_name, count in top_cities:
        st.write(f"🏙️ **{city_name}** – {count} reviews this week")
else:
    st.caption("No trending destinations yet this week.")
