import os
import streamlit as st
from supabase import create_client
from dotenv import load_dotenv

# ================================
# 🌍 ENV SETUP
# ================================
load_dotenv()
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # use service key for admin ops
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

st.set_page_config(page_title="MonTravels – Explore the World", page_icon="🧭", layout="wide")
st.title("🧭 MonTravels – Explore, Connect & Discover")

# ================================
# 🔐 LOGIN / SIGNUP
# ================================
st.sidebar.header("Login / Sign Up")
mode = st.sidebar.radio("", ["Login", "Sign Up"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button(mode):
    if not email or not password:
        st.sidebar.error("Please enter both email and password.")
    else:
        profile = supabase.table("users").select("*").eq("email", email).execute()

        if mode == "Sign Up":
            if profile.data:
                st.sidebar.error("User already exists.")
            else:
                supabase.table("users").insert({
                    "email": email,
                    "name": "",
                    "bio": "",
                    "country": ""
                }).execute()
                st.sidebar.success("✅ Account created. Please log in.")

        elif mode == "Login":
            if not profile.data:
                st.sidebar.error("No account found. Sign up first.")
            else:
                st.session_state["user"] = profile.data[0]
                st.sidebar.success("✅ Logged in!")

# Stop app if not logged in
if "user" not in st.session_state:
    st.stop()

user = st.session_state["user"]
st.success(f"Welcome back, {user.get('email')} 👋")

st.markdown("""
### 🌍 Welcome to MonTravels  
Use the sidebar to navigate through:
- ✈️ **Trip Planner** – Plan AI-powered itineraries  
- 🧑‍🤝‍🧑 **Community** – Share & explore travel reviews  
- 🤝 **Friends** – Build your travel circle & chat  
- 🏠 **Dashboard** – See trends & friend activities  
- 🧠 **Trip Discovery AI** – Get personalized destination ideas
""")
