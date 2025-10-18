import streamlit as st
from supabase import create_client, Client
import os

# ================================
# 🔌 SUPABASE CONNECTION
# ================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="MonTravels Community", layout="wide")

st.title("👥 MonTravels – Social Travel Community")

# ================================
# 🔐 AUTH – SIGNUP & LOGIN
# ================================
if "user" not in st.session_state:
    st.session_state.user = None

auth_mode = st.radio("Choose Action", ["Login", "Sign Up"])

email = st.text_input("Email")
password = st.text_input("Password", type="password")

if auth_mode == "Sign Up":
    if st.button("Create Account"):
        try:
            response = supabase.auth.sign_up({"email": email, "password": password})
            st.success("✅ Account created! Check your email for verification.")
        except Exception as e:
            st.error(f"❌ Sign-up failed: {e}")

elif auth_mode == "Login":
    if st.button("Login"):
        try:
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            st.session_state.user = response.user
            st.success("✅ Logged in successfully!")
        except Exception as e:
            st.error(f"❌ Login failed: {e}")

# If logged in, show next steps
if st.session_state.user:
    st.sidebar.success(f"Logged in as: {st.session_state.user.email}")
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"user": None}))
    st.success("🎉 You’re logged in! Profile setup coming next...")
