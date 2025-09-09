#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from datetime import date, timedelta
import urllib.parse
import requests
import streamlit as st

# =========================
# RapidAPI (Booking)
# =========================
RAPIDAPI_HOST = os.getenv("RAPIDAPI_BOOKING_HOST", "booking-com15.p.rapidapi.com")
RAPIDAPI_KEY  = os.getenv("RAPIDAPI_KEY", "027ec4cc59mshd687f4217321cf9p188043jsna60603f095a6")

def rapidapi_headers() -> dict:
    return {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": RAPIDAPI_HOST}

# =========================
# UI helpers
# =========================
def combined_query(city: str, area: str) -> str:
    return (f"{city} {area}".strip() if area else city).strip()

def booking_deeplink(city_or_area: str, checkin: date, checkout: date, adults: int = 2) -> str:
    q = urllib.parse.quote(city_or_area)
    return (
        "https://www.booking.com/searchresults.html"
        f"?ss={q}"
        f"&checkin={checkin:%Y-%m-%d}"
        f"&checkout={checkout:%Y-%m-%d}"
        f"&group_adults={adults}&no_rooms=1&group_children=0"
    )

# -------------------------
# Geocoding (OSM Nominatim)
# -------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_osm(query: str):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": query, "format": "json", "limit": 1}
        headers = {"User-Agent": "MonTravels/1.0 (contact@montravels.app)"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        js = r.json() or []
        if js:
            return {"lat": float(js[0]["lat"]), "lon": float(js[0]["lon"]), "name": js[0]["display_name"]}
    except Exception:
        pass
    return None

# =========================
# Destination lookup (robust to vendor schema)
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def booking_dest_id(query: str, debug: bool = False) -> str | None:
    candidate_endpoints = [
        "/api/v1/hotels/searchDestination",
        "/api/v1/hotels/locations",
        "/api/v1/meta/getLocation",
        "/v1/hotels/locations",
    ]
    last_err = None
    for path in candidate_endpoints:
        url = f"https://{RAPIDAPI_HOST}{path}"
        try:
            params = {"query": query, "name": query, "locale": "en-us"}
            r = requests.get(url, headers=rapidapi_headers(), params=params, timeout=20)
            r.raise_for_status()
            js = r.json()

            if debug:
                st.write(f"🔍 Endpoint tried: {path}")
                st.write(js)

            if isinstance(js, list):
                for item in js:
                    if isinstance(item, dict):
                        if item.get("dest_type", "").lower() in {"city", "district"} and item.get("dest_id"):
                            return str(item["dest_id"])
                for item in js:
                    if isinstance(item, dict):
                        for k in ("dest_id", "id", "regionId", "region_id"):
                            if k in item:
                                return str(item[k])
                continue

            if isinstance(js, dict):
                for key in ("dest_id", "id", "regionId", "region_id"):
                    if key in js:
                        return str(js[key])
                for arr_key in ("data", "results", "locations", "result"):
                    arr = js.get(arr_key)
                    if isinstance(arr, list) and arr:
                        for item in arr:
                            if isinstance(item, dict):
                                if item.get("dest_type", "").lower() in {"city", "district"} and item.get("dest_id"):
                                    return str(item["dest_id"])
                        for item in arr:
                            if isinstance(item, dict):
                                for k in ("dest_id", "id", "regionId", "region_id"):
                                    if k in item:
                                        return str(item[k])
                continue

            last_err = f"Unexpected response type from {path}: {type(js).__name__}"
        except Exception as e:
            last_err = f"{path} -> {e}"
            continue

    if debug and last_err:
        st.warning(f"Destination lookup issues: {last_err}")
    return None

# =========================
# Hotel search (multi-strategy)
# =========================
def extract_hotel_results(js: dict) -> list:
    for key in ("result", "results", "data", "hotels", "properties"):
        if isinstance(js.get(key), list) and js[key]:
            return js[key]
    for k in ("payload", "response"):
        inner = js.get(k)
        if isinstance(inner, dict):
            for key in ("result", "results", "data", "hotels", "properties"):
                if isinstance(inner.get(key), list) and inner[key]:
                    return inner[key]
    return []

@st.cache_data(ttl=600, show_spinner=False)
def booking_search_hotels_multi(dest_id: str | None, checkin: date, checkout: date,
                                adults: int = 2, rooms: int = 1,
                                coords: dict | None = None, debug: bool = False):
    strategies = []

    if dest_id:
        strategies.append(("searchHotels (dest_id)", "/api/v1/hotels/searchHotels", {
            "dest_id": dest_id, "search_type": "CITY",
            "checkin_date": checkin.strftime("%Y-%m-%d"),
            "checkout_date": checkout.strftime("%Y-%m-%d"),
            "adults_number": adults, "room_number": rooms,
            "order_by": "popularity", "currency_code": "USD",
            "locale": "en-us", "units": "metric", "page_number": 1
        }))
        strategies.append(("search (dest_id)", "/api/v1/hotels/search", {
            "dest_id": dest_id, "dest_type": "CITY",
            "checkin_date": checkin.strftime("%Y-%m-%d"),
            "checkout_date": checkout.strftime("%Y-%m-%d"),
            "adults_number": adults, "room_number": rooms,
            "currency": "USD", "locale": "en-us",
            "order_by": "popularity", "page_number": 1
        }))

    if coords:
        lat, lon = coords["lat"], coords["lon"]
        strategies.append(("searchByCoordinates", "/api/v1/hotels/searchByCoordinates", {
            "latitude": lat, "longitude": lon,
            "checkin_date": checkin.strftime("%Y-%m-%d"),
            "checkout_date": checkout.strftime("%Y-%m-%d"),
            "adults_number": adults, "room_number": rooms,
            "radius": 5000, "currency_code": "USD",
            "locale": "en-us", "units": "metric"
        }))
        strategies.append(("search-by-coordinates (alt)", "/v1/hotels/search-by-coordinates", {
            "latitude": lat, "longitude": lon,
            "checkin_date": checkin.strftime("%Y-%m-%d"),
            "checkout_date": checkout.strftime("%Y-%m-%d"),
            "adults_number": adults, "room_number": rooms,
            "radius": 5, "currency": "USD", "locale": "en-us"
        }))

    last_err = None
    for name, path, params in strategies:
        try:
            url = f"https://{RAPIDAPI_HOST}{path}"
            r = requests.get(url, headers=rapidapi_headers(), params=params, timeout=25)
            r.raise_for_status()
            js = r.json()
            if debug:
                st.write(f"🔍 Tried: {name}")
                st.write(js)
            if extract_hotel_results(js):
                return {"data": js, "strategy": name}
        except Exception as e:
            last_err = f"{name} -> {e}"
            continue

    if debug and last_err:
        st.warning(f"No strategy returned results. Last error: {last_err}")
    return {"data": {}, "strategy": None}

# =========================
# Offline Itinerary (no APIs)
# =========================
def build_offline_itinerary(city: str, start_date: date, end_date: date, area: str, interests: list[str], budget: str):
    days = max((end_date - start_date).days, 1)
    loc = f"{city} ({area})" if area else city
    interest_str = ", ".join(interests) if interests else "sightseeing"
    lines = [f"## {loc} — {days}-Day Itinerary (Budget: {budget})",
             f"_Interests: {interest_str}_"]
    daily = [
        ("Morning", "iconic landmark & neighborhood walk"),
        ("Afternoon", "museum/park + café stop"),
        ("Evening", "viewpoint or riverfront + local dinner")
    ]
    for i in range(days):
        lines.append(f"\n### Day {i+1}")
        for part, plan in daily:
            lines.append(f"- **{part}**: {plan}")
        lines.append("- **Cost**: $$")
    lines += [
        "\n**Tips**",
        "- Get a day transport pass; keep small cash for markets.",
        "- Check opening hours; prebook popular spots.",
        "- Respect local etiquette; watch the weather."
    ]
    return "\n".join(lines)

# =========================
# UI
# =========================
st.set_page_config(page_title="MonTravels — Hotels & Itinerary", page_icon="🧭", layout="wide")
st.title("🧭 MonTravels")

with st.sidebar:
    st.header("Trip Inputs")
    city = st.text_input("Destination city*", placeholder="e.g., London, Karachi, Dubai").strip()
    area = st.text_input("Area / neighborhood (optional)", placeholder="e.g., Clifton, Soho, Marina").strip()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=date.today() + timedelta(days=14))
    with col2:
        end_date = st.date_input("End date", value=date.today() + timedelta(days=18))
    adults = st.number_input("Adults", min_value=1, max_value=10, value=2, step=1)
    rooms = st.number_input("Rooms", min_value=1, max_value=5, value=1, step=1)

    st.markdown("**Interests**")
    interests = st.multiselect(
        "",
        ["history", "museums", "food", "nature", "nightlife", "architecture", "shopping", "family"],
        default=["food", "history"]
    )
    budget = st.radio("Budget", options=["tight", "moderate", "luxury"], index=1, horizontal=True)

    show_debug = st.toggle("Show debug JSON (dev)", value=False)
    go = st.button("✨ Build Plan")

st.caption(f"Using RapidAPI host: `{RAPIDAPI_HOST}`")

if go:
    if not city:
        st.error("Please enter a destination city.")
        st.stop()
    if end_date <= start_date:
        st.error("End date must be after Start date.")
        st.stop()

    query_text = combined_query(city, area)

    # 1) Itinerary (offline)
    st.subheader("🗓️ Your Itinerary")
    st.markdown(build_offline_itinerary(city, start_date, end_date, area, interests, budget))

    # 2) Resolve dest id + geocode
    st.subheader("🔎 Searching destination")
    with st.spinner(f"Resolving '{query_text}'…"):
        dest_id = booking_dest_id(query_text, debug=show_debug)
    if dest_id:
        st.success(f"Found destination id: {dest_id}")
    else:
        st.warning("Could not resolve destination id; will try coordinates-based search.")

    coords = geocode_osm(query_text)
    if coords:
        st.caption(f"📍 {coords['name']}  ({coords['lat']:.4f}, {coords['lon']:.4f})")
    else:
        st.caption("📍 Could not geolocate—will attempt id-based search only.")

    # 3) Hotels (multi-strategy)
    st.subheader("🏨 Hotels")
    with st.spinner("Fetching hotels…"):
        res = booking_search_hotels_multi(dest_id, start_date, end_date, adults=adults, rooms=rooms, coords=coords, debug=show_debug)

    data = res.get("data", {})
    strategy = res.get("strategy")
    hotels = extract_hotel_results(data)
    if strategy:
        st.caption(f"Strategy that returned data: **{strategy}**")

    if not hotels:
        st.info("No hotels returned. Try different dates or nearby areas.")
        st.link_button("Open full results on Booking.com", booking_deeplink(query_text, start_date, end_date, adults))
    else:
        for h in hotels[:12]:
            with st.container(border=True):
                name = (h.get("hotel_name") or h.get("name") or "Hotel").strip()
                st.markdown(f"**{name}**")

                meta = []
                price_bd = h.get("price_breakdown") or {}
                price = price_bd.get("gross_price") or h.get("min_total_price") or h.get("price")
                currency = price_bd.get("currency") or h.get("currency") or "USD"
                if price: meta.append(f"Price: {price} {currency}")
                rating = h.get("review_score") or h.get("rating")
                reviews = h.get("review_nr") or h.get("reviews") or h.get("number_of_reviews")
                if rating: meta.append(f"⭐ {rating} ({reviews or 0} reviews)")
                for addr_key in ("address_trans", "address", "city", "neighborhood"):
                    if h.get(addr_key):
                        meta.append(str(h[addr_key])); break
                st.caption(" • ".join(meta))

                img = h.get("max_photo_url") or h.get("main_photo_url") or h.get("imageUrl")
                if img: st.image(img, use_column_width=True)

                deep = h.get("url") or h.get("hotel_url")
                if deep:
                    st.link_button("View on Booking.com", deep)
                else:
                    st.link_button("View on Booking.com", booking_deeplink(query_text, start_date, end_date, adults))

        st.link_button("🔗 Open full results on Booking.com", booking_deeplink(query_text, start_date, end_date, adults))

else:
    st.info("Enter a city (optionally an area), choose dates & preferences, then click **Build Plan**.")

st.divider()
st.caption("Itinerary generated locally (no AI). Hotel data via Booking.com through RapidAPI; prices/availability may change.")


# In[ ]:




