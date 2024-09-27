import streamlit as st
import requests

API_URL = "http://localhost:5000/items"

def fetch_items():
    response = requests.get(API_URL)
    if response.status_code == 200:
        return response.json()
    st.error(f"Error fetching items: {response.status_code}")
    return []


st.title("Custom Application with BackEnd")

st.subheader("Fetch Items")
if st.button("Fetch All Items"):
    items = fetch_items()
    st.write(items)

st.subheader("Create Item")


st.subheader("Update Item")


st.subheader("Delete Item")
