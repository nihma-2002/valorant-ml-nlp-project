import streamlit as st
import pandas as pd
import joblib

# ---------------------------------
# Load trained model
# ---------------------------------
model = joblib.load("models/role_model.pkl")

# ---------------------------------
# Role → Agents mapping
# ---------------------------------
role_to_agents = {
    "Duelist": ["Jett", "Reyna", "Raze", "Phoenix", "Neon", "Yoru", "Iso"],
    "Controller": ["Omen", "Brimstone", "Viper", "Astra", "Harbor", "Clove"],
    "Sentinel": ["Sage", "Cypher", "Killjoy", "Chamber", "Deadlock"],
    "Initiator": ["Sova", "Breach", "Skye", "Fade", "Gekko", "KAY/O"]
}

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(
    page_title="Valorant Agent Recommendation",
    layout="centered"
)

st.title("🎮 Valorant Agent Recommendation System")
st.write(
    "Adjust player performance metrics below to predict the most suitable **role** "
    "and receive **agent recommendations**."
)

# ---------------------------------
# User Inputs
# ---------------------------------
kills = st.slider("Kills per Match", 0.0, 45.0, 18.0)
deaths = st.slider("Deaths per Match", 0.0, 30.0, 14.0)
assists = st.slider("Assists per Match", 0.0, 20.0, 6.0)
acs = st.slider("ACS", 50.0, 350.0, 200.0)
utility = st.slider("Utility Usage", 0.0, 1.0, 0.5)
clutch = st.slider("Clutch Success Rate", 0.0, 1.0, 0.3)
win_rate = st.slider("Win Rate", 0.0, 1.0, 0.5)

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("🎯 Recommend Agents"):
    # Base input from user
    input_df = pd.DataFrame([{
        "kills_per_match": kills,
        "deaths_per_match": deaths,
        "assists_per_match": assists,
        "acs": acs,
        "utility_usage": utility,
        "clutch_success_rate": clutch,
        "win_rate": win_rate,
        "headshot_pct": 0.25  # realistic default
    }])

    # Add missing playstyle one-hot columns
    playstyle_columns = [
        "playstyle_Aim-Focused Duelists",
        "playstyle_Low-Impact / Learning Players",
        "playstyle_Utility & Clutch Specialists"
    ]

    for col in playstyle_columns:
        input_df[col] = 0

    # Ensure exact feature order used during training
    input_df = input_df[model.feature_names_in_]

    # Predict role
    predicted_role = model.predict(input_df)[0]
    recommended_agents = role_to_agents[predicted_role][:3]

    # Display results
    st.success(f"🧩 Predicted Role: **{predicted_role}**")
    st.subheader("Recommended Agents")
    for agent in recommended_agents:
        st.write(f"• {agent}")
