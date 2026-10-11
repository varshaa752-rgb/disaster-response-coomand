import streamlit as st
import folium
from streamlit_folium import st_folium
from ultralytics import YOLO
import tempfile
import os
import time

# Page Configuration
st.set_page_config(layout="wide", page_title="Mission 6: Disaster-Response-Command")

st.title("🚁 Mission 6: Disaster Response & Situational Awareness Command Center")
st.markdown("### AI-Powered Aerial Scouting, Damage Detection, GIS Mapping & Predictive Dispatch")

# Sidebar Controls
st.sidebar.header("Control Panel")
mode = st.sidebar.selectbox("Choose Mode", ["Live Map & Sectors", "AI Damage Detection (Video/Image)"])

# ----------------- MODE 1: MAP & GIS -----------------
if mode == "Live Map & Sectors":
    st.subheader("🗺️ Simulated Autonomous Drone Fleet Surveillance Grid")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=15)
        
        folium.Polygon(
            locations=[
                [12.975, 77.590], [12.975, 77.600],
                [12.965, 77.600], [12.965, 77.590]
            ],
            color="red",
            weight=3,
            fill=True,
            fill_opacity=0.4,
            popup="Priority Zone A: Severe Impact / Stranded Civilians"
        ).add_to(m)
        
        st_folium(m, width=800, height=480)
        
    with col2:
        st.metric(label="Monitored Sector Area", value="4.2 km²")
        st.metric(label="System Risk Level", value="CRITICAL (92%)", delta="+5% increase")
        st.metric(label="Active Drone Units", value="3 En Route")
        st.error("🚨 ALERT: Sector A requires immediate medical and rescue deployment based on predictive GIS telemetry.")

# ----------------- MODE 2: AI COMPUTER VISION & DISPATCH -----------------
elif mode == "AI Damage Detection (Video/Image)":
    st.subheader("🧠 Autonomous Aerial Feed Analysis & Emergency Dispatcher")
    st.write("Upload drone or aerial imagery of a disaster location to run real-time hazard detection and calculate evacuation priority.")

    uploaded_file = st.file_uploader("Upload Aerial Image", type=["jpg", "jpeg", "png"])
    
    model = YOLO("yolov8n.pt")

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(file_bytes)
        temp_file.close()
        
        with st.spinner("Analyzing aerial terrain and detecting survivors/hazards..."):
            results = model(temp_file.name, conf=0.15)
            res_plotted = results[0].plot()
            time.sleep(0.5)
            
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.image(res_plotted, channels="BGR", caption="AI Vision Output - Detected Civilians & Obstacles")
        
        with col_b:
            detected_count = len(results[0].boxes)
            st.metric(label="Total Targets / Anomalies Found", value=detected_count)
            
            risk_score = min(98, 40 + (detected_count * 5))
            st.metric(label="Calculated Priority Score", value=f"{risk_score}%", delta="High Urgency")
            
            if detected_count > 0:
                st.warning("⚠️ Civilians/Hazards detected in scan area.")
                
                if st.button("🚀 Dispatch Emergency Response Team"):
                    with st.spinner("Transmitting encrypted GPS coordinates to nearest response unit..."):
                        time.sleep(1.5)
                    st.success("✅ Dispatch successful! Unit #4 deployed to coordinates [12.9716, 77.5946].")
            else:
                st.info("ℹ️ Clear area. No immediate human targets detected.")
        
        os.unlink(temp_file.name)