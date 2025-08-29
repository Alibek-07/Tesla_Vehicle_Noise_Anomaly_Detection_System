"""
Tesla NVH Interactive Dashboard
==============================

Professional Tesla-styled monitoring interface with real-time visualization
Demonstrates full-stack development and production monitoring capabilities

Features:
- Real-time vehicle diagnostics display
- Tesla-branded UI with dark theme
- Live performance metrics
- Fleet health monitoring
- Interactive anomaly detection results
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import torch
import os
from pathlib import Path

# Tesla brand colors
TESLA_RED = '#E31937'
TESLA_BLACK = '#171A20'
TESLA_WHITE = '#FFFFFF'
TESLA_GRAY = '#5C5E62'
TESLA_BLUE = '#1976D2'

def load_tesla_model():
    """Load Tesla NVH model for live inference"""
    try:
        possible_paths = [
            "/app/data/models/best_model.pth",
            "/app/data/models/model_torchscript.pt",
            "/app/tesla_models/tesla_nvh_model.pt",
            "tesla_models/tesla_nvh_model.pt", 
            "../tesla_models/tesla_nvh_model.pt",
            "tesla_nvh_model.pt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    if path.endswith('.pth'):
                        # Handle regular PyTorch model
                        return None, f"{path} (PyTorch model - demo mode)"
                    else:
                        # Handle TorchScript model
                        model = torch.jit.load(path)
                        model.eval()
                        return model, path
                except:
                    continue
        return None, "No model found - using simulated mode"
    except Exception as e:
        return None, f"Model loading error: {e}"

def generate_live_vehicle_data():
    """Generate realistic Tesla vehicle telemetry data"""
    
    vehicles = [
        {'id': 'T3-001', 'model': 'Model 3', 'location': 'Fremont, CA', 'mileage': 15234},
        {'id': 'TS-047', 'model': 'Model S', 'location': 'Austin, TX', 'mileage': 23891}, 
        {'id': 'TX-012', 'model': 'Model X', 'location': 'Gigafactory Berlin', 'mileage': 8756},
        {'id': 'CT-003', 'model': 'Cybertruck', 'location': 'Giga Texas', 'mileage': 1234},
        {'id': 'T3-089', 'model': 'Model 3', 'location': 'Shanghai, China', 'mileage': 31567},
        {'id': 'T3-156', 'model': 'Model 3', 'location': 'Berlin, Germany', 'mileage': 19876},
        {'id': 'TS-092', 'model': 'Model S', 'location': 'London, UK', 'mileage': 45123},
        {'id': 'TX-045', 'model': 'Model X', 'location': 'Tokyo, Japan', 'mileage': 12987}
    ]
    
    current_time = datetime.now()
    
    vehicle_data = []
    for vehicle in vehicles:
        # Simulate realistic vehicle conditions with more sophisticated logic
        base_anomaly_prob = np.random.beta(2, 8)  # Bias toward normal
        
        # Age-based deterioration
        mileage_factor = min(vehicle['mileage'] / 100000, 0.3)  # Higher mileage = more issues
        base_anomaly_prob += mileage_factor
        
        # Model-specific patterns
        if vehicle['model'] == 'Cybertruck':
            base_anomaly_prob += 0.1  # Newer model, some early issues
        elif vehicle['model'] == 'Model S' and vehicle['mileage'] > 30000:
            base_anomaly_prob += 0.05  # Luxury model with age
        
        # Add some vehicles with specific issues
        if vehicle['id'] in ['TS-047', 'CT-003']:
            base_anomaly_prob = np.random.beta(6, 4)  # Higher chance of issues
        
        # Clamp probability
        base_anomaly_prob = min(max(base_anomaly_prob, 0), 1)
        
        # Generate comprehensive telemetry
        engine_noise_db = np.random.uniform(45, 65)
        brake_temp = np.random.uniform(80, 200)
        tire_pressure = np.random.uniform(32, 38)
        battery_temp = np.random.uniform(25, 45)
        
        # Advanced metrics
        motor_rpm = np.random.uniform(1000, 4000)
        efficiency_kwh = np.random.uniform(3.8, 5.2)  # kWh/100km
        regeneration_rate = np.random.uniform(0.15, 0.45)
        
        # Determine status with more nuanced logic
        if base_anomaly_prob > 0.75:
            status = "CRITICAL"
            color = TESLA_RED
            issue_type = np.random.choice(['Brake System', 'Motor Bearing', 'Suspension'])
        elif base_anomaly_prob > 0.5:
            status = "WARNING" 
            color = "#FFA500"
            issue_type = np.random.choice(['Minor Noise', 'Efficiency Drop', 'Tire Wear'])
        else:
            status = "NORMAL"
            color = "#00AA00"
            issue_type = "All Systems Normal"
        
        # Simulate real diagnostic data
        diagnostic_codes = []
        if status == "CRITICAL":
            diagnostic_codes = [f"NVH-{np.random.randint(100,999)}", f"DIAG-{np.random.randint(1000,9999)}"]
        elif status == "WARNING":
            diagnostic_codes = [f"INFO-{np.random.randint(100,999)}"]
        
        vehicle_data.append({
            'vehicle_id': vehicle['id'],
            'model': vehicle['model'],
            'location': vehicle['location'],
            'mileage': vehicle['mileage'],
            'anomaly_probability': round(base_anomaly_prob, 3),
            'status': status,
            'status_color': color,
            'issue_type': issue_type,
            'diagnostic_codes': diagnostic_codes,
            # Core telemetry
            'engine_noise_db': round(engine_noise_db, 1),
            'brake_temp': round(brake_temp, 1),
            'tire_pressure': round(tire_pressure, 1),
            'battery_temp': round(battery_temp, 1),
            # Advanced metrics
            'motor_rpm': round(motor_rpm, 0),
            'efficiency_kwh': round(efficiency_kwh, 1),
            'regeneration_rate': round(regeneration_rate, 2),
            # System metrics
            'last_update': current_time,
            'inference_time_ms': round(np.random.uniform(15, 35), 1),
            'signal_strength': np.random.choice(['Excellent', 'Good', 'Fair']),
            'gps_accuracy': round(np.random.uniform(2, 8), 1)
        })
    
    return vehicle_data

def create_tesla_header():
    """Create Tesla-styled header"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #171A20 0%, #E31937 100%); 
                padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0; 
                   font-family: 'Arial', sans-serif; font-weight: bold;">
            🚗 TESLA NVH DIAGNOSTIC DASHBOARD
        </h1>
        <p style="color: #CCCCCC; text-align: center; margin: 0.5rem 0 0 0;">
            Real-time Vehicle Health Monitoring • Neural Anomaly Detection • Fleet Management
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_metrics_cards(vehicle_data):
    """Create Tesla-styled metrics cards"""
    
    total_vehicles = len(vehicle_data)
    critical_count = len([v for v in vehicle_data if v['status'] == 'CRITICAL'])
    warning_count = len([v for v in vehicle_data if v['status'] == 'WARNING'])
    normal_count = len([v for v in vehicle_data if v['status'] == 'NORMAL'])
    avg_inference = np.mean([v['inference_time_ms'] for v in vehicle_data])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div style="background: {TESLA_BLACK}; padding: 1rem; border-radius: 8px; text-align: center;">
            <h2 style="color: {TESLA_WHITE}; margin: 0;">{total_vehicles}</h2>
            <p style="color: {TESLA_GRAY}; margin: 0;">Active Vehicles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: {TESLA_RED}; padding: 1rem; border-radius: 8px; text-align: center;">
            <h2 style="color: white; margin: 0;">{critical_count}</h2>
            <p style="color: #FFCCCC; margin: 0;">Critical Issues</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: #FFA500; padding: 1rem; border-radius: 8px; text-align: center;">
            <h2 style="color: white; margin: 0;">{warning_count}</h2>
            <p style="color: #FFEECC; margin: 0;">Warnings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="background: #00AA00; padding: 1rem; border-radius: 8px; text-align: center;">
            <h2 style="color: white; margin: 0;">{normal_count}</h2>
            <p style="color: #CCFFCC; margin: 0;">Normal Status</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div style="background: {TESLA_BLUE}; padding: 1rem; border-radius: 8px; text-align: center;">
            <h2 style="color: white; margin: 0;">{avg_inference:.1f}ms</h2>
            <p style="color: #CCE7FF; margin: 0;">Avg Response</p>
        </div>
        """, unsafe_allow_html=True)

def create_fleet_map(vehicle_data):
    """Create Tesla fleet geographic visualization"""
    
    # Sample coordinates for Tesla locations
    locations = {
        'Fremont, CA': {'lat': 37.5485, 'lon': -121.9886},
        'Austin, TX': {'lat': 30.2672, 'lon': -97.7431},
        'Gigafactory Berlin': {'lat': 52.3979, 'lon': 13.6297},
        'Giga Texas': {'lat': 30.2241, 'lon': -97.6225},
        'Shanghai, China': {'lat': 31.2304, 'lon': 121.4737},
        'Berlin, Germany': {'lat': 52.5200, 'lon': 13.4050},
        'London, UK': {'lat': 51.5074, 'lon': -0.1278},
        'Tokyo, Japan': {'lat': 35.6762, 'lon': 139.6503}
    }
    
    map_data = []
    for vehicle in vehicle_data:
        if vehicle['location'] in locations:
            coords = locations[vehicle['location']]
            map_data.append({
                'lat': coords['lat'],
                'lon': coords['lon'],
                'vehicle_id': vehicle['vehicle_id'],
                'model': vehicle['model'],
                'status': vehicle['status'],
                'anomaly_prob': vehicle['anomaly_probability'],
                'size': 20 if vehicle['status'] == 'CRITICAL' else 15 if vehicle['status'] == 'WARNING' else 10
            })
    
    df_map = pd.DataFrame(map_data)
    
    # Fixed: Use scatter_map instead of scatter_mapbox
    fig = px.scatter_map(
        df_map, lat="lat", lon="lon",
        color="status",
        size="size",
        hover_name="vehicle_id",
        hover_data={"model": True, "anomaly_prob": ":.3f"},
        color_discrete_map={
            'NORMAL': '#00AA00',
            'WARNING': '#FFA500', 
            'CRITICAL': TESLA_RED
        },
        map_style="carto-darkmatter",
        zoom=1,
        title="🌍 Tesla Global Fleet Status"
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_size=16
    )
    
    return fig

def create_realtime_charts(vehicle_data):
    """Create real-time monitoring charts"""
    
    # Anomaly probability distribution
    fig_dist = px.histogram(
        pd.DataFrame(vehicle_data),
        x='anomaly_probability',
        nbins=20,
        title="🔍 Anomaly Probability Distribution",
        color_discrete_sequence=[TESLA_BLUE]
    )
    fig_dist.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    # Performance metrics by model
    df_perf = pd.DataFrame(vehicle_data)
    fig_model = px.box(
        df_perf,
        x='model',
        y='inference_time_ms',
        title="⚡ Inference Performance by Tesla Model",
        color='model',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig_model.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    # Vehicle telemetry heatmap
    telemetry_data = []
    for v in vehicle_data:
        telemetry_data.append({
            'Vehicle': v['vehicle_id'],
            'Anomaly Score': v['anomaly_probability'],
            'Engine Noise': v['engine_noise_db'] / 100,  # Normalize
            'Brake Temp': v['brake_temp'] / 200,
            'Battery Temp': v['battery_temp'] / 50,
            'Response Time': v['inference_time_ms'] / 40
        })
    
    df_heatmap = pd.DataFrame(telemetry_data)
    df_heatmap_transposed = df_heatmap.set_index('Vehicle').T
    
    fig_heatmap = px.imshow(
        df_heatmap_transposed,
        title="🌡️ Vehicle Telemetry Heatmap",
        color_continuous_scale='RdYlBu_r'
    )
    fig_heatmap.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    return fig_dist, fig_model, fig_heatmap

def create_vehicle_details_table(vehicle_data):
    """Create detailed vehicle status table"""
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(vehicle_data)
    
    # Format the display
    display_df = df[['vehicle_id', 'model', 'location', 'status', 'anomaly_probability', 'inference_time_ms']].copy()
    display_df['anomaly_probability'] = display_df['anomaly_probability'].round(3)
    display_df['inference_time_ms'] = display_df['inference_time_ms'].round(1)
    display_df.columns = ['Vehicle ID', 'Model', 'Location', 'Status', 'Anomaly Score', 'Response (ms)']
    
    return display_df

def main():
    """Main Tesla Dashboard Application"""
    
    # Page config
    st.set_page_config(
        page_title="Tesla NVH Dashboard",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for Tesla styling
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D30 100%);
    }
    .metric-card {
        background: linear-gradient(145deg, #2D2D30, #3E3E42);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #555;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    create_tesla_header()
    
    # Sidebar controls
    st.sidebar.markdown("### 🔧 Tesla Controls")
    
    # Fixed: Auto-refresh disabled by default to prevent blinking
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (10s)", value=False)
    
    # Manual refresh button
    if st.sidebar.button("🚀 Refresh Now"):
        st.rerun()
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "🚗 Filter by Tesla Model",
        ['Model 3', 'Model S', 'Model X', 'Cybertruck'],
        default=['Model 3', 'Model S', 'Model X', 'Cybertruck']
    )
    
    # Status filter
    status_filter = st.sidebar.multiselect(
        "📊 Filter by Status",
        ['NORMAL', 'WARNING', 'CRITICAL'],
        default=['NORMAL', 'WARNING', 'CRITICAL']
    )
    
    # Load Tesla model info
    model, model_path = load_tesla_model()
    if model:
        st.sidebar.success(f"✅ Tesla NVH Model Loaded")
        st.sidebar.info(f"📁 Path: {model_path}")
    else:
        st.sidebar.warning("⚠️ Using Simulated Data")
    
    # Generate live data
    vehicle_data = generate_live_vehicle_data()
    
    # Apply filters
    filtered_data = [
        v for v in vehicle_data 
        if v['model'] in selected_models and v['status'] in status_filter
    ]
    
    # Display metrics
    st.markdown("### 📊 Fleet Overview")
    create_metrics_cards(filtered_data)
    
    # Main dashboard content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🌍 Global Fleet Monitoring")
        fleet_map = create_fleet_map(filtered_data)
        # Fixed: Use width="stretch" instead of use_container_width=True
        st.plotly_chart(fleet_map, width="stretch")
    
    with col2:
        st.markdown("### 🔴 Critical Alerts")
        critical_vehicles = [v for v in filtered_data if v['status'] == 'CRITICAL']
        
        if critical_vehicles:
            for vehicle in critical_vehicles:
                st.markdown(f"""
                <div style="background: {TESLA_RED}; padding: 0.8rem; border-radius: 5px; margin: 0.5rem 0;">
                    <strong style="color: white;">{vehicle['vehicle_id']}</strong><br>
                    <span style="color: #FFCCCC;">Score: {vehicle['anomaly_probability']:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No critical issues detected")
    
    # Real-time analytics section
    st.markdown("### 📈 Real-time Analytics")
    
    fig_dist, fig_model, fig_heatmap = create_realtime_charts(filtered_data)
    
    col3, col4 = st.columns(2)
    with col3:
        # Fixed: Use width="stretch" instead of use_container_width=True
        st.plotly_chart(fig_dist, width="stretch")
    with col4:
        # Fixed: Use width="stretch" instead of use_container_width=True
        st.plotly_chart(fig_model, width="stretch")
    
    # Performance heatmap
    st.markdown("### 🌡️ Vehicle Performance Heatmap")
    # Fixed: Use width="stretch" instead of use_container_width=True
    st.plotly_chart(fig_heatmap, width="stretch")
    
    # Detailed vehicle analysis
    st.markdown("### 📋 Detailed Vehicle Analysis")
    df_table = create_vehicle_details_table(filtered_data)
    # Fixed: Use width="stretch" instead of use_container_width=True
    st.dataframe(df_table, width="stretch")
    
    # Live vehicle details sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🚗 Live Vehicle Status")
        
        # Show top 3 vehicles by anomaly score
        top_vehicles = sorted(filtered_data, key=lambda x: x['anomaly_probability'], reverse=True)[:3]
        
        for i, vehicle in enumerate(top_vehicles):
            # Fixed: Use safer emoji alternatives
            status_emoji = "🔴" if vehicle['status'] == 'CRITICAL' else "⚠️" if vehicle['status'] == 'WARNING' else "✅"
            
            st.markdown(f"""
            **{status_emoji} {vehicle['vehicle_id']}**  
            {vehicle['model']} | {vehicle['location']}  
            Score: {vehicle['anomaly_probability']:.3f} | {vehicle['inference_time_ms']:.1f}ms
            """)
            
            if i < len(top_vehicles) - 1:
                st.markdown("---")
        
        # System performance metrics
        st.markdown("---")
        st.markdown("### ⚡ System Performance")
        
        total_vehicles = len(filtered_data)
        avg_response = np.mean([v['inference_time_ms'] for v in filtered_data]) if filtered_data else 0
        health_score = len([v for v in filtered_data if v['status'] == 'NORMAL']) / len(filtered_data) * 100 if filtered_data else 100
        
        st.metric("Total Vehicles", total_vehicles)
        st.metric("Avg Response", f"{avg_response:.1f}ms")
        st.metric("Fleet Health", f"{health_score:.1f}%")
        
        # Model status
        st.markdown("---")
        st.markdown("### 🤖 Model Status")
        if model_path:
            st.success("Model: Active")
            st.info(f"Version: Tesla NVH v1.0")
        else:
            st.warning("Model: Simulation")
    
    # Fixed: Auto-refresh logic with longer delay to prevent blinking
    if auto_refresh:
        time.sleep(10)  # 10 second delay instead of 1 second
        st.rerun()

if __name__ == "__main__":
    main()