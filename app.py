
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
import os

# Page configuration
st.set_page_config(
    page_title="Global Wildlife Habitat Monitor",
    page_icon="🌿",
    layout="wide"
)

# --- Sidebar ---
st.sidebar.title("🌿 Global Wildlife Monitor")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Earth_Eastern_Hemisphere.jpg/800px-Earth_Eastern_Hemisphere.jpg", width=100)
st.sidebar.markdown("""
## About this Project
This application visualizes vegetation health trends in major protected wildlife areas around the world using satellite imagery analysis.

**Key Features:**
- NDVI (Normalized Difference Vegetation Index) time series
- Seasonal variation analysis
- Forest cover change detection
- Wildlife corridor connectivity metrics

*Developed for conservation organizations monitoring habitat health and biodiversity.*
""")

# --- Main Content ---
st.title("Satellite Monitoring of Global Wildlife Habitats")
st.markdown("""
This dashboard provides key vegetation health indicators derived from Sentinel-2 satellite imagery 
across major protected areas worldwide. The data helps conservation teams identify habitat degradation, 
monitor restoration efforts, and support wildlife protection strategies.
""")

# Protected areas around the world with geographical coordinates
protected_areas = {
    "Yellowstone National Park": {"lat": 44.4280, "lon": -110.5885, "area_km2": 8983},
    "Serengeti National Park": {"lat": -2.3333, "lon": 34.8333, "area_km2": 14763},
    "Great Barrier Reef Marine Park": {"lat": -18.2871, "lon": 147.7000, "area_km2": 344400},
    "Kruger National Park": {"lat": -23.9884, "lon": 31.5547, "area_km2": 19485},
    "Banff National Park": {"lat": 51.4968, "lon": -115.9281, "area_km2": 6641},
    "Galapagos National Park": {"lat": -0.8333, "lon": -91.1333, "area_km2": 7995}
}

years = list(range(2017, 2025))
seasons = ["Dry Season", "Transition Season", "Wet Season"]

# --- Generate simulated data ---
@st.cache_data
def generate_ndvi_data():
    """Generate simulated NDVI data"""
    np.random.seed(42)
    data = []
    
    # Create specific patterns for each area
    for area, details in protected_areas.items():
        # Base NDVI pattern (higher in wet season, lower in dry)
        if "Yellowstone" in area:  # Temperate forest ecosystem
            base = 0.68
            trend = 0.002  # slight improvement
            seasonal_amp = 0.22
        elif "Serengeti" in area:  # Savanna ecosystem, high seasonal variation
            base = 0.58
            trend = -0.002  # slight decline
            seasonal_amp = 0.25
        elif "Great Barrier Reef" in area:  # Marine ecosystem
            base = 0.35
            trend = -0.015  # significant decline
            seasonal_amp = 0.05
        elif "Kruger" in area:  # Subtropical woodland
            base = 0.62
            trend = -0.001  # stable to slight decline
            seasonal_amp = 0.18
        elif "Banff" in area:  # Alpine forest, high baseline
            base = 0.72
            trend = 0.001  # stable
            seasonal_amp = 0.26
        else:  # Galapagos - arid islands with unique ecosystem
            base = 0.50
            trend = -0.002
            seasonal_amp = 0.10
            
        # Generate data for each year and season
        for i, year in enumerate(years):
            yearly_base = base + trend * i
            
            for season in seasons:
                if "Dry" in season:
                    seasonal_factor = -seasonal_amp
                elif "Transition" in season:
                    seasonal_factor = seasonal_amp * 0.3
                else:
                    seasonal_factor = seasonal_amp
                
                # Add random noise
                ndvi = yearly_base + seasonal_factor + np.random.normal(0, 0.03)
                ndvi = min(max(ndvi, 0.2), 0.95)  # Clip values to reasonable NDVI range
                
                # Add some random drops for disturbances in certain years/areas
                if (year == 2019 and "Yellowstone" in area and "Dry" in season) or \
                   (year == 2022 and "Great Barrier Reef" in area):
                    ndvi -= 0.15
                
                data.append({
                    "Location": area,
                    "Year": year,
                    "Season": season,
                    "NDVI": ndvi,
                    "Area_km2": details["area_km2"],
                    "Latitude": details["lat"],
                    "Longitude": details["lon"],
                    "Data_Source": "Simulated"
                })
    
    return pd.DataFrame(data)

# Generate additional habitat metrics based on NDVI
@st.cache_data
def generate_habitat_metrics(df):
    # Create a copy to avoid modifying the original
    metrics_df = df.copy()
    
    # Forest cover percentage (derived from NDVI)
    metrics_df["Forest_Cover_Pct"] = metrics_df["NDVI"].apply(
        lambda x: min(100, max(0, (x - 0.2) * 130))
    )
    
    # Wildlife corridor connectivity (0-100 scale)
    # Higher NDVI areas generally have better connectivity
    metrics_df["Connectivity_Index"] = metrics_df["NDVI"].apply(
        lambda x: min(100, max(0, (x - 0.3) * 120)) + np.random.normal(0, 5)
    )
    
    # Habitat fragmentation index (0-100, higher is more fragmented)
    # Inverse relationship with NDVI
    metrics_df["Fragmentation_Index"] = metrics_df["NDVI"].apply(
        lambda x: min(100, max(0, (0.9 - x) * 120)) + np.random.normal(0, 7)
    )
    
    # Clip values to valid ranges
    metrics_df["Connectivity_Index"] = metrics_df["Connectivity_Index"].clip(0, 100)
    metrics_df["Fragmentation_Index"] = metrics_df["Fragmentation_Index"].clip(0, 100)
    
    return metrics_df

# Display notice that we're using simulated data
st.sidebar.warning("⚠️ Using simulated data only. Earth Engine integration will be available in a future update.")

# Load simulated data
ndvi_df = generate_ndvi_data()

# Generate habitat metrics
metrics_df = generate_habitat_metrics(ndvi_df)

# --- Location Selection ---
selected_area = st.selectbox("Select Protected Area:", list(protected_areas.keys()))

# Filter data for selected area
area_df = metrics_df[metrics_df["Location"] == selected_area]

# --- Dashboard layout ---
st.markdown(f"## {selected_area}")
col1, col2 = st.columns([7, 3])

with col1:
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["NDVI Trends", "Seasonal Patterns", "Habitat Metrics"])
    
    with tab1:
        # NDVI time series by year (averaged across seasons)
        yearly_avg = area_df.groupby("Year")["NDVI"].mean().reset_index()
        
        fig_trend = px.line(
            yearly_avg, x="Year", y="NDVI", 
            markers=True, 
            line_shape="spline",
            title=f"Annual NDVI Trend (2017-2024)"
        )
        fig_trend.update_traces(line=dict(width=4))
        fig_trend.update_layout(
            xaxis=dict(tickmode='linear'),
            yaxis=dict(range=[0.4, 0.9], title="NDVI Value"),
            hovermode="x unified"
        )
        
        # Add reference line for global average NDVI
        fig_trend.add_hline(y=0.6, line_dash="dash", line_color="gray", 
                    annotation_text="Global Forest Average", annotation_position="bottom right")
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Calculate change metrics
        start_ndvi = yearly_avg[yearly_avg["Year"] == 2017]["NDVI"].values[0]
        end_ndvi = yearly_avg[yearly_avg["Year"] == 2024]["NDVI"].values[0]
        pct_change = ((end_ndvi - start_ndvi) / start_ndvi) * 100
        
        # Create metrics row
        st.markdown("### Vegetation Health Change (2017-2024)")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("NDVI Change", f"{(end_ndvi - start_ndvi):.3f}", 
                      f"{pct_change:.1f}%", delta_color="normal" if pct_change >= 0 else "inverse")
        with metric_col2:
            st.metric("Initial NDVI (2017)", f"{start_ndvi:.3f}")
        with metric_col3:
            st.metric("Current NDVI (2024)", f"{end_ndvi:.3f}")
        
    with tab2:
        # Seasonal patterns across years
        fig_seasonal = px.line(
            area_df, x="Year", y="NDVI", color="Season", 
            title=f"Seasonal NDVI Patterns (2017-2024)",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_seasonal.update_layout(
            xaxis=dict(tickmode='linear'),
            yaxis=dict(title="NDVI Value"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Seasonal box plots
        fig_box = px.box(
            area_df, x="Season", y="NDVI", 
            title="NDVI Distribution by Season",
            color="Season", 
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    with tab3:
        # Habitat metrics over time
        metrics_yearly = area_df.groupby("Year")[["Forest_Cover_Pct", "Connectivity_Index", "Fragmentation_Index"]].mean().reset_index()
        
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Scatter(
            x=metrics_yearly["Year"], y=metrics_yearly["Forest_Cover_Pct"],
            mode="lines+markers", name="Forest Cover %",
            line=dict(width=3, color="green")
        ))
        fig_metrics.add_trace(go.Scatter(
            x=metrics_yearly["Year"], y=metrics_yearly["Connectivity_Index"],
            mode="lines+markers", name="Connectivity Index",
            line=dict(width=3, color="blue")
        ))
        fig_metrics.add_trace(go.Scatter(
            x=metrics_yearly["Year"], y=metrics_yearly["Fragmentation_Index"],
            mode="lines+markers", name="Fragmentation Index",
            line=dict(width=3, color="red")
        ))
        
        fig_metrics.update_layout(
            title="Habitat Quality Metrics Over Time",
            xaxis=dict(tickmode='linear'),
            yaxis=dict(title="Index Value (0-100)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)

with col2:
    # Area information and metadata
    st.markdown("### Area Information")
    st.markdown(f"""
    **Size:** {protected_areas[selected_area]['area_km2']} km²  
    **Location:** {protected_areas[selected_area]['lat']:.4f}°N, {protected_areas[selected_area]['lon']:.4f}°E  
    """)
    
    # Show data source
    st.markdown("**Data Source:** <span style='color:orange'>Simulated</span>", unsafe_allow_html=True)
    
    # Map placeholder 
    st.markdown("### Location Map")
    map_placeholder = st.empty()
    
    # Use a conditional to show appropriate placeholder image based on selected area
    if "Yellowstone" in selected_area:
        map_placeholder.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Yellowstone_National_Park_satellite_map.jpg/640px-Yellowstone_National_Park_satellite_map.jpg", caption="Yellowstone National Park (Satellite Image)")
    elif "Serengeti" in selected_area:
        map_placeholder.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Serengeti_national_park_map.png/640px-Serengeti_national_park_map.png", caption="Serengeti National Park (Map View)")
    elif "Great Barrier Reef" in selected_area:
        map_placeholder.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Great_Barrier_Reef_satellite_image.jpg/640px-Great_Barrier_Reef_satellite_image.jpg", caption="Great Barrier Reef (Satellite Image)")
    elif "Kruger" in selected_area:
        map_placeholder.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Kruger_National_Park_topographic_map-en.svg/640px-Kruger_National_Park_topographic_map-en.svg.png", caption="Kruger National Park (Topographic Map)")
    elif "Banff" in selected_area:
        map_placeholder.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Satellite_image_of_Banff_and_Canmore%2C_Alberta.jpg/640px-Satellite_image_of_Banff_and_Canmore%2C_Alberta.jpg", caption="Banff National Park (Satellite Image)")
    else:
        map_placeholder.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Satellite_image_of_the_Gal%C3%A1pagos_Islands.jpg/640px-Satellite_image_of_the_Gal%C3%A1pagos_Islands.jpg", caption="Galapagos Islands (Satellite Image)")
    
    # Biodiversity stats (simulated)
    st.markdown("### Biodiversity Statistics")
    
    # Simulate some biodiversity data based on NDVI
    recent_ndvi = area_df[area_df["Year"] == 2024]["NDVI"].mean()
    
    # Customize species count based on selected area
    if "Yellowstone" in selected_area:
        species_count = int(380 + (recent_ndvi * 200))
        endangered_count = int(24 + (1 - recent_ndvi) * 30)
    elif "Serengeti" in selected_area:
        species_count = int(500 + (recent_ndvi * 300))
        endangered_count = int(38 + (1 - recent_ndvi) * 40)
    elif "Great Barrier Reef" in selected_area:
        species_count = int(1500 + (recent_ndvi * 500))
        endangered_count = int(120 + (1 - recent_ndvi) * 100)
    elif "Kruger" in selected_area:
        species_count = int(450 + (recent_ndvi * 250))
        endangered_count = int(42 + (1 - recent_ndvi) * 35)
    elif "Banff" in selected_area:
        species_count = int(290 + (recent_ndvi * 180))
        endangered_count = int(15 + (1 - recent_ndvi) * 25)
    else:  # Galapagos
        species_count = int(600 + (recent_ndvi * 400))
        endangered_count = int(80 + (1 - recent_ndvi) * 50)
    
    st.metric("Estimated Species Count", species_count)
    st.metric("Endangered Species", endangered_count)
    
    # Conservation status
    ndvi_trend = yearly_avg["NDVI"].iloc[-1] - yearly_avg["NDVI"].iloc[0]
    if ndvi_trend > 0.01:
        status = "Improving"
        color = "green"
    elif ndvi_trend < -0.01:
        status = "Declining"
        color = "red"
    else:
        status = "Stable"
        color = "orange"
        
    st.markdown(f"### Conservation Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)

# --- Comparison section ---
st.markdown("## Compare with Other Protected Areas")

# Select areas to compare
areas_to_compare = st.multiselect(
    "Select areas to compare:", 
    list(protected_areas.keys()),
    default=[selected_area]
)

if areas_to_compare:
    # Filter data for selected areas
    compare_df = metrics_df[metrics_df["Location"].isin(areas_to_compare)]
    
    # Calculate yearly averages for each area
    compare_yearly = compare_df.groupby(["Location", "Year"])["NDVI"].mean().reset_index()
    
    # Create comparison chart
    fig_compare = px.line(
        compare_yearly, x="Year", y="NDVI", color="Location",
        title="NDVI Comparison Across Protected Areas",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig_compare.update_layout(
        xaxis=dict(tickmode='linear'),
        yaxis=dict(title="NDVI Value"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # Bar chart for latest year comparison
    latest_year_df = compare_df[compare_df["Year"] == 2024].groupby("Location")[["NDVI", "Forest_Cover_Pct", "Connectivity_Index"]].mean().reset_index()
    
    fig_bar = px.bar(
        latest_year_df, x="Location", y="NDVI",
        title="2024 NDVI Comparison",
        color="Location",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Download Section ---
st.markdown("## Download Data")

# Add functionality to download chart as image
def get_chart_as_png(fig):
    img_bytes = fig.to_image(format="png", width=1200, height=600)
    return img_bytes

# Create download buttons for charts
col1, col2, col3 = st.columns(3)
with col1:
    st.download_button(
        label="Download NDVI Trend Chart",
        data=get_chart_as_png(fig_trend),
        file_name=f"{selected_area}_ndvi_trend.png",
        mime="image/png"
    )
with col2:
    st.download_button(
        label="Download Seasonal Chart",
        data=get_chart_as_png(fig_seasonal),
        file_name=f"{selected_area}_seasonal_ndvi.png",
        mime="image/png"
    )
with col3:
    st.download_button(
        label="Download Comparison Chart",
        data=get_chart_as_png(fig_compare),
        file_name="protected_areas_comparison.png",
        mime="image/png"
    )

# --- Download raw data ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(area_df)
st.download_button(
    label="Download Raw Data (CSV)",
    data=csv,
    file_name=f"{selected_area}_ndvi_data.csv",
    mime='text/csv',
)

# --- About the Data ---
with st.expander("About the Satellite Data"):
    st.markdown("""
    ### Simulated Satellite Data
    
    This application currently uses **simulated NDVI data** that mimics patterns typically observed in different global ecosystems.
    
    **About the Simulation:**
    - Data is modeled to reflect typical seasonal variations in different habitat types
    - Yearly trends are based on known conservation patterns in each protected area
    - Random variations and disturbances are added to simulate real-world conditions
    
    **About NDVI:**
    NDVI values range from -1 to +1, with:
    - Values < 0: Water bodies, built-up areas
    - 0.1 - 0.3: Sparse vegetation, bare soil
    - 0.3 - 0.5: Moderate vegetation (grasslands, shrubs)
    - 0.5 - 0.9: Dense, healthy vegetation (forests)
    
    **Future Updates:**
    A future update will integrate real Sentinel-2 satellite data via Earth Engine for more accurate monitoring.
    """)

# --- Footer ---
st.markdown("""
---
Created for Global Wildlife Conservation Initiative | Last updated: July 2025
""")