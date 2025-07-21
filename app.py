import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
import folium
from streamlit_folium import st_folium
import os

# Optional Earth Engine import with robust error handling
try:
    import ee
    EE_AVAILABLE = True
except ImportError as e:
    st.warning(f"Earth Engine not available: {e}. Using simulated data only.")
    EE_AVAILABLE = False
    ee = None

# Page configuration
st.set_page_config(
    page_title="Vietnam Wildlife Habitat Monitor",
    page_icon="🌿",
    layout="wide"
)

# Initialize Earth Engine with enhanced error handling
if EE_AVAILABLE:
    try:
        ee.Initialize()
        use_simulated_data = False
        st.sidebar.success("✅ Earth Engine connected successfully")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Earth Engine initialization failed: {e}")
        use_simulated_data = True
        EE_AVAILABLE = False
else:
    use_simulated_data = True

# --- Sidebar ---
st.sidebar.title("🌿 Vietnam Wildlife Monitor")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Flag_of_Vietnam.svg/800px-Flag_of_Vietnam.svg.png", width=100)
st.sidebar.markdown("""
## About this Project
This application visualizes vegetation health trends in Vietnam's protected wildlife areas using satellite imagery analysis.

**Key Features:**
- NDVI (Normalized Difference Vegetation Index) time series
- Seasonal variation analysis
- Forest cover change detection
- Wildlife corridor connectivity metrics

*Developed for conservation organizations monitoring habitat health and biodiversity.*
""")

# --- Main Content ---
st.title("Satellite Monitoring of Vietnam's Wildlife Habitats")
st.markdown("""
This dashboard provides key vegetation health indicators derived from Sentinel-2 satellite imagery 
across Vietnam's protected areas. The data helps conservation teams identify habitat degradation, 
monitor restoration efforts, and support wildlife protection strategies.
""")

# Protected areas in Vietnam with geographical coordinates
protected_areas = {
    "Cat Ba National Park": {"lat": 20.7980, "lon": 107.0480, "area_km2": 263},
    "Du Gia National Park": {"lat": 23.0893, "lon": 105.8844, "area_km2": 147},
    "Phong Nha-Ke Bang": {"lat": 17.5906, "lon": 106.2836, "area_km2": 857},
    "Pu Mat National Park": {"lat": 19.0218, "lon": 104.7160, "area_km2": 911},
    "Bidoup Nui Ba": {"lat": 12.1844, "lon": 108.6981, "area_km2": 704},
    "Yok Don National Park": {"lat": 12.8825, "lon": 107.7458, "area_km2": 1155}
}

years = list(range(2017, 2025))
seasons = ["Dry (Nov-Apr)", "Early Wet (May-Jul)", "Late Wet (Aug-Oct)"]

# --- EARTH ENGINE FUNCTIONS ---
def get_aoi(area_name):
    """Get area of interest as ee.Geometry for selected protected area"""
    if not EE_AVAILABLE:
        return None
    try:
        details = protected_areas[area_name]
        # Create a buffer around the point to approximate the protected area
        point = ee.Geometry.Point([details['lon'], details['lat']])
        # Calculate buffer radius (approximate) based on area
        radius = np.sqrt(details['area_km2']) * 500  # rough estimate in meters
        return point.buffer(radius)
    except Exception as e:
        st.warning(f"Error creating AOI for {area_name}: {e}")
        return None

def get_season_dates(year, season):
    """Get start and end dates for a specific season in a given year"""
    if "Dry" in season:
        # Dry season spans across year boundary
        if year == 2017:  # Special case for first year
            start_date = f"{year}-01-01"
        else:
            start_date = f"{year-1}-11-01"
        end_date = f"{year}-04-30"
    elif "Early Wet" in season:
        start_date = f"{year}-05-01"
        end_date = f"{year}-07-31"
    else:  # Late Wet
        start_date = f"{year}-08-01"
        end_date = f"{year}-10-31"
    return start_date, end_date

def calculate_ndvi(start_date, end_date, aoi):
    """Calculate NDVI from Sentinel-2 imagery for a given time period and area"""
    if not EE_AVAILABLE or aoi is None:
        return None
        
    try:
        # Filter Sentinel-2 surface reflectance collection
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate(start_date, end_date) \
            .filterBounds(aoi) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        
        # Check if collection is empty
        if s2_collection.size().getInfo() == 0:
            return None
            
        # Function to calculate NDVI for each image
        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return image.addBands(ndvi)
        
        # Map the NDVI calculation over the collection
        ndvi_collection = s2_collection.map(add_ndvi)
        
        # Calculate median NDVI to reduce cloud influence
        median_ndvi = ndvi_collection.select('NDVI').median()
        
        # Get mean NDVI value for the AOI
        mean_ndvi = median_ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        ).get('NDVI')
        
        return mean_ndvi.getInfo()
    except Exception as e:
        st.warning(f"Error calculating NDVI: {e}")
        return None

@st.cache_data(ttl=3600*24)  # Cache for 24 hours
def fetch_ndvi_data():
    """Fetch real NDVI data from Earth Engine for all protected areas, years and seasons"""
    if not EE_AVAILABLE:
        st.info("Earth Engine not available. Using simulated data.")
        return generate_ndvi_data()
        
    data = []
    
    # Show progress
    progress_text = "Fetching satellite data... This may take a few minutes."
    progress_bar = st.progress(0)
    total_iterations = len(protected_areas) * len(years) * len(seasons)
    current_iteration = 0
    
    for area_name, details in protected_areas.items():
        aoi = get_aoi(area_name)
        
        # If AOI creation failed, skip to simulated data for this area
        if aoi is None:
            for year in years:
                for season in seasons:
                    # Generate fallback data for this area
                    fallback_ndvi = generate_fallback_ndvi(area_name, year, season)
                    data.append({
                        "Location": area_name,
                        "Year": year,
                        "Season": season,
                        "NDVI": fallback_ndvi,
                        "Area_km2": details["area_km2"],
                        "Latitude": details["lat"],
                        "Longitude": details["lon"],
                        "Data_Source": "Simulated (EE unavailable)"
                    })
                    current_iteration += 1
                    progress_bar.progress(current_iteration / total_iterations)
            continue
        
        for year in years:
            for season in seasons:
                start_date, end_date = get_season_dates(year, season)
                
                try:
                    ndvi_value = calculate_ndvi(start_date, end_date, aoi)
                    if ndvi_value is not None:
                        # Clip to reasonable NDVI range
                        ndvi_value = min(max(ndvi_value, 0.1), 0.95)
                        
                        data.append({
                            "Location": area_name,
                            "Year": year,
                            "Season": season,
                            "NDVI": ndvi_value,
                            "Area_km2": details["area_km2"],
                            "Latitude": details["lat"],
                            "Longitude": details["lon"],
                            "Data_Source": "Satellite (Sentinel-2)"
                        })
                    else:
                        # Use fallback for this specific entry
                        fallback_ndvi = generate_fallback_ndvi(area_name, year, season)
                        data.append({
                            "Location": area_name,
                            "Year": year,
                            "Season": season,
                            "NDVI": fallback_ndvi,
                            "Area_km2": details["area_km2"],
                            "Latitude": details["lat"],
                            "Longitude": details["lon"],
                            "Data_Source": "Estimated (No satellite data)"
                        })
                except Exception as e:
                    st.warning(f"Error processing {area_name} for {season} {year}: {str(e)}")
                    # Use estimated value as fallback
                    fallback_ndvi = generate_fallback_ndvi(area_name, year, season)
                    data.append({
                        "Location": area_name,
                        "Year": year,
                        "Season": season,
                        "NDVI": fallback_ndvi,
                        "Area_km2": details["area_km2"],
                        "Latitude": details["lat"],
                        "Longitude": details["lon"],
                        "Data_Source": "Estimated (Satellite data error)"
                    })
                
                # Update progress bar
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)
    
    progress_bar.empty()
    return pd.DataFrame(data)

def generate_fallback_ndvi(area_name, year, season):
    """Generate fallback NDVI value for a specific area, year, and season"""
    np.random.seed(hash(f"{area_name}_{year}_{season}") % 1000)  # Consistent random seed
    
    # Use same logic as the original fallback
    if "Cat Ba" in area_name:
        base = 0.65 - 0.005 * (year - 2017)
    elif "Phong Nha" in area_name:
        base = 0.78 + 0.001 * (year - 2017)
    elif "Yok Don" in area_name:
        base = 0.58 - 0.01 * (year - 2017)
    elif "Bidoup Nui Ba" in area_name:
        base = 0.82 + 0.005 * (year - 2017)
    else:
        base = 0.70 - 0.002 * (year - 2017)
        
    # Add seasonal variation
    seasonal_amp = 0.1
    if "Dry" in season:
        seasonal_factor = -seasonal_amp
    elif "Early Wet" in season:
        seasonal_factor = seasonal_amp * 0.5
    else:
        seasonal_factor = seasonal_amp
    
    # Add random noise and clip
    ndvi = base + seasonal_factor + np.random.normal(0, 0.02)
    ndvi = min(max(ndvi, 0.2), 0.95)
    
    return ndvi

# --- Generate simulated data ---
@st.cache_data
def generate_ndvi_data():
    """Generate simulated NDVI data when Earth Engine is not available"""
    np.random.seed(42)
    data = []
    
    # Create specific patterns for each area
    for area, details in protected_areas.items():
        # Base NDVI pattern (higher in wet season, lower in dry)
        if "Cat Ba" in area:  # Island ecosystem, less seasonal variation
            base = 0.65
            trend = -0.005  # slight decline
            seasonal_amp = 0.08
        elif "Phong Nha" in area:  # Limestone forest, high baseline
            base = 0.78
            trend = 0.001  # stable
            seasonal_amp = 0.12
        elif "Yok Don" in area:  # Dry deciduous forest, lower baseline, high seasonal variation
            base = 0.58
            trend = -0.01  # moderate decline
            seasonal_amp = 0.20
        elif "Bidoup Nui Ba" in area:  # Highland forest, high NDVI
            base = 0.82
            trend = 0.005  # improving
            seasonal_amp = 0.10
        else:  # Other areas
            base = 0.70
            trend = -0.002
            seasonal_amp = 0.15
            
        # Generate data for each year and season
        for i, year in enumerate(years):
            yearly_base = base + trend * i
            
            for season in seasons:
                if "Dry" in season:
                    seasonal_factor = -seasonal_amp
                elif "Early Wet" in season:
                    seasonal_factor = seasonal_amp * 0.5
                else:
                    seasonal_factor = seasonal_amp
                
                # Add random noise
                ndvi = yearly_base + seasonal_factor + np.random.normal(0, 0.03)
                ndvi = min(max(ndvi, 0.2), 0.95)  # Clip values to reasonable NDVI range
                
                # Add some random drops for disturbances in certain years/areas
                if (year == 2019 and "Yok Don" in area and "Dry" in season) or \
                   (year == 2022 and "Cat Ba" in area):
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

# --- Add a data source selector ---
if EE_AVAILABLE:
    data_source = st.sidebar.radio(
        "Data Source:",
        ["Real Satellite Data", "Simulated Data"],
        index=0
    )
else:
    st.sidebar.info("Earth Engine unavailable - using simulated data")
    data_source = "Simulated Data"

# Load data based on selection
if data_source == "Real Satellite Data" and EE_AVAILABLE:
    with st.spinner("Fetching satellite data from Earth Engine..."):
        ndvi_df = fetch_ndvi_data()
        st.sidebar.success("Using real Sentinel-2 satellite data")
else:
    ndvi_df = generate_ndvi_data()
    if not EE_AVAILABLE:
        st.sidebar.warning("Using simulated data (Earth Engine unavailable)")
    else:
        st.sidebar.warning("Using simulated data")

# Generate habitat metrics
metrics_df = generate_habitat_metrics(ndvi_df)

# --- Location Selection ---
selected_area = st.selectbox("Select Protected Area:", list(protected_areas.keys()))

# Filter data for selected area
area_df = metrics_df[metrics_df["Location"] == selected_area]

# --- Display Folium Map ---
st.sidebar.markdown("### View Interactive Map")
show_map = st.sidebar.checkbox("Show Interactive Map", value=True)

if show_map:
    with st.expander("Protected Area Map", expanded=True):
        # Create a map centered on the selected area
        details = protected_areas[selected_area]
        
        # Create folium map
        m = folium.Map(
            location=[details['lat'], details['lon']], 
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Add marker for the protected area
        folium.Marker(
            [details['lat'], details['lon']],
            popup=f"""
            <b>{selected_area}</b><br>
            Area: {details['area_km2']} km²<br>
            Coordinates: {details['lat']:.4f}°N, {details['lon']:.4f}°E
            """,
            tooltip=selected_area,
            icon=folium.Icon(color='green', icon='tree', prefix='fa')
        ).add_to(m)
        
        # Add a circle to represent the approximate protected area
        radius = np.sqrt(details['area_km2']) * 500  # rough estimate in meters
        folium.Circle(
            location=[details['lat'], details['lon']],
            radius=radius,
            popup=f"{selected_area} Boundary (Approximate)",
            color='green',
            fillColor='lightgreen',
            fillOpacity=0.3,
            weight=2
        ).add_to(m)
        
        # Add all other protected areas as reference points
        for area_name, area_details in protected_areas.items():
            if area_name != selected_area:
                folium.Marker(
                    [area_details['lat'], area_details['lon']],
                    popup=f"""
                    <b>{area_name}</b><br>
                    Area: {area_details['area_km2']} km²<br>
                    Click to select this area
                    """,
                    tooltip=area_name,
                    icon=folium.Icon(color='blue', icon='leaf', prefix='fa')
                ).add_to(m)
        
        # Add satellite layer option
        if EE_AVAILABLE and data_source == "Real Satellite Data":
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite Imagery',
                overlay=False,
                control=True
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Display the map
        st_folium(m, width=700, height=500)
        
        # Add NDVI context information
        if data_source == "Real Satellite Data" and EE_AVAILABLE:
            st.info("🛰️ When using real satellite data, NDVI values are calculated from Sentinel-2 imagery")
        else:
            st.info("📊 Currently showing simulated NDVI data - enable Earth Engine for real satellite data")

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
    if "Data_Source" in area_df.columns and len(area_df) > 0:
        data_source_info = area_df["Data_Source"].iloc[0] if "Data_Source" in area_df.columns else "Simulated"
        source_color = "green" if data_source_info != "Simulated" else "orange"
        st.markdown(f"**Data Source:** <span style='color:{source_color}'>{data_source_info}</span>", unsafe_allow_html=True)
    
    # Biodiversity stats (simulated)
    st.markdown("### Biodiversity Statistics")
    
    # Simulate some biodiversity data based on NDVI
    recent_ndvi = area_df[area_df["Year"] == 2024]["NDVI"].mean()
    species_count = int(100 + (recent_ndvi * 300))
    endangered_count = int(15 + (1 - recent_ndvi) * 30)
    
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
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600)
        return img_bytes
    except Exception as e:
        st.warning(f"Chart image export not available. Install kaleido for image downloads: pip install kaleido")
        return None

# Create download buttons for charts
col1, col2, col3 = st.columns(3)

# Check if chart export is available
try:
    chart_data = get_chart_as_png(fig_trend)
    if chart_data:
        with col1:
            st.download_button(
                label="Download NDVI Trend Chart",
                data=chart_data,
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
    else:
        st.info("📊 Chart image downloads require additional dependencies. CSV data download is available below.")
except Exception:
    with col1:
        st.button("Download NDVI Trend Chart", disabled=True, help="Requires kaleido package")
    with col2:
        st.button("Download Seasonal Chart", disabled=True, help="Requires kaleido package")
    with col3:
        st.button("Download Comparison Chart", disabled=True, help="Requires kaleido package")

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
    ### Satellite Data Sources
    
    This application uses **Sentinel-2** satellite imagery to calculate NDVI (Normalized Difference Vegetation Index).
    
    **Data Processing:**
    - Imagery is filtered for cloud coverage below 30%
    - NDVI is calculated using NIR (Band 8) and Red (Band 4) spectral bands
    - Temporal composites are created for each season to minimize cloud effects
    - Spatial averaging is performed across the protected area boundaries
    
    **About NDVI:**
    NDVI values range from -1 to +1, with:
    - Values < 0: Water bodies, built-up areas
    - 0.1 - 0.3: Sparse vegetation, bare soil
    - 0.3 - 0.5: Moderate vegetation (grasslands, shrubs)
    - 0.5 - 0.9: Dense, healthy vegetation (forests)
    
    **Data Limitations:**
    - Cloud cover in wet seasons can reduce data availability
    - Protected area boundaries are approximated
    - Derived metrics (connectivity, fragmentation) are modeled from NDVI
    """)

# --- Footer ---
st.markdown("""
---
Created for Vietnam Wildlife Conservation Association | Last updated: July 2024
""")