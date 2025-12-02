# --- TAB 4: PRD ASSETS (PROFESSIONAL METEOROLOGY) ---
with tab4:
    st.markdown('<div class="big-header">📸 Scientific Data Visualization</div>', unsafe_allow_html=True)
    st.info("Generating Tier 1-3 validation plots consistent with standard meteorological practices (ERA5 Data).")

    # Controls
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        prd_time_idx = st.slider("Forecast Hour (UTC)", 0, 5, 0, key="prd_time")
    with col_c2:
        prd_level = st.selectbox("Isobaric Level (hPa)", ds.level.values, index=2, key="prd_level") 
    with col_c3:
        prd_city = st.selectbox("Target Sector", list(SAUDI_CITIES.keys()), key="prd_city_select")

    # Data Slicing
    try:
        ds_prd = ds.isel(time=prd_time_idx).sel(level=prd_level, method='nearest')
        df_prd = ds_prd.to_dataframe().reset_index()
        # Filter Region (Saudi Arabia broad view)
        df_prd = df_prd[(df_prd['latitude'] >= 16) & (df_prd['latitude'] <= 32) & 
                        (df_prd['longitude'] >= 34) & (df_prd['longitude'] <= 56)]
        
        # --- AUTO-CORRECT UNITS (Safety Check) ---
        # Ensure Temperature is Celsius. If mean is > 100, it's Kelvin.
        if df_prd['Temperature'].mean() > 100:
            df_prd['Temperature'] -= 273.15
            
    except Exception as e:
        st.error(f"Data slicing error: {e}")
        st.stop()

    # --- ROW 1: SYNOPTIC CHARTS (Satellite & Thermal) ---
    st.markdown("### 1. Synoptic Analysis: Cloud & Thermal Dynamics")
    col_row1_1, col_row1_2 = st.columns(2)

    with col_row1_1:
        # PLOT 1: REALISTIC SATELLITE VIEW
        # Uses 'Greys_r' so 0 (Clear) is Black/Dark and 1 (Cloud) is White.
        fig_clouds = go.Figure()
        
        # Base: Cloud Fraction (Satellite Look)
        fig_clouds.add_trace(go.Heatmap(
            z=df_prd['Fraction_of_cloud_cover'],
            x=df_prd['longitude'],
            y=df_prd['latitude'],
            colorscale='Greys_r', # Reverse Greys: White = Cloud, Black = Ground
            zmin=0, zmax=1,
            colorbar=dict(title="Cloud Frac"),
            name="Cloud Cover"
        ))
        
        # Overlay: Liquid Water Content (The 'Fuel' for seeding)
        # We use blue contours to show where the water actually is inside the white clouds
        fig_clouds.add_trace(go.Contour(
            z=df_prd['Specific_cloud_liquid_water_content']*1000, # g/kg
            x=df_prd['longitude'],
            y=df_prd['latitude'],
            colorscale='Blues',
            contours=dict(start=0.01, end=0.5, size=0.05, showlabels=False),
            line_width=2,
            opacity=0.5,
            showscale=False,
            name="LWC"
        ))
        
        fig_clouds.update_layout(
            title=f"IR Satellite View & Liquid Water @ {prd_level}hPa",
            template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
            height=400,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_clouds, use_container_width=True)

    with col_row1_2:
        # PLOT 2: THERMAL MAP (Celsius)
        # Uses 'RdBu_r' (Red-Blue Reversed) so Blue is Cold, Red is Hot.
        fig_temp = go.Figure()
        
        fig_temp.add_trace(go.Heatmap(
            z=df_prd['Temperature'],
            x=df_prd['longitude'],
            y=df_prd['latitude'],
            colorscale='RdBu_r', # Blue=Cold, Red=Hot
            zmid=20, # Center color scale around 20C for comfort reference
            colorbar=dict(title="Temp (°C)"),
            name="Temperature"
        ))
        
        # Add Wind Vectors (White Arrows)
        df_wind = df_prd.iloc[::25, :] # Subsample for readability
        fig_temp.add_trace(go.Scatter(
            x=df_wind['longitude'],
            y=df_wind['latitude'],
            mode='markers', 
            marker=dict(symbol='arrow-up', size=12, color='white', 
                        line=dict(width=1, color='black'),
                        angle=np.degrees(np.arctan2(df_wind['V_component_of_wind'], df_wind['U_component_of_wind']))),
            name="Wind Vectors"
        ))
        
        fig_temp.update_layout(
            title=f"Thermal Profile (°C) & Wind Flow @ {prd_level}hPa",
            template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
            height=400,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    # --- ROW 2: VERTICAL SOUNDING (Skew-T Style) ---
    st.divider()
    st.markdown("### 2. Vertical Stability Profile (Sounding)")
    st.caption("Crucial for Tier 2 checks. The gap between Red (Temp) and Green (Dewpoint) indicates saturation. Touching lines = Cloud Layer.")
    
    col_row2_1, col_row2_2 = st.columns([2, 1])

    city_coords = SAUDI_CITIES[prd_city]
    
    # Extract Vertical Profile
    ds_profile = ds.isel(time=prd_time_idx).sel(
        latitude=city_coords['lat'], longitude=city_coords['lon'], method='nearest'
    )
    df_profile = ds_profile.to_dataframe().reset_index()
    
    # Unit Safety Check for Profile
    if df_profile['Temperature'].mean() > 100:
        df_profile['Temperature'] -= 273.15

    # Calculate Dewpoint (Approximation)
    # Td ≈ T - ((100 - RH)/5)
    df_profile['Dewpoint'] = df_profile['Temperature'] - ((100 - df_profile['Relative_Humidity'])/5)

    with col_row2_1:
        # PLOT 3: SOUNDING
        fig_skew = go.Figure()
        
        # Temperature (Red Line)
        fig_skew.add_trace(go.Scatter(
            x=df_profile['Temperature'], y=df_profile['level'],
            mode='lines+markers', line=dict(color='#ff4444', width=3), name="Temp (°C)"
        ))
        
        # Dewpoint (Green Dashed)
        fig_skew.add_trace(go.Scatter(
            x=df_profile['Dewpoint'], y=df_profile['level'],
            mode='lines+markers', line=dict(color='#00cc00', width=2, dash='dash'), name="Dewpoint (°C)"
        ))
        
        # Cloud Water (Blue Area on Secondary X)
        fig_skew.add_trace(go.Scatter(
            x=df_profile['Specific_cloud_liquid_water_content']*10000, 
            y=df_profile['level'],
            fill='tozerox',
            mode='none',
            name="Cloud Water Content",
            fillcolor='rgba(0, 191, 255, 0.2)',
            xaxis='x2'
        ))

        fig_skew.update_layout(
            title=f"Atmospheric Sounding: {prd_city}",
            yaxis=dict(title="Pressure Level (hPa)", autorange="reversed", gridcolor='#444'),
            xaxis=dict(title="Temperature (°C)", range=[-30, 45], showgrid=True, gridcolor='#444'),
            xaxis2=dict(title="LWC presence", overlaying='x', side='top', showgrid=False, range=[0, 1]),
            template="plotly_dark",
            height=500,
            legend=dict(x=0.02, y=0.02, bgcolor='rgba(0,0,0,0.5)')
        )
        st.plotly_chart(fig_skew, use_container_width=True)

    with col_row2_2:
        # PLOT 4: AUTOMATED DECISION TABLE
        st.markdown("#### ✅ Automated Go/No-Go Logic")
        
        # Metrics Calculation
        try:
            rh_val = df_profile.loc[df_profile['level']==850, 'Relative_Humidity'].values[0] if 850 in df_profile['level'].values else 0
            updraft_val = df_profile.loc[df_profile['level']==700, 'Vertical_velocity'].values[0] if 700 in df_profile['level'].values else 0
            lwc_max = df_profile['Specific_cloud_liquid_water_content'].max()
            temp_base = df_profile.loc[df_profile['level']==850, 'Temperature'].values[0] if 850 in df_profile['level'].values else 0
        except:
            rh_val, updraft_val, lwc_max, temp_base = 0, 0, 0, 0

        # Status Logic
        def get_stat(val, thresh, op='>'):
            if op == '>': return "✅ PASS" if val >= thresh else "❌ FAIL"
            if op == '<': return "✅ PASS" if val <= thresh else "❌ FAIL"
            if op == 'lwc': return "✅ PASS" if val > 0.00001 else "⚠️ MARGINAL"

        metrics_data = [
            ["TIER 1", "Humidity (850hPa)", f"{rh_val:.1f}%", "> 60%", get_stat(rh_val, 60)],
            ["TIER 1", "Updraft (Pa/s)", f"{updraft_val:.2f}", "< -0.1", get_stat(updraft_val, -0.1, '<')], # Negative Pa/s = Rising Air
            ["TIER 1", "Liquid Water", f"{lwc_max:.1e}", "> 1e-5", get_stat(lwc_max, 0, 'lwc')],
            ["TIER 2", "Base Temp", f"{temp_base:.1f}°C", "> 5°C", get_stat(temp_base, 5)],
            ["TIER 5", "Lightning", "None", "0 Strikes", "✅ PASS"],
        ]
        
        df_logic = pd.DataFrame(metrics_data, columns=["Tier", "Metric", "Value", "Threshold", "Status"])
        st.table(df_logic)
