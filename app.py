# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import utils_eda
import corrected_results as cr
from pipeline.features import build_features

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pest Forecasting Project", page_icon="🐞", layout="wide")


# --- LOAD DATA AND RESULTS (runs once) ---
@st.cache_data
def load_all_data():
    """Load the datasets; replace the leaking Recent_Activity column with the corrected one."""
    engineered_data, ts_data = utils_eda.load_data()
    if engineered_data is None or ts_data is None:
        return None, None
    rebuilt = build_features(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_merged_data.csv"))[["Date", "Location", "Recent_Activity"]]
    engineered_data = engineered_data.drop(columns=["Recent_Activity"]).merge(rebuilt, on=["Date", "Location"], how="left")
    return engineered_data, ts_data


@st.cache_data
def load_corrected_results():
    return cr.load_results()


engineered_df, ts_data = load_all_data()
results = load_corrected_results()


# --- Main Application Tabbing Structure ---
homepage_tab, presentation_tab = st.tabs([
    "🏠 Project Homepage",
    "📊 Presentation",
])


# --- HOMEPAGE TAB ---
with homepage_tab:
    st.title("🐞 Pest Forecasting with Meteorological Data")
    st.header("Information Systems & Business Intelligence Final Project")
    st.markdown("A university team project that tried to predict insect catches in five field traps from daily weather and trap records (6 July to 23 August 2024).")
    st.markdown("---")
    st.header("📖 Project Overview and Result")
    cr.render_story(st, results)
    st.markdown("---")
    st.header("👨‍💻 Meet the Team")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Parham Khosh Solat")
        st.markdown("*ML Model Tournament Designer*")
        st.markdown("- Designed ML model tournaments architecture\n- Implemented advanced ML algorithms\n- Developed time series forecasting components\n- Led model evaluation methodology")
    with col2:
        st.subheader("Giovanni Montanile")
        st.markdown("*Business Intelligence Specialist*")
        st.markdown("- Developed Power BI dashboards\n- Created business intelligence solutions\n- Designed interactive visualizations\n- Implemented data reporting systems")
    with col3:
        st.subheader("Farshad Farahtaj")
        st.markdown("*Data Science Lead*")
        st.markdown("- Led data preprocessing workflow\n- Conducted exploratory data analysis\n- Performed feature engineering\n- Implemented data validation protocols")


# --- PRESENTATION TAB (WITH NESTED SUB-TABS) ---
with presentation_tab:
    st.title("Technical Presentation and Model Analysis")

    eda_sub_tab, results_sub_tab, withdrawn_sub_tab = st.tabs([
        "📊 Data Loading & EDA",
        "📏 Corrected Results",
        "⚠️ What Was Withdrawn",
    ])

    # --- SUB-TAB 1: DATA LOADING AND EDA ---
    with eda_sub_tab:
        st.header("🚦 Data Preprocessing & Exploratory Data Analysis (EDA)")
        if engineered_df is not None and ts_data is not None:
            # 1. Data Quality and Loading
            st.subheader("1. Data Quality and Loading")
            st.markdown("""
            Our analysis is built upon two distinct, clean datasets derived from the same source. Each is tailored for a specific modeling approach:
            - **Engineered Dataset (for ML Models)**: This granular dataset contains records for each trap location per day. It includes a rich set of engineered features designed to provide deep context for our classification and regression machine learning models.
            - **Time-Series Dataset (for Forecasting Models)**: This dataset aggregates daily data across all locations, creating a single time series for total insect counts and average weather conditions. It's specifically structured for time-series forecasting models like ARIMAX and Prophet.
            """)
            st.markdown("##### Engineered Dataset Preview")
            st.dataframe(engineered_df.head())
            st.markdown("##### Time-Series Dataset Preview")
            st.dataframe(ts_data.head())


            # 2. Data Cleaning and Preprocessing
            st.subheader("2. Data Cleaning & Feature Engineering")
            st.markdown("""
            To move beyond raw data, we engineered several new features to provide more powerful predictive signals for our models. This process creates context that the algorithms can learn from more effectively. Our key engineered variables include:
            - **`Temp_Range`**: The difference between the day's maximum and minimum temperatures. A wider range can indicate specific weather patterns that might influence insect behavior.
            - **`Temp_Avg_3d` & `Humidity_Avg_3d`**: 3-day rolling averages for temperature and humidity. These capture recent weather trends, as insect activity is often influenced by conditions over the past few days, not just a single day's snapshot.
            - **`Insects_Lag1` & `Insects_Lag3`**: The number of insects caught one and three days prior. This is crucial for capturing the cyclical nature and momentum of pest populations.
            - **`Recent_Activity`**: 1 if the same trap had a new catch in the previous 7 days. The original version of this column also counted the day itself, which leaked the target; the column shown here is the corrected one from `pipeline/features.py`.
            - **`Days_Since_Cleaning`**: Tracks the number of days since a trap was last serviced. The effectiveness of a trap can change over time due to saturation or bait degradation.
            - **`Season`**: Categorizes dates into distinct agricultural seasons (e.g., Early Summer, Mid Summer), allowing the model to learn broader seasonal patterns in pest activity.
            """)

            # 3. Exploratory Data Analysis (EDA)
            st.subheader("3. Exploratory Data Analysis (EDA)")
            st.markdown("In this section, we visually inspect the data to uncover patterns, identify relationships, and validate our assumptions.")
            
            st.markdown("#### Analyzing the Target Variable: Insect Counts")
            st.markdown("First, we examine the distribution of our primary target variable, `Number of insects`. The histogram shows that most days have very few captures, with a long tail of high-capture events, indicating that major pest activity is infrequent but significant.")
            utils_eda.create_target_variable_plots(engineered_df)
            
            st.markdown("The box plot breaks this down by location. Activity differs between traps, but with so few catches per trap the differences are rough.")
            st.markdown("#### Total Insects Caught by Location")
            st.markdown("The bar chart below shows the summed daily insect counts per trap over the whole period.")
            total_by_loc = engineered_df.groupby('Location')['Number of insects'].sum().sort_values(ascending=False).reset_index()
            fig_bar_loc = px.bar(
                total_by_loc, x='Location', y='Number of insects',
                title='Total Cumulative Insect Captures per Location',
                text='Number of insects', color='Location'
            )
            fig_bar_loc.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig_bar_loc.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig_bar_loc, use_container_width=True)


            st.markdown("#### Is Pest Activity Seasonal?")
            st.markdown("We investigate if there are broad seasonal trends in pest activity. The bar chart shows the average daily captures for each defined season.")
            seasonal_activity = engineered_df.groupby('Season')['Number of insects'].mean().reset_index()
            fig_season = px.bar(
                seasonal_activity, x='Season', y='Number of insects',
                title='Average Daily Insect Captures by Season',
                labels={'Number of insects': 'Average Daily Insects Caught'},
                color='Season', text='Number of insects'
            )
            fig_season.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_season, use_container_width=True)
            st.markdown("Interestingly, there is no dramatic difference in average daily captures between seasons. This suggests that while seasonality might play a role, more immediate factors like short-term weather changes are likely stronger drivers of day-to-day pest activity.")

            st.markdown("#### Distribution of Catch vs. No-Catch Days")
            n_catch_days = int((engineered_df['New catches'] > 0).sum())
            st.markdown(f"For our classification task, we care about whether *any* new insects were caught. Only {n_catch_days} of {len(engineered_df)} trap-days ({n_catch_days / len(engineered_df):.0%}) had a new catch, so catch days are rare.")
            df_copy_pie = engineered_df.copy()
            df_copy_pie['Catch Event'] = (df_copy_pie['New catches'] > 0).map({True: 'Catch (>=1 insect)', False: 'No Catch (0 insects)'})
            catch_counts = df_copy_pie['Catch Event'].value_counts()
            fig_pie = px.pie(
                values=catch_counts.values,
                names=catch_counts.index,
                title="Proportion of Days with and without New Pest Catches",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig_pie.update_traces(textinfo='percent+label', pull=[0, 0.1])
            st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("4. Correlation and Feature Analysis")
            st.markdown("Correlation matrices help us understand the linear relationships between variables. A value close to 1 (dark blue) indicates a strong positive correlation, while a value close to -1 (dark red) indicates a strong negative correlation.")
            
            st.markdown("##### Correlation Matrix (Engineered Dataset)")
            utils_eda.create_correlation_plots(engineered_df)

            st.markdown("##### Correlation Matrix (Time-Series Dataset)")
            st.markdown("The time-series data shows a moderate positive correlation between temperature and insect captures, and a weaker one with humidity.")
            ts_numeric_cols = ts_data.select_dtypes(include=np.number).columns
            ts_corr_matrix = ts_data[ts_numeric_cols].corr(numeric_only=True)
            fig_corr_ts = px.imshow(ts_corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig_corr_ts, use_container_width=True)

            st.markdown("##### Top Features Correlated with Insect Counts")
            st.markdown("This chart isolates the features with the strongest absolute correlation to `Number of insects`. Correlation on the full data set does not show that a feature can forecast a later catch; see the corrected results for that.")
            numeric_cols_corr = engineered_df.select_dtypes(include=np.number).columns
            corr_matrix_top10 = engineered_df[numeric_cols_corr].corr(numeric_only=True)
            target_column_corr = 'Number of insects'
            if target_column_corr in corr_matrix_top10:
                corr_target = corr_matrix_top10[[target_column_corr]].drop(target_column_corr).sort_values(by=target_column_corr, ascending=False)
                corr_target_abs = corr_target.abs()
                top_10 = corr_target_abs.head(10)
                fig_top10 = px.bar(
                    top_10, x=top_10.index, y=target_column_corr,
                    title=f'Top 10 Features Correlated with "{target_column_corr}"',
                    labels={'x': 'Feature', 'y': 'Absolute Correlation'}, color=top_10.index
                )
                st.plotly_chart(fig_top10, use_container_width=True)

            st.subheader("5. Weather Analysis")
            st.markdown("Here, we visualize the relationship between key weather variables and insect captures more directly.")

            st.markdown("##### Temperature vs. Insect Activity")
            st.markdown("This scatter plot shows that higher insect counts tend to occur when the average temperature is between 20°C and 30°C. Very high or very low temperatures seem to suppress activity.")
            fig_temp = px.scatter(engineered_df, x='Average Temperature', y='Number of insects', color='Number of insects', color_continuous_scale='Viridis',
                                  title='Insect Activity by Average Daily Temperature', trendline="lowess", trendline_color_override="red")
            st.plotly_chart(fig_temp, use_container_width=True)
            
            st.markdown("##### Humidity vs. Insect Activity")
            st.markdown("The relationship with humidity is less defined, though there's a slight tendency for higher captures to occur in the 60-80% humidity range.")
            fig_humidity = px.scatter(engineered_df, x='Average Humidity', y='Number of insects', color='Number of insects', color_continuous_scale='Plasma',
                                    title='Insect Activity by Average Daily Humidity', trendline="lowess", trendline_color_override="blue")
            st.plotly_chart(fig_humidity, use_container_width=True)

            st.markdown("##### 3D View: Weather vs. Insect Activity")
            st.markdown("This 3D plot combines temperature, humidity, and insect counts. The highest peaks (largest insect captures) are concentrated in the middle ranges of both temperature and humidity, confirming our observations from the 2D plots.")
            fig_3d = px.scatter_3d(
                engineered_df, x='Average Temperature', y='Average Humidity', z='Number of insects',
                color='Number of insects', title='3D Relationship between Weather and Insect Captures',
                labels={'Average Temperature': 'Avg. Temp (°C)', 'Average Humidity': 'Avg. Humidity (%)', 'Number of insects': 'Insect Count'},
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig_3d.update_traces(marker=dict(size=3, opacity=0.8))
            st.plotly_chart(fig_3d, use_container_width=True)

            st.markdown("##### Interactive Weather Conditions by Location")
            st.markdown("Use the dropdown below to explore the specific temperature and humidity trends over time for each trap location.")
            locations_weather = engineered_df['Location'].unique()
            selected_loc_weather = st.selectbox("Select a Location to View Weather Data", options=locations_weather, key="weather_loc_select")

            loc_df_weather = engineered_df[engineered_df['Location'] == selected_loc_weather].copy()
            loc_df_weather = loc_df_weather.sort_values('Date')

            fig_weather = go.Figure()
            fig_weather.add_trace(go.Scatter(x=loc_df_weather['Date'], y=loc_df_weather['Average Temperature'], mode='lines', name='Avg Temp (°C)', line=dict(color='red')))
            fig_weather.add_trace(go.Scatter(x=loc_df_weather['Date'], y=loc_df_weather['Average Humidity'], mode='lines', name='Avg Humidity (%)', line=dict(color='blue'), yaxis='y2'))
            fig_weather.update_layout(
                title=f'Weather Conditions Over Time for {selected_loc_weather}', xaxis_title='Date', yaxis_title='Average Temperature (°C)',
                yaxis=dict(titlefont=dict(color='red'), tickfont=dict(color='red')),
                yaxis2=dict(title='Average Humidity (%)', overlaying='y', side='right', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
                legend=dict(x=0, y=1.2, orientation="h")
            )
            st.plotly_chart(fig_weather, use_container_width=True)
        else:
            st.error("Data could not be loaded. Please check your CSV files.")

    # --- SUB-TAB 2: CORRECTED RESULTS ---
    with results_sub_tab:
        st.header("📏 Corrected Results")
        st.markdown("All numbers below are read from `pipeline/results/results.json`, written by `pipeline/evaluate.py`.")
        cr.render_method(st, results)
        st.subheader("Will a trap record a new catch today? (classification)")
        cr.render_classification(st, px, results)
        st.subheader("How many insects are in the trap today? (regression)")
        cr.render_regression(st, px, results)
        st.success(results["conclusion"]["summary"])
        cr.render_prediction_note(st)

    # --- SUB-TAB 3: WHAT WAS WITHDRAWN ---
    with withdrawn_sub_tab:
        st.header("⚠️ What Was Withdrawn, and Why")
        cr.render_withdrawn(st)
