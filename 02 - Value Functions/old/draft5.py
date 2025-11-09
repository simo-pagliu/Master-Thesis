# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import curve_fit

# Session state for data and navigation
if 'df' not in st.session_state:
    st.session_state.df = None
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'points' not in st.session_state:
    st.session_state.points = []

# Step 1: Load CSV
if st.session_state.step == 0:
    uploaded_file = st.file_uploader("Upload CSV")
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.step = 1

# Step 2: Elicit points
if st.session_state.step == 1 and st.session_state.df is not None:
    attr = st.session_state.df.iloc[0]
    st.write(f"Attribute: {attr['name']} ({attr['min']} to {attr['max']})")
    peak = st.number_input("Peak value", min_value=attr['min'], max_value=attr['max'])
    if st.button("Add Point"):
        st.session_state.points.append((peak, 1.0))
        fig = go.Figure(data=go.Scatter(x=[p[0] for p in st.session_state.points],
                                        y=[p[1] for p in st.session_state.points]))
        st.plotly_chart(fig)
    if st.button("Next"):
        st.session_state.step = 2

# Step 3: Fit curve
if st.session_state.step == 2:
    lambda_param = st.slider("Polynomial order", 1, 5, 2)
    x = np.array([p[0] for p in st.session_state.points])
    y = np.array([p[1] for p in st.session_state.points])
    coeffs = np.polyfit(x, y, deg=lambda_param)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = np.polyval(coeffs, x_fit)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines'))
    st.plotly_chart(fig)
    if st.button("Back"):
        st.session_state.step = 1
