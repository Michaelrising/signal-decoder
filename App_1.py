import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Evaluator import Evaluator
from utils import load_signal, create_calendar_plot
from yre.data import DailyDB

st.set_page_config(page_title="Signal Evaluator", layout="wide", page_icon="🧊")

@st.cache_resource
def get_database():
    return DailyDB('US').clip_date('2020-01-01')

@st.cache_data
def get_signal(file_path):
    return load_signal(file_path)

@st.cache_resource
def get_evaluator(_db, signal):
    return Evaluator(save_path='./results', db=_db, signal=signal)

@st.cache_data
def run_overview(_evaluator):
    return _evaluator.Overview()

@st.cache_data
def run_ls_decoder(_evaluator):
    return _evaluator.LS_Decoder()

@st.cache_data
def run_sector_decoder(_evaluator, selected_sector):
    return _evaluator.Sector_Decoder(selected_sector)

@st.cache_data
def run_asset_decoder(_evaluator, topk, btmk):
    return _evaluator.Asset_Decoder(topk, btmk)

@st.cache_data
def run_calendar_decoder(_evaluator, by):
    return _evaluator.Calendar_Decoder(by)

@st.cache_data
def run_shift_decoder(_evaluator, n):
    return _evaluator.Shift_Decoder(n)

@st.cache_data()
def run_mktcap_decoder(_evaluator, q0, q1):
    return _evaluator.MktCap_Decoder(q0, q1)

def main():
    st.title("Signal Evaluator")

    # Initialize database and load signal
    db = get_database()
    signal = get_signal('signal.csv')

    # Initialize Evaluator
    evaluator = get_evaluator(db, signal)

    st.header("Overview")
    top_long_names, top_short_names, lmv, smv = run_overview(evaluator)
    
    # User input for percentage
    pct = st.number_input("Enter the percentage for estimating capacity (e.g., 0.02 for 2%)", min_value=0.0, max_value=0.1, value=0.02, step=0.01)
    
    # Estimate strategy capacity
    est_cap = evaluator.Estimate_Cap(pct=pct)
    st.write(f"#### Estimated Strategy Capacity: {est_cap:.2f} Million")

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("#### Top 10 Long Equities")
        st.dataframe(top_long_names)
    with col2:
        st.write("#### Top 10 Short Equities")
        st.dataframe(top_short_names)
    with col3:
        st.write("#### Overall Performance Metrics")
        metrics, fig = run_ls_decoder(evaluator)
        st.dataframe(metrics.style.format("{:.4f}"))

    # Long-Short Decoder
    st.header("Long-Short Decoder")
    st.plotly_chart(fig, use_container_width=True)

    # Sector Decoder
    st.header("Sector Decoder")
    sector_columns = ['gics_ind', 'gics_sector', 'bics_sector', 'bics_sector_sm', 'bics_samart_sector', 'lse_sector']
    selected_sector = st.selectbox("Choose Sector Classification", sector_columns)
    pie_chart, time_series, metrics = run_sector_decoder(evaluator, selected_sector)
    metrics.index.names = ['sector', 'side']
    metrics = metrics.query('side == "Total"').droplevel('side')
    st.dataframe(metrics.style.format("{:.4f}"))
    st.plotly_chart(pie_chart, use_container_width=True)
    st.plotly_chart(time_series, use_container_width=True)

    # Asset Decoder
    st.header("Asset Decoder")
    fig, comparison = run_asset_decoder(evaluator, topk=10, btmk=10)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Metrics Comparison")
    st.dataframe(comparison.style.format({
        'All Assets': '{:.4f}',
        'Without Top 10': '{:.4f}',
        'Without Bottom 10': '{:.4f}',
        '% Change (No Top)': '{:.2f}%',
        '% Change (No Bottom)': '{:.2f}%'
    }))

    # Calendar Decoder
    st.header("Calendar Decoder")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Day of Week Analysis")
        metrics = run_calendar_decoder(evaluator, by='dow')
        fig = create_calendar_plot(metrics[0], 'Mean PnL and Sharpe by Day of Week', 'Day of Week')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(metrics[0].style.format("{:.4f}"))

    metrics = run_calendar_decoder(evaluator, by='dom')
    with col2:
        st.subheader("Day of Month Analysis: look Forward")
        fig = create_calendar_plot(metrics[0], 'Mean PnL and Sharpe by Day of Month', 'Day of Month')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(metrics[0].style.format("{:.4f}"))

    with col3:
        st.subheader("Day of Month Analysis: Look Backward")
        fig = create_calendar_plot(metrics[1], 'Mean PnL and Sharpe by Day of Month', 'Day of Month')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(metrics[1].style.format("{:.4f}"))

    # Shift Decoder
    st.header("Shift Decoder")
    fig = run_shift_decoder(evaluator, n=10)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    This analysis shows the daily return and sharpe changes by shifting the signal.
    """)

    # Market Cap Decoder
    st.header("Market Cap Decoder")
    q0 = st.number_input("q0", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    q1 = st.number_input("q1", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
    fig = run_mktcap_decoder(evaluator, q0=q0, q1=q1)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    This analysis shows the contribution and distribution of PnLs across different market cap groups:
    - The bar chart on the left shows the total PnL contribution of each market cap group.
    - The box plot on the right shows the distribution of daily PnLs for each market cap group.
    """)

if __name__ == "__main__":
    main()