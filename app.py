import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from signal_decoder import SignalDecoder
from utils import load_signal, plot_results, calculate_rolling_sharpe, plot_single_asset_results
from yre.data import DailyDB
import calendar

st.set_page_config(page_title="Signal Decoder", layout="wide")

st.title("Signal Decoder")

# Initialize session state to store the database and signal
if 'db' not in st.session_state:
    st.session_state.db = None
if 'signal' not in st.session_state:
    st.session_state.signal = None

# Function to set the database
def set_database(db: DailyDB):
    st.session_state.db = db

# File uploader for signal
# signal_file = st.file_uploader("Upload signal file (CSV)", type="csv")
signal_file = 'signal.csv'

# Initialize database
try:
    db = DailyDB('US').clip_date('2023-01-01')
    set_database(db)
except Exception as e:
    st.error(f"Error initializing database: {str(e)}")
    st.stop()
    
# Load signal data
signal = load_signal(signal_file)
st.session_state.signal = signal

# Initialize SignalDecoder with the correct DataFrame from DailyDB
decoder = SignalDecoder(st.session_state.db, signal) 
if signal_file and st.session_state.db is not None:

    
    # Sidebar for analysis options
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Overview", "Sector", "Calendar", "Single Asset", "Shift Analysis"]
    )
    
    if analysis_type == "Overview":
        st.subheader("Overview")
        long_pnl, short_pnl, total_pnl, correlation, metrics = decoder.decode_long_short(signal)
        plot_results(long_pnl, short_pnl, total_pnl, correlation)
            
        st.write("#### Performance Statistics")
        st.table(metrics.style.format("{:.4f}"))
        
        # Add hedge return analysis
        st.write("#### PnL from Hedge Return Analysis")
        h_long_pnl, h_short_pnl, h_total_pnl, h_correlation, h_metrics = decoder.decode_hedge_ret(signal)
        plot_results(h_long_pnl, h_short_pnl, h_total_pnl, h_correlation, title_prefix="Hedge Return")
        
        st.write("##### Hedge Return Performance Statistics")
        st.table(h_metrics.style.format("{:.4f}"))
        
        # Compare residual return and hedge return
        st.write("##### Residual vs Hedge")
        comparison_df = pd.DataFrame({
            'Metric': metrics.index,
            'Residual': metrics['Sharpe Ratio'],
            'Hedge': h_metrics['Sharpe Ratio']
        }).set_index('Metric')
        
        st.table(comparison_df.style.format("{:.4f}"))
        
        # Visualize the comparison including sharpe and pnl 
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        comparison_df.plot(kind='bar', ax=ax[0])
        ax[0].set_title('Residual vs Hedge')
        ax[0].set_xlabel('')
        ax[0].set_ylabel('Sharpe Ratio')
        ax[0].legend(['Residual', 'Hedge'])
        ax[0].tick_params(axis='x', rotation=45)

        # Plot PnL
        pnl_comparison = pd.DataFrame({
            "Metric": metrics.index,
            'Residual': metrics['Avg Daily PnL'],
            'Hedge': h_metrics['Avg Daily PnL']
        }).set_index('Metric')
        pnl_comparison.plot(kind='bar', ax=ax[1], rot=45)
        ax[1].set_ylabel('Avg Daily PnL')
        ax[1].set_xlabel('')
        ax[1].legend(['Residual', 'Hedge'])
        plt.tight_layout()
        st.pyplot(fig)

    elif analysis_type == "Sector":
        st.subheader("Sector Analysis")
        sector_names = st.session_state.db['u'].columns  # Assuming 'u' contains sector information
        sector_names = [i for i in sector_names if 'sector' in i.lower()]
        sector_name = st.selectbox("Select Sector Type", sector_names)
        sectors = st.session_state.db['u'][sector_name].unique()
        sectors = np.append(sectors, 'All')
        sector = st.selectbox("Select Sector", sectors)
        if sector == 'All':
            tabs = {}
            # Show all sectors results
            for sector in sectors:
                if sector == 'All':
                    continue
                # st.write(f"### {sector}")
                lpnl, spnl, tpnl, corr, metrics = decoder.decode_sector(sector_name, sector)
                tabs[sector] = metrics
            tabs = pd.concat(tabs)
            fig, axes = plt.subplots(len(tabs.columns), 1, figsize=(18, 20))    
            for i, col in enumerate(tabs.columns):
                _sub_tab = tabs[col]
                _sub_tab.index.names = ['Sector', 'PtfSide']
                _sub_tab = _sub_tab.reset_index()
                # plot vertical bar plots
                sns.barplot(x='Sector', y=col, hue='PtfSide', data=_sub_tab, ax=axes[i])
                axes[i].set_title(f'{col}')
                axes[i].set_xlabel('')
                axes[i].legend(fontsize=10)
                axes[i].tick_params(axis='both', which='major', labelsize=10, rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
                
            # plot tabs based on the summary results

        else:
            lpnl, spnl, tpnl, corr, metrics = decoder.decode_sector(sector_name, sector)
            plot_results(lpnl, spnl, tpnl, corr)

            st.write("#### Performance Statistics")
            st.table(metrics.style.format("{:.4f}"))
            
    elif analysis_type == "Calendar":
        st.subheader("Calendar Analysis")
        by = st.radio("Analyze by", ["day_of_week", "month", "quarter"])
        calendar_results = decoder.decode_calender(by=by)
        
        # Create DataFrames to store the results
        results_df = pd.DataFrame(columns=['Period', 'Total PnL', 'Sharpe Ratio', 'Daily Ret', 'Volatility'])
        metrics_df = pd.DataFrame()
        
        for period, data in calendar_results.items():
            total_pnl = data['total_pnl'].sum(axis=1).sum()
            sharpe_ratio = data['metrics'].loc['Total', 'Sharpe Ratio']
            daily_pnl = data['metrics'].loc['Total', 'Avg Daily PnL']
            daily_ret = data['metrics'].loc['Total', 'Avg Daily Return']
            vol = data['metrics'].loc['Total', 'Volatility']
            
            # Map period to name
            if by == 'day_of_week':
                period_name = list(calendar.day_name)[int(period)]
            elif by == 'month':
                period_name = list(calendar.month_name)[int(period)]
            elif by == 'quarter':
                period_name = f"Q{period}"
            else:
                period_name = str(period)
            
            results_df = pd.concat([results_df, pd.DataFrame({
                'Period': period_name,
                'Total PnL': total_pnl,
                'Sharpe Ratio': sharpe_ratio,
                'Daily Ret': daily_ret,
                'Volatility': vol
            }, index=[0])], ignore_index=True)
            
            # Add metrics to metrics_df
            period_metrics = data['metrics'].copy()
            period_metrics.index = [f"{period_name} - {idx}" for idx in period_metrics.index]
            metrics_df = pd.concat([metrics_df, period_metrics])
        
        # Sort the results
        results_df = results_df.sort_values('Total PnL', ascending=False)
        
        # Display the summary results
        st.write("#### Summary Results")
        st.write(results_df.style.format({
            'Total PnL': '{:.2f}',
            'Sharpe Ratio': '{:.2f}',
            'Correlation': '{:.2f}'
        }))
        
        # Plot the summary results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        results_df.plot(x='Period', y='Total PnL', kind='bar', ax=ax1)
        ax1.set_title(f'Total PnL by {by.capitalize()}')
        ax1.set_ylabel('Total PnL')
        ax1.tick_params(axis='x', rotation=45)
        
        results_df.plot(x='Period', y='Sharpe Ratio', kind='bar', ax=ax2)
        ax2.set_title(f'Sharpe Ratio by {by.capitalize()}')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display detailed metrics
        st.write("#### Detailed Metrics")
        st.write(metrics_df.style.format({
            'Avg Daily PnL': '{:.2f}',
            'Avg Daily Return': '{:.2%}',
            'Volatility': '{:.2%}',
            'Sharpe Ratio': '{:.2f}',
            'Max Drawdown (%)': '{:.2f}%'
        }))


    elif analysis_type == "Single Asset":
        st.subheader("Single Asset Analysis")
        asset = st.selectbox("Select Asset", signal.columns)
        if signal[asset].abs().sum() == 0:
            st.warning("This asset has no signal, please select another asset.")
            st.stop()
        asset_results = decoder.decode_single_asset(asset)
        
        # Display contribution
        st.write(f"Contribution of {asset} to total PnL: {asset_results['contribution']:.2%}")
        
        # Plot results
        plot_single_asset_results(asset_results, asset)
        
        # Display metrics
        metrics_df = pd.concat({
            asset: asset_results['target_metrics'],
            'Total': asset_results['total_metrics'],
            'Rest': asset_results['rest_metrics']
        })
        
        st.write("#### Performance Metrics")
        st.table(metrics_df.style.format("{:.4f}"))
        
        # Display correlation
        correlation = asset_results['target_pnl'].corr(asset_results['rest_pnl'])
        st.write(f"Correlation between {asset} and the rest of the portfolio: {correlation:.4f}")
        
        # Display top correlated assets
        st.write("#### Top Correlated Assets")
        correlations = asset_results['pnl_by_stock'].corr()[asset].sort_values(ascending=False)
        top_correlated = correlations[correlations.index != asset].head(10)
        st.bar_chart(top_correlated)

    elif analysis_type == "Shift Analysis":
        st.subheader("Shift Analysis")
        n = st.slider("Maximum number of days to shift", 1, 20, 10)
        shifted_res, peak_metrics = decoder.shift_analysis(n)
        
        # Plot results
        fig, axes = plt.subplots(2, 1, figsize=(20, 8))
        
        # Sharpe Ratio plot
        shifted_res[['total_sharpe', 'long_sharpe', 'short_sharpe']].plot(ax=axes[0], marker='o')
        axes[0].set_title('Sharpe Ratio vs Shift Days')
        axes[0].set_xlabel('Shift Days')
        axes[0].set_ylabel('Sharpe Ratio')
        axes[0].legend(['Total', 'Long', 'Short'])
        
        # Average Daily Return plot
        shifted_res[['total_return', 'long_return', 'short_return']].plot(ax=axes[1], marker='o')
        axes[1].set_title('Average Daily Return vs Shift Days')
        axes[1].set_xlabel('Shift Days')
        axes[1].set_ylabel('Average Daily Return')
        axes[1].legend(['Total', 'Long', 'Short'])
        
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display summary statistics
        st.write("#### Peak Metrics")
        peak_data = []
        for key, value in peak_metrics.items():
            if isinstance(value, tuple):
                peak_data.append({"Metric": key, "Days": value[0], "Value": f"{value[1]:.4f}"})
            else:
                peak_data.append({"Metric": key, "Days": value, "Value": "-"})
        
        peak_df = pd.DataFrame(peak_data)
        st.table(peak_df.set_index("Metric"))
        
        # Display the shift results table
        st.write("#### Shift Analysis Results")
        st.dataframe(shifted_res.style.format({
            'total_sharpe': '{:.4f}',
            'long_sharpe': '{:.4f}',
            'short_sharpe': '{:.4f}',
            'total_return': '{:.6f}',
            'long_return': '{:.6f}',
            'short_return': '{:.6f}',
            'correlation': '{:.4f}',
            'max_drawdown': '{:.2f}%'
        }))


else:
    if st.session_state.db is None:
        st.warning("Please initialize the DailyDB instance before running the app.")
    if signal_file is None:
        st.warning("Please upload the signal file (CSV) to begin analysis.")

# Note: You need to call set_database(db) with your DailyDB instance before running the Streamlit app
