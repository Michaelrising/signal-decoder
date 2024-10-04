import pandas as pd
import numpy as np
from typing import Union, Literal, Tuple, Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from yre.data import DailyDB, IntraDB, RateDB
from yre.sim import Actual
from pathlib import Path
import plotly.colors as pc
from yre.sim.helper import perf_by_dom, perf_by_dow
from utils import calhalfic
import bottleneck as bn
from yin.common import mat_util, to_array

class Evaluator:
    def __init__(self, 
                 save_path: str,
                 db: Union[DailyDB, IntraDB, RateDB], 
                 signal: pd.DataFrame,
                 actual = None):
        self.db = db
        self.signal = signal.reindex(db['u'].index, axis=1) # type: ignore
        self.signal.index.name = 'date'
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        # if actual is None:
        #     self.run_sim('dsim')
        # else:
        #     self.actual = actual
    
    def run_sim(self, method: str):
        if method.lower() == 'dsim':
            from yre.sim import DSim
            self.actual = DSim(self.signal,
                                self.db,
                                hedge_inst = self.db['hedge_inst'])
        else:
            pass

    def Overview(self):
        signals = self.signal.copy()
        lmv = signals.where(signals > 0)
        smv = -1 * signals.where(signals < 0)
        gmv = lmv + smv
        top_long_names = lmv.sum(axis=0).sort_values(ascending=False).head(10)
        top_short_names = smv.sum(axis=0).sort_values(ascending=False).head(10)
        top_long_names = top_long_names.rename('Capital')
        top_short_names = top_short_names.rename('Capital')

        return top_long_names, top_short_names, lmv.sum().sum(), smv.sum().sum()

    def LS_Decoder(self, signals=None) -> Tuple[pd.DataFrame, go.Figure]:
        if signals is None:
            signals = self.signal.copy()
        pnls = self._get_pnls(signals)
        metrics = self._get_metrics(pnls, signals)
        
        # Calculate daily PnLs
        long_pnl = pnls['long'].sum(axis=1)
        short_pnl = pnls['short'].sum(axis=1)
        total_pnl = pnls['total'].sum(axis=1)
        hedge_total = pnls['hedge_total'].sum(axis=1)
        lmv = signals.where(signals > 0).sum(axis=1)
        smv = -1 * signals.where(signals < 0).sum(axis=1)
        gmv = lmv + smv
        l_contrib = long_pnl / gmv
        s_contrib = short_pnl / gmv
        h_contrib = hedge_total / gmv

        # Create a DataFrame for contributions
        contributions = pd.DataFrame({
            'Long': l_contrib,
            'Short': s_contrib,
            'Hedge': h_contrib
        })

        # Create subplot figure
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Distribution of Long/Short Contributions', 
                                                            '30-Day Rolling Average of Long/Short Contributions'))
        # Visualization 1: Histogram
        fig.add_trace(
            go.Histogram(x=contributions['Long'], name='Long', opacity=0.7, marker_color='green'),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=contributions['Short'], name='Short', opacity=0.7, marker_color='red'),
            row=1, col=1
        )
        # fig.add_trace(
        #     go.Histogram(x=contributions['Hedge'], name='Hedge', opacity=0.7, marker_color='blue'),
        #     row=1, col=1
        # )

        # Visualization 2: Time Series with Rolling Window
        window = 30  # 30-day rolling window
        rolling_contributions = contributions.sort_index().rolling(window=window).mean()

        fig.add_trace(
            go.Scatter(x=rolling_contributions.index, y=rolling_contributions['Long'], 
                       name='Long', fill='tozeroy', fillcolor='rgba(0,255,0,0.1)', line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=rolling_contributions.index, y=rolling_contributions['Short'], 
                       name='Short', fill='tozeroy', fillcolor='rgba(255,0,0,0.1)', line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=rolling_contributions.index, y=rolling_contributions['Hedge'], 
                       name='Hedge', fill='tozeroy', fillcolor='rgba(0,0,255,0.1)', line=dict(color='blue')),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            height=800, 
            title_text="Long/Short Contributions Analysis",
            showlegend=True
        )

        # Update x-axis and y-axis labels
        fig.update_xaxes(title_text="Contribution", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Contribution", row=2, col=1)

        # Add vertical line at x=0 for histogram
        fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, yref="paper", xref="x",
                      line=dict(color="black", width=2, dash="dash"), row=1, col=1)

        # Save the interactive plot
        fig.write_html(self.save_path / 'long_short_decoding_interactive.html')

        return metrics, fig
    
    def Sector_Decoder(self, sector_name: str, topk: int=10) -> Tuple[go.Figure, go.Figure, pd.DataFrame]:
        sector_metrics, sector_pnls, total_contribution = self._decode_by_group(sector_name)
        topk_contribution = total_contribution[:topk]
        other = ('Others', sum([item[1] for item in total_contribution]) - sum([item[1] for item in total_contribution[:topk]]))
        topk_contribution.append(other)
        
        # sector_pnls['Others'] = sector_pnls.sum(axis=1) - sector_pnls[total_contribution[0][0]]
        pie_chart = go.Figure(data=[go.Pie(
            labels=[item[0] for item in topk_contribution], 
            values=[item[1] for item in topk_contribution],
            textposition='inside',
            textinfo='percent+label',
            insidetextorientation='radial'
        )])
        pie_chart.update_layout(
            title_text=f"Top {topk} {sector_name} Contribution",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Create time series plot
        time_series = go.Figure()
        for sector, _ in topk_contribution:
            if sector == 'Others':
                continue
            pnl = sector_pnls[sector].sum(axis=1)
            time_series.add_trace(go.Scatter(x=pnl.index, y=pnl.cumsum(), 
                                             mode='lines', name=sector))
        time_series.update_layout(title_text=f"Top {topk} cumulative PnL by {sector_name}",
                                  xaxis_title="Date",
                                  yaxis_title="Cumulative PnL")
        sector_metrics = pd.concat(sector_metrics)
        return pie_chart, time_series, sector_metrics

    def Asset_Decoder(self, topk: int=10, btmk: int=10):
        _, asset_pnls, total_contribution = self._decode_by_group(pd.Series(self.signal.columns, index=self.signal.columns))
        topk_contribution = total_contribution[:topk]
        btmk_contribution = total_contribution[-btmk:]

        fig = go.Figure()

        # Generate color scales
        greens = pc.n_colors('rgb(255,255,255)', 'rgb(0,255,0)', topk, colortype='rgb')
        reds = pc.n_colors('rgb(255,255,255)', 'rgb(255,0,0)', btmk, colortype='rgb')

        # Plot top contributors
        for i, (asset, _) in enumerate(reversed(topk_contribution)):
            pnl = asset_pnls[asset].sum(axis=1)
            fig.add_trace(go.Scatter(
                x=pnl.index, 
                y=pnl.cumsum(),
                mode='lines', 
                name=asset,
                line=dict(color=greens[i]),  # type: ignore
                legendgroup='top'
            ))

        # Plot bottom contributors
        for i, (asset, _) in enumerate(btmk_contribution):
            pnl = asset_pnls[asset].sum(axis=1)
            fig.add_trace(go.Scatter(
                x=pnl.index, 
                y=pnl.cumsum(),
                mode='lines', 
                name=asset,
                line=dict(color=reds[i]),  # type: ignore
                legendgroup='bottom'
            ))

        fig.update_layout(
            title_text=f"Top {topk} and Bottom {btmk} Assets cumulative PnL",
            xaxis_title="Date",
            yaxis_title="Cumulative PnL",
            legend_title_text="Assets",
            legend=dict(
                groupclick="toggleitem",
                tracegroupgap=10
            )
        )
        # if we romove the topk or btmk names, how the metrics will change
        topk_names = [item[0] for item in topk_contribution]
        btmk_names = [item[0] for item in btmk_contribution]
        pnls = self._get_pnls(self.signal)
        metrics = self._get_metrics(pnls, self.signal, pd.DataFrame(True, index=self.signal.index, columns=self.signal.columns))
        topk_mask = pd.DataFrame(True, index=self.signal.index, columns=self.signal.columns)
        topk_mask[topk_names] = False
        btmk_mask = pd.DataFrame(True, index=self.signal.index, columns=self.signal.columns)
        btmk_mask[btmk_names] = False
        metrics_without_top = self._get_metrics(pnls, self.signal, mask=topk_mask) 
        metrics_without_bottom = self._get_metrics(pnls, self.signal, mask=btmk_mask) 

        comparison = pd.DataFrame({
            'All Assets': metrics.loc['Total'],
            f'W/O Top {topk}': metrics_without_top.loc['Total'],
            f'W/O Bottom {btmk}': metrics_without_bottom.loc['Total']
        })
        comparison['% Change (No Top)'] = (comparison[f'W/O Top {topk}'] - comparison['All Assets']) / comparison['All Assets'] * 100
        comparison['% Change (No Bottom)'] = (comparison[f'W/O Bottom {btmk}'] - comparison['All Assets']) / comparison['All Assets'] * 100

        # Create the cumulative PnL plot
        # fig1 = go.Figure()
        return fig, comparison
    
    def Calendar_Decoder(self, by: str='dow'):
        pnls = self._get_pnls(self.signal)
        pnl = pnls['total'].sum(axis=1)
        if by == 'dow':
            metrics = perf_by_dow(pnl=pnl)
            metrics = (metrics.droplevel(-1), )
        elif by == 'dom':
            metrics = perf_by_dom(pnl=pnl)
            metrics = (metrics[0].droplevel(-1), metrics[1].droplevel(-1))
        else:
            raise ValueError("Invalid 'by' parameter. Choose 'dow' or 'dom'.")

        return metrics
    
    def Shift_Decoder(self, n: int=10):
        _shifted_res = {}
        for i in range(0, n+1):  # Only positive shifts
            shifted_sigs = self.signal.shift(i).copy()
            pnls = self._get_pnls(shifted_sigs)
            metrics = self._get_metrics(pnls, shifted_sigs)
            _shifted_res[i] = metrics
        # plot metrics: including sharpe, mean return
        fig = go.Figure()
        _shifted_res = pd.concat(_shifted_res)
        _shifted_res = _shifted_res.xs('Total', level=-1)
        _shifted_res.index.name = 'shift_days'
        # calculate half life of the sharpe and mean return 
        
        sharpe_half_life = n / np.log2(_shifted_res['Sharpe'].iloc[0] / _shifted_res['Sharpe'].iloc[-1])
        # mean_return_half_life = np.log(2) / np.log(1 + _shifted_res['Avg Daily Return'].iloc[0] / _shifted_res['Avg Daily Return'].iloc[-1])
        fig.add_trace(go.Bar(x=_shifted_res.index.get_level_values('shift_days'), 
                            y=_shifted_res['Avg Daily Return'], width=0.5,name='Mean Daily Return', marker_color='lightblue', yaxis='y'))
        fig.add_trace(go.Scatter(x=_shifted_res.index.get_level_values('shift_days'), 
                                 y=_shifted_res['Sharpe'], mode='lines', name='Sharpe', line=dict(color='orange'), yaxis='y2'))

        fig.update_layout(
            title='Sharpe and Mean Return by Shift: HalfLife {}'.format(np.ceil(sharpe_half_life)),
            xaxis=dict(title='Shift Days'),
            yaxis=dict(title='Mean Daily Return'),
            yaxis2=dict(title='Sharpe', overlaying='y', side='right', showgrid=False), 
            legend=dict(x=0.85, y=0.99, bgcolor='rgba(255,255,255,0.5)')
        )

        return fig
    
    def MktCap_Decoder(self, q0: float=0.1, q1: float=0.9):
        pnls = self._get_pnls(self.signal)
        mktcap: pd.DataFrame = self.db['mkt_cap_usd'] # type: ignore
        last_mktcap = mktcap[self.signal.columns].loc[self.signal.index.max()]
        mktcap_q0 = last_mktcap.quantile(q0)
        mktcap_q1 = last_mktcap.quantile(q1)
        # cut the mktcap into three groups
        mktcap_group = pd.cut(last_mktcap.dropna(), bins=[0, mktcap_q0, mktcap_q1, np.inf], labels=['Small', 'Medium', 'Large']) # type: ignore
        
        # Calculate PnLs for each group
        total_pnl = pnls['total']#.sum(axis=1)
        grouped_pnls = {group: total_pnl[mktcap_group[mktcap_group == group].index] for group in mktcap_group.unique()}
        
        # Calculate total contribution for each group
        total_contribution = {group: pnl.sum().sum() for group, pnl in grouped_pnls.items() if not pnl.empty}
        
        # Create subplot figure
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Market Cap Contribution', 'PnL Distribution by Market Cap'))
        
        # Bar chart
        keys = ['Small', 'Medium', 'Large']
        fig.add_trace(
            go.Bar(
                x=keys,
                y=[total_contribution[key] for key in keys],
                width=0.5,
                name='Total Contribution',
                opacity=0.6
            ),
            row=1, col=1
        )
        
        # Box plot
        for group in keys:
            pnl = grouped_pnls[group]
            fig.add_trace(
                go.Box(y=pnl.sum(axis=1).values.flatten(), name=group, boxpoints=False, jitter=0.3, pointpos=-1.8),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Market Cap Analysis",
            height=500,
            width=1000,
            showlegend=False
        )
        
        # Update y-axis 
        fig.update_yaxes(title_text="Total PnL", row=1, col=1)
        fig.update_yaxes(title_text="Daily PnL", row=1, col=2)
        
        return fig
    
    def Estimate_Cap(self, pct: float=0.02, vol=None):
        sigs = self.signal.copy()
        mkt_tvr = self.db['tvr_usd']
        mkt_tvr = mkt_tvr.reindex(sigs.index, axis=0).reindex(sigs.columns, axis=1).values # type: ignore
        net_sigs = sigs.abs().groupby(sigs.index.normalize()).sum() # type: ignore
        net_sigs_abs = net_sigs.abs().values
        adv_win_start = (np.arange(len(net_sigs))[bn.nansum(net_sigs_abs, axis=1) > mat_util.epsilon])[0]
        if vol is None:
            tvr = bn.nansum(net_sigs_abs[adv_win_start:])
        else:
            net_sigs_abs = net_sigs_abs * vol.reindex_like(net_sigs).values
            tvr = bn.nansum(net_sigs_abs[adv_win_start:])
        est_cap = (bn.nansum(np.maximum(pct, np.minimum(pct, net_sigs_abs / mkt_tvr)) * mkt_tvr * net_sigs_abs[adv_win_start:]) / tvr) if tvr != 0 else np.nan
        est_cap_in_million = est_cap / 1e6
        return est_cap_in_million



    def _decode_by_group(self, group: Union[str, pd.Series], signal=None):
        if isinstance(group, str):
            info: pd.DataFrame = self.db['u']  # type: ignore
            try:
                group_info = info[group]
            except KeyError:
                raise ValueError(f"Group {group} not found in database")
        else:
            group_info = group
        if signal is None:
            sigs = self.signal.copy()
        else:
            sigs = signal.copy()
        sigs = sigs.reindex(group_info.index, axis=1)
        # Calculate PnLs
        pnls = self._get_pnls(sigs)
        total_pnl = pnls['total']
        # get the metrics for each sector
        group_metrics = {}
        for group in group_info.unique():
            this_group_names = group_info[group_info == group].index
            this_group_pnl = {k: v[this_group_names] for k, v in pnls.items()}
            group_metrics[group] = self._get_metrics(this_group_pnl, sigs[this_group_names])
        # Group PnLs by sector
        group_pnls = {}
        for group in group_info.unique():
            group_stocks = group_info[group_info == group].index
            group_pnls[group] = total_pnl[group_stocks]#.sum(axis=1)

        # Calculate overall contribution
        total_contribution = {group: pnl.sum().sum() for group, pnl in group_pnls.items()}
        total_contribution = sorted(total_contribution.items(), key=lambda x: x[1], reverse=True)

        return group_metrics, group_pnls, total_contribution

    def _get_pnls(self, sigs: pd.DataFrame) -> Dict[str, pd.DataFrame]:

        _hedged_ret = self.db['resid']
        _hedged_ret.index.name = 'date' # type: ignore
        assert sigs.columns.isin(_hedged_ret.columns).all(), 'signal must have the same equities as DB' # type: ignore
        sigs = sigs.reindex(_hedged_ret.columns, axis=1)# type: ignore
        long_pnl = sigs.where(sigs > 0).shift() * _hedged_ret
        short_pnl = sigs.where(sigs < 0).shift() * _hedged_ret
        total_pnl = long_pnl.add(short_pnl, fill_value=0)
        raw_ret = self.db['last'].pct_change() # type: ignore
        _hedged_inst_ret = raw_ret - _hedged_ret
        h_long_pnl = _hedged_inst_ret * sigs.where(sigs > 0).shift()
        h_short_pnl = _hedged_inst_ret * sigs.where(sigs < 0).shift()
        h_total_pnl = h_long_pnl.add(h_short_pnl, fill_value=0)

        return {'long': long_pnl, 'short': short_pnl, 'total': total_pnl, 'hedge_long': h_long_pnl, 'hedge_short': h_short_pnl, 'hedge_total': h_total_pnl}


    
    def _get_metrics(self, 
                    pnls: Dict[str, pd.DataFrame],
                    signal: pd.DataFrame,
                    mask=None):
        '''
        Calculate metrics for total, long, and short pnl.
        ''' 
        if mask is not None:
            total_pnl = pnls['total'] * mask
            long_pnl = pnls['long'] * mask
            short_pnl = pnls['short'] * mask
            hedge_total = pnls['hedge_total'] * mask
            signal = signal * mask
        else:
            total_pnl = pnls['total']
            long_pnl = pnls['long']
            short_pnl = pnls['short']
            hedge_total = pnls['hedge_total']

        daily_total_pnl = total_pnl.sum(axis=1)
        daily_long_pnl = long_pnl.sum(axis=1)
        daily_short_pnl = short_pnl.sum(axis=1) 
        daily_hedge_total = hedge_total.sum(axis=1)

        lmv = signal.where(signal > 0).sum(axis=1)
        smv = -1 * signal.where(signal < 0).sum(axis=1)
        gmv = lmv + smv
        daily_total_ret = daily_total_pnl / gmv
        daily_long_ret = daily_long_pnl / lmv
        daily_short_ret = daily_short_pnl / smv
        daily_hedge_ret = daily_hedge_total / gmv
        

        summary_stats = pd.DataFrame({
            'Avg Daily PnL': [daily_total_pnl.mean(), daily_long_pnl.mean(), daily_short_pnl.mean(), daily_hedge_total.mean()   ],
            'Avg Daily Return': [daily_total_ret.mean(), daily_long_ret.mean(), daily_short_ret.mean(), daily_hedge_ret.mean()],
            'Volatility': [daily_total_ret.std(), daily_long_ret.std(), daily_short_ret.std(), daily_hedge_ret.std()],
            'Sharpe': [(daily_total_ret.mean() / daily_total_ret.std()) * np.sqrt(252),
                        (daily_long_ret.mean() / daily_long_ret.std()) * np.sqrt(252),
                        (daily_short_ret.mean() / daily_short_ret.std()) * np.sqrt(252),
                        (daily_hedge_ret.mean() / daily_hedge_ret.std()) * np.sqrt(252)],
        }, index=['Total', 'Long', 'Short', 'Hedge'])

        return summary_stats
