import pandas as pd
import numpy as np
from typing import Union, Literal, Tuple
import matplotlib.pyplot as plt

def calhalfic(x):
    # Implement the calhalfic function if it's not defined elsewhere
    # This is a placeholder implementation, adjust as needed
    return np.log(2) / np.log(1 + x.iloc[0] / x.iloc[-1])

class SignalDecoder:
    def __init__(self, db: pd.DataFrame, signal: pd.DataFrame):
        self.db = db
        if not isinstance(signal, pd.DataFrame):
            raise TypeError('signal must be a pd.DataFrame')
        self.signal = signal
    
    def decode_correlation(self, sigs: pd.DataFrame) -> pd.DataFrame:
        '''
        calcualte the correlation between the signal and the hedged return  
        '''
        _hedged_ret = self.db['resid']
        assert sigs.columns.isin(_hedged_ret.columns).all(), 'signal must have the same equities as DailyDB'
        _hedged_ret = _hedged_ret[sigs.columns]
        sigs = sigs.shift().T
        _hedged_ret = _hedged_ret.T


    def decode_long_short(self, sigs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
        # Ensure 'resid' column exists in self.db
        # if 'resid' not in self.db.columns:
        #     raise KeyError("'resid' column not found in the database")
        _hedged_ret = self.db['resid']#.shift(-1)
        assert sigs.columns.isin(_hedged_ret.columns).all(), 'signal must have the same equities as DailyDB'
        _hedged_ret = _hedged_ret[sigs.columns]
        long_pnl = sigs.where(sigs > 0).shift() * _hedged_ret
        short_pnl = sigs.where(sigs < 0).shift() * _hedged_ret
        total_pnl = long_pnl.add(short_pnl, fill_value=0)

        correlation = self._calculate_correlation(long_pnl, short_pnl)
        metrics = self.get_metrics(total_pnl, long_pnl, short_pnl)

        return long_pnl, short_pnl, total_pnl, correlation, metrics
    
    def decode_hedge_ret(self, sigs: pd.DataFrame) -> pd.DataFrame:
        '''
        decode the hedge return
        '''
        _hedge_ret = self.db['last'].pct_change() - self.db['resid'] # TODO 
        assert sigs.columns.isin(_hedge_ret.columns).all(), 'signal must have the same equities as DailyDB'
        _hedge_ret = _hedge_ret[sigs.columns]
        h_long_pnl = sigs.where(sigs > 0).shift() * _hedge_ret
        h_short_pnl = sigs.where(sigs < 0).shift() * _hedge_ret
        h_total_pnl = h_long_pnl.add(h_short_pnl, fill_value=0)
        h_correlation = self._calculate_correlation(h_long_pnl, h_short_pnl)
        h_metrics = self.get_metrics(h_total_pnl, h_long_pnl, h_short_pnl)

        return h_long_pnl, h_short_pnl, h_total_pnl, h_correlation, h_metrics   


    def _calculate_correlation(self, long_pnl: pd.DataFrame, short_pnl: pd.DataFrame) -> float:
        """Calculate correlation between long and short components."""
        long_mean = long_pnl.mean(axis=1)
        short_mean = short_pnl.mean(axis=1)
        return long_mean.corr(short_mean)

    def decode_sector(self, sector_name: str, sector: str):
        info = self.db['u']
        names_in_this_sector = info[sector_name][info[sector_name] == sector].index
        this_sector_sigs = self.signal.loc[:, names_in_this_sector].copy()
        _res = {}
        lpnl, spnl, tpnl, corr, metrics = self.decode_long_short(this_sector_sigs)
        _res[sector] = metrics
            
        return lpnl, spnl, tpnl, corr,metrics
        
    def decode_industry(self, industry_name: str):
        return self.decode_sector(industry_name)

    def decode_calender(self, by: str='day_of_week'):
        sigs = self.signal.copy()
        sigs.index.name = 'date'
        _res = {}
        
        if by == 'day_of_week':
            sigs['day_of_week'] = sigs.index.dayofweek
            grouper = 'day_of_week'
        elif by == 'month':
            sigs['month'] = sigs.index.month
            grouper = 'month'
        elif by == 'quarter':
            sigs['quarter'] = sigs.index.quarter
            grouper = 'quarter'
        else:
            raise ValueError("Invalid 'by' parameter. Choose 'day_of_week', 'month', or 'quarter'.")

        for period, sig in sigs.groupby(grouper):
            lpnl, spnl, tpnl, corr, metrics = self.decode_long_short(sig.drop(grouper, axis=1))
            _res[period] = {
                'total_pnl': tpnl,
                'long_pnl': lpnl,
                'short_pnl': spnl,
                'correlation': corr,
                'metrics': metrics
            }
        
        return _res

    def decode_single_asset(self, asset: str):
        lpnl_t, spnl_t, tpnl_t, _, _ = self.decode_long_short(self.signal.copy())
        lpnl_target, spnl_target, tpnl_target = lpnl_t[asset], spnl_t[asset], tpnl_t[asset]
        lpnl_rest, spnl_rest, tpnl_rest = lpnl_t.drop(asset, axis=1), spnl_t.drop(asset, axis=1), tpnl_t.drop(asset, axis=1)

        
        target_metrics = self.get_metrics(tpnl_target.to_frame(), lpnl_target.to_frame(), spnl_target.to_frame())
        total_metrics = self.get_metrics(tpnl_t, lpnl_t, spnl_t)
        rest_metrics = self.get_metrics(tpnl_rest, lpnl_rest, spnl_rest)
            
        tpnl_t1 = tpnl_t.sum(axis=1)
        tpnl_rest = tpnl_rest.sum(axis=1)
        contribution = tpnl_target.sum() / tpnl_t1.sum()
        
        return {
            'asset': asset,
            'target_pnl': tpnl_target,
            'total_pnl': tpnl_t1,
            'pnl_by_stock': tpnl_t,
            'rest_pnl': tpnl_rest,
            'target_metrics': target_metrics,
            'total_metrics': total_metrics,
            'rest_metrics': rest_metrics,
            'contribution': contribution
        }

    def shift_analysis(self, n: int = 10):
        _shifted_res = {}
        for i in range(0, n+1):  # Only positive shifts
            shifted_sigs = self.signal.shift(i).copy()
            lpnl, spnl, tpnl, corr, metrics = self.decode_long_short(shifted_sigs)
            
            # Store more detailed metrics for each shift
            _shifted_res[i] = {
                'total_sharpe': metrics.loc['Total', 'Sharpe Ratio'],
                'long_sharpe': metrics.loc['Long', 'Sharpe Ratio'],
                'short_sharpe': metrics.loc['Short', 'Sharpe Ratio'],
                'total_return': metrics.loc['Total', 'Avg Daily Return'],
                'long_return': metrics.loc['Long', 'Avg Daily Return'],
                'short_return': metrics.loc['Short', 'Avg Daily Return'],
                'correlation': corr,
                'max_drawdown': metrics.loc['Total', 'Max Drawdown (%)']
            }
        
        shifted_res = pd.DataFrame(_shifted_res).T
        shifted_res.index.name = 'shift_days'
        
        # Calculate peak metrics
        peak_total_sharpe = shifted_res['total_sharpe'].idxmax()
        peak_long_sharpe = shifted_res['long_sharpe'].idxmax()
        peak_short_sharpe = shifted_res['short_sharpe'].idxmax()
        
        # Calculate half-lives
        total_half_life = self.calculate_half_life(shifted_res['total_sharpe'], peak_total_sharpe)
        long_half_life = self.calculate_half_life(shifted_res['long_sharpe'], peak_long_sharpe)
        short_half_life = self.calculate_half_life(shifted_res['short_sharpe'], peak_short_sharpe)
        
        peak_metrics = {
            'peak_total_sharpe': (peak_total_sharpe, shifted_res.loc[peak_total_sharpe, 'total_sharpe']),
            'peak_long_sharpe': (peak_long_sharpe, shifted_res.loc[peak_long_sharpe, 'long_sharpe']),
            'peak_short_sharpe': (peak_short_sharpe, shifted_res.loc[peak_short_sharpe, 'short_sharpe']),
            'total_half_life': total_half_life,
            'long_half_life': long_half_life,
            'short_half_life': short_half_life
        }
        
        return shifted_res, peak_metrics
    
    def decode_capital_effect(self, q0: float=0.1, q1: float=0.9):
        lpnl_t, spnl_t, tpnl_t, _, _ = self.decode_long_short(self.signal.copy())
        market_cap = self.db['mkt_cap_usd']

        # Ensure market_cap and signal have the same columns (stocks)
        market_cap = market_cap[self.signal.columns]

        # Calculate daily quantiles
        daily_q0 = market_cap.quantile(q0, axis=1)
        daily_q1 = market_cap.quantile(q1, axis=1)

        # Create masks for small, mid, and large cap stocks
        small_cap_mask = market_cap.lt(daily_q0, axis=0)
        large_cap_mask = market_cap.ge(daily_q1, axis=0)
        mid_cap_mask = ~(small_cap_mask | large_cap_mask)


        # Calculate metrics for each cap group
        small_cap_metrics = self.get_metrics(lpnl_t, spnl_t, tpnl_t, small_cap_mask)
        mid_cap_metrics = self.get_metrics(lpnl_t, spnl_t, tpnl_t, mid_cap_mask)
        large_cap_metrics = self.get_metrics(lpnl_t, spnl_t, tpnl_t, large_cap_mask)


        return {
            'small_cap': {'metrics': small_cap_metrics},
            'mid_cap': {'metrics': mid_cap_metrics},
            'large_cap': {'metrics': large_cap_metrics}
        }

    def calculate_max_drawdown(self, cumulative_returns):
        peak = cumulative_returns.sort_index().expanding(min_periods=1).max()
        df = pd.DataFrame({'peak': peak, 'cumulative_returns': cumulative_returns})
        df = df.query('peak > 0')   
        mdd = ((df['peak'] - df['cumulative_returns']) / df['peak']).max() * -100
        return mdd

    def calculate_half_life(self, series, peak):
        if pd.isna(peak) or pd.isna(series[peak]):
            return None
        half_peak_value = series[peak] / 2
        for i in range(peak, len(series)):
            if pd.isna(series.iloc[i]) or series.iloc[i] < half_peak_value:
                return i - peak
        return None  # Return None if half-life is not found

    def est_sharpe(self, pnl: pd.DataFrame):
        if len(pnl.shape) == 2:
            pnl = pnl.mean(axis=1) # type: ignore
        mean_daily_ret = pnl.mean()
        std_daily_ret = pnl.std()
        sharpe_ratio = (mean_daily_ret / std_daily_ret) * np.sqrt(252)
        return sharpe_ratio

        
    def get_metrics(self, 
                    total_pnl: pd.DataFrame, 
                    long_pnl: pd.DataFrame, 
                    short_pnl: pd.DataFrame,
                    mask: pd.DataFrame|None=None):
        '''
        Calculate metrics for total, long, and short pnl.
        ''' 
        daily_total_pnl = total_pnl.sum(axis=1)
        daily_long_pnl = long_pnl.sum(axis=1)
        daily_short_pnl = short_pnl.sum(axis=1) 

        if mask is not None:
            daily_total_pnl = daily_total_pnl * mask
            daily_long_pnl = daily_long_pnl * mask
            daily_short_pnl = daily_short_pnl * mask
            signal = self.signal.copy() * mask

        lmv = signal.where(signal > 0).sum(axis=1)
        smv = -1 * signal.where(signal < 0).sum(axis=1)
        gmv = lmv + smv
        daily_total_ret = daily_total_pnl / gmv
        daily_long_ret = daily_long_pnl / lmv
        daily_short_ret = daily_short_pnl / smv
        

        summary_stats = pd.DataFrame({
            'Avg Daily PnL': [daily_total_pnl.mean(), daily_long_pnl.mean(), daily_short_pnl.mean()],
            'Avg Daily Return': [daily_total_ret.mean(), daily_long_ret.mean(), daily_short_ret.mean()],
            'Volatility': [daily_total_ret.std(), daily_long_ret.std(), daily_short_ret.std()],
            'Sharpe Ratio': [(daily_total_ret.mean() / daily_total_ret.std()) * np.sqrt(252),
                                (daily_long_ret.mean() / daily_long_ret.std()) * np.sqrt(252),
                                (daily_short_ret.mean() / daily_short_ret.std()) * np.sqrt(252)],
            'Max Drawdown (%)': [self.calculate_max_drawdown(daily_total_pnl.cumsum()),
                            self.calculate_max_drawdown(daily_long_pnl.cumsum()),
                            self.calculate_max_drawdown(daily_short_pnl.cumsum())]
        }, index=['Total', 'Long', 'Short'])

        return summary_stats

    
    def get_signal_stats(self):
        '''
        Get signal statistics.
        output:
            - stats: Signal statistics (pd.Series)
        '''
        stats = pd.Series()
        stats['mean'] = self.signal.mean().mean()
        stats['std'] = self.signal.std().mean()
        stats['min'] = self.signal.min().min()
        stats['max'] = self.signal.max().max()
        return stats
    
