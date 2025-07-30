import pandas as pd
import numpy as np
from scipy.stats import norm, skew, kurtosis
import yfinance as yf # Using yfinance to make the example runnable for everyone

# =============================================================================
# Helper Functions from your Exercise 14.2 (with slight improvements)
# =============================================================================

def getHHI(betRet):
    if betRet is None or betRet.shape[0] <= 2:
        return np.nan
    # Ensure no division by zero if sum is zero
    if betRet.sum() == 0:
        return 0 
    wght = betRet / betRet.sum()
    hhi = (wght**2).sum()
    hhi = (hhi - betRet.shape[0]**-1) / (1. - betRet.shape[0]**-1)
    return hhi

def computeDD_TuW(series):
    # This improved version calculates DD on the equity curve for accuracy
    equity_curve = (1 + series).cumprod()
    hwm = equity_curve.expanding().max()
    
    # Calculate Drawdown
    dd = (hwm - equity_curve) / hwm
    dd = dd[dd > 0] # We only care about positive drawdowns

    # Calculate Time under Water
    tuw_periods = (~(equity_curve >= hwm)).astype(int)
    # Find blocks of consecutive drawdowns
    tuw_blocks = (tuw_periods.diff().fillna(0) != 0).astype(int).cumsum()
    tuw = tuw_periods.groupby(tuw_blocks).sum()
    tuw = tuw[tuw > 0] # We only care about periods where we were under water
    
    # This version returns DD and TuW in number of bars, not days, which is
    # more general for high-frequency data.
    return dd, tuw


def get_PSR(annualized_sr, sr_benchmark_annual, returns):
    T = len(returns)
    if T <= 1: return np.nan
    
    # PSR is typically calculated on non-annualized returns and SRs
    # We need to find the number of bets per year to de-annualize
    bars_per_day = T / len(returns.index.normalize().unique())
    N = bars_per_day * 252 # Annualization factor
    
    sr = annualized_sr / np.sqrt(N)
    sr_benchmark = sr_benchmark_annual / np.sqrt(N)
    
    gamma3 = skew(returns)
    gamma4 = kurtosis(returns, fisher=True) + 3 # Pearson definition

    denom = np.sqrt(1 - gamma3 * sr + ((gamma4 - 1) / 4) * sr**2)
    if np.isclose(denom, 0): return np.nan
    
    psr = norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / denom)
    return psr

def get_DSR(sr_ann, sr_var, n_trials=100):
    gamma = 0.5772
    sr_std = np.sqrt(sr_var)
    if np.isclose(sr_std, 0): return np.nan

    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.exp(1)))

    sr_star = sr_std * ((1 - gamma) * z1 + gamma * z2)
    return norm.cdf((sr_ann - sr_star) / sr_std)

# =============================================================================
# 14.3 Solution (Odd/Even DAY Strategy)
# =============================================================================

# 1. Load Data & Calculate Underlying Returns
dollar_df = pd.read_csv('./data/dollar_df_2025_14_days.csv')
dollar_df['timestamp'] = pd.to_datetime(dollar_df['timestamp'])
dollar_df.set_index('timestamp', inplace=True)


underlying_ret = dollar_df['close'].pct_change().dropna()
underlying_ret.name = 'underlying_ret'

# 2. Define the Strategy
# Side = +1 for even days, -1 for odd days
side = (underlying_ret.index.day % 2 == 0).astype(int) * 2 - 1
side = pd.Series(side, index=underlying_ret.index, name='side')

# 3. Calculate Strategy Returns
strategy_ret = (side * underlying_ret).rename('strategy_ret')

# --- (a) Repeat the calculations from exercise 2 ---
print("\n--- (a) Performance Metrics for the Odd/Even DAY Strategy ---")
results = {}

# (a,b) HHI on returns
results['HHI_Positive_Returns'] = getHHI(strategy_ret[strategy_ret >= 0])
results['HHI_Negative_Returns'] = getHHI(strategy_ret[strategy_ret < 0])

# (c) HHI on time between bets
# This measures the concentration of trading activity across days.
daily_counts = strategy_ret.groupby(pd.Grouper(freq='D')).count()
results['HHI_Time_Between_Bets'] = getHHI(daily_counts[daily_counts > 0])

# (d,e) Drawdown and Time under Water
dd, tuw = computeDD_TuW(strategy_ret)
results['DD_95_Percentile (in % of equity)'] = dd.quantile(0.95) if not dd.empty else np.nan
results['TuW_95_Percentile (in # of bars)'] = tuw.quantile(0.95) if not tuw.empty else np.nan

# (f) Annualized average return
# Calculate annualization factor N based on the data's frequency
bars_per_day = len(strategy_ret) / len(strategy_ret.index.normalize().unique())
trading_days_per_year = 365
N = bars_per_day * trading_days_per_year
results['Annualized_Return'] = strategy_ret.mean() * N

# (g, h) Avg returns from hits and misses
results['Avg_Return_Hits'] = strategy_ret[strategy_ret > 0].mean()
results['Avg_Return_Misses'] = strategy_ret[strategy_ret < 0].mean()

# (i, j) Annualized SR and Information Ratio
ann_sr = (strategy_ret.mean() / strategy_ret.std()) * np.sqrt(N)
results['Annualized_SR'] = ann_sr
results['Information_Ratio'] = ann_sr # vs risk-free rate of 0

# (k) PSR (Probabilistic Sharpe Ratio)
# Let's test against a benchmark Sharpe Ratio of 0, as any profitable strategy must beat this.
results['PSR (SR_benchmark=0)'] = get_PSR(ann_sr, 0, strategy_ret)

# (l) DSR (Deflated Sharpe Ratio)
results['DSR (trials=100, var=0.5)'] = get_DSR(ann_sr, sr_var=0.5, n_trials=100)

# Print the results from part (a)
for key, value in results.items():
    print(f"{key}: {value:.4f}")

# --- (b) What is the correlation to the underlying? ---
print("\n--- (b) Correlation to the Underlying ---")
overall_corr = strategy_ret.corr(underlying_ret)
print(f"Overall Correlation: {overall_corr:.4f}")

# For a more detailed look:
corr_even_days = strategy_ret[side == 1].corr(underlying_ret[side == 1])
corr_odd_days = strategy_ret[side == -1].corr(underlying_ret[side == -1])
print(f"Correlation on Even Days: {corr_even_days:.4f}")
print(f"Correlation on Odd Days: {corr_odd_days:.4f}")


# 14.4 The results from a 2-year back test are that monthly returns 
# have a mean of 3.6% and astandard deviation of 0.079%.
# (a) What isthe SR?
# SR_monthly = (μ_monthly - rf_monthly) / σ_monthly
# SR_monthly = (0.036 - 0) / 0.00079
# SR_monthly ≈ 45.57

# (b) What isthe annualized SR?
# (0.036 * 12)  / np.sqrt(((0.00079 ** 2) * 12) )
# SR_annualized = 45.57 * √12
# SR_annualized ≈ 45.57 * 3.464
# SR_annualized ≈ 157.87