# 14.1 A strategy exhibits a high turnover, high leverage, and high number of bets, with
# a short holding period, low return on execution costs, and a high Sharpe ratio.
# Is it likely to have large capacity? What kind of strategy do you think it is?

# High Turnover: utilizes AUM frequently
# High Leverage: Further increases position sizes wrt AUM
# High Number of Bets: High frequency of trades
# Short Holding Period: Trades are closed quickly
# Low return on execution costs: Return is low compared to the execution costs
# High Sharpe Ratio: Higher returns compared to risk taken

# This indicates a fragile strategy. It already makes a lot of bets
# if it becomes slighlty less profitable, execution costs might cause significant losses.
# As the capacity increases slippage increases and it can turn to be not profitable.

# 14.2 Onthe dollar bars dataset forE-mini S&P 500 futures, compute
import pandas as pd
import numpy as np
dollar_df = pd.read_csv('./data/dollar_df_2025_14_days.csv')
dollar_df['timestamp'] = pd.to_datetime(dollar_df['timestamp'])
dollar_df.set_index('timestamp', inplace=True)


# (a) HHI index onpositive returns.
def getHHI(betRet):
    if betRet.shape[0]<=2:
        return np.nan
    wght=betRet/betRet.sum()
    hhi=(wght**2).sum()
    hhi=(hhi-betRet.shape[0]**-1)/(1.-betRet.shape[0]**-1)
    return hhi


ret=dollar_df['close'].pct_change().dropna()
ret.index = dollar_df.iloc[1:,].index

rHHIPos=getHHI(ret[ret>=0]) # concentration of positive returns per bet
#————————————————————————————————————————

# (b) HHI index onnegative returns.
rHHINeg=getHHI(ret[ret<0]) # concentration of negative returns per bet

# (c) HHI index ontimebetween bars.
tHHI=getHHI(ret.groupby(pd.Grouper(freq='D')).count()) # concentr. bets/day

# (d) The 95-percentile DD.
def computeDD_TuW(series,dollars=False):
    # compute series of drawdowns and the time under water associated with them
    df0=series.to_frame('pnl')
    df0['hwm']=series.expanding().max()
    df1=df0.groupby('hwm').min().reset_index()
    df1.columns=['hwm','min']
    df1.index=df0['hwm'].drop_duplicates(keep='first').index # time of hwm
    df1=df1[df1['hwm']>df1['min']] # hwm followed by a drawdown
    if dollars:
        dd=df1['hwm']-df1['min']
    else:
        dd=1-df1['min']/df1['hwm']

    tuw=((df1.index[1:]-df1.index[:-1]) / np.timedelta64(1,'D')).values# in years
    
    tuw=pd.Series(tuw,index=df1.index[:-1])
    return dd,tuw

dd, tuw = computeDD_TuW(ret)

dd.quantile(0.95)

# (e) The 95-percentile TuW.
tuw.quantile(0.95)

# (f) Annualized average return.

bars_per_day = ret.groupby(pd.Grouper(freq='D')).count().mean()
trading_days_per_year = 365#
N = bars_per_day * trading_days_per_year

ann_ret = ret.mean() * N 
print("Annualized Return:", ann_ret)

# (g) Average returnsfrom hits(positive returns).
avg_hit = ret[ret > 0].mean()
print("Avg Return (Hits):", avg_hit)

# (h) Average return from misses(negative returns).
avg_miss = ret[ret < 0].mean()
print("Avg Return (Misses):", avg_miss)

# (i) Annualized SR.
ann_sr = (ret.mean() / ret.std()) * np.sqrt(N)
print("Annualized Sharpe Ratio:", ann_sr)

# (j) Information ratio, where the benchmark is the risk-freerate.
info_ratio = (ret.mean() - 0) / ret.std() * np.sqrt(N)
print("Information Ratio:", info_ratio)

# (k) PSR.
from scipy.stats import norm, kurtosis, skew

def get_PSR(sr, sr_benchmark, returns):
    """
    sr: observed Sharpe ratio (non-annualized)
    sr_benchmark: benchmark Sharpe ratio
    returns: array-like of returns (non-annualized)
    """
    T = len(returns)
    gamma3 = skew(returns)
    gamma4 = kurtosis(returns, fisher=False)  # Pearson definition, so Gaussian = 3

    denom = np.sqrt(1 - gamma3 * sr + ((gamma4 - 1) / 4) * sr**2)
    psr = norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / denom)
    return psr
psr = get_PSR(ann_sr, 1.0, ret)

print("PSR (Sharpe > 1):", psr)

# (l) DSR,where we assume there were 100 trials,and the variance of the trials’ SR was 0.5.
def get_DSR(sr, sr_var, n_trials=100):
    from scipy.stats import norm
    import numpy as np

    gamma = 0.5772  # Euler-Mascheroni constant
    sr_std = np.sqrt(sr_var)

    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.exp(1)))

    sr_star = sr_std * ((1 - gamma) * z1 + gamma * z2)

    # Deflated Sharpe Ratio = PSR using SR* as benchmark
    return norm.cdf((sr - sr_star) / sr_std)

# sr_var is important and normally needs to be estimated by averaging 
# sharpe ratios of backtest runs. We hard code here for simplicity.
dsr = get_DSR(ann_sr, sr_var=0.5, n_trials=100)
print("Deflated Sharpe Ratio (DSR):", dsr)

