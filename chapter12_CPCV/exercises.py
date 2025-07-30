# This section present a new method,which addresses them a in drawback of the
# WF and CV methods, namely that those schemes test a single path.
# I call it the“combinatorial purged cross-validation” (CPCV) method.


# 12.1 Suppose that you develop a momentum strategy on a futures contract, where
# the forecast is based on an AR(1) process. You backtest this strategy using the
# WF method, and the Sharpe ratio is 1.5. You then repeat the backtest on the
# reversed series and achieve a Sharpe ratio of –1.5. What would be the mathematical grounds fordisregarding the second result,ifany?

# ANSWER:
# The Sharpe ratio of -1.5 on the reversed series confirms that the strategy is indeed capturing a time-asymmetric, momentum-like effect. It doesn't invalidate the original result; it actually supports the hypothesis that the strategy is a momentum strategy. Therefore, you don't "disregard" the second result in the sense of ignoring it; you interpret it as evidence consistent with a genuine momentum effect. You would disregard it as a reason to not pursue the strategy based on the original positive Sharpe.
# You wouldn't disregard it if you were trying to prove your strategy isn't momentum. But if you claim it is momentum, the reversed test supports your claim.


# 12.2 You develop a mean-reverting strategy on a futures contract. Your WF backtest
# achieves a Sharpe ratio of 1.5. You increase the length of the warm-up period,
# and the Sharpe ratio drops to 0.7. You go a head and present only the result with
# the higher Sharpe ratio, arguing that a strategy with a shorter warm-up is more
# realistic.Is this selection bias?

# ANSWER:
# Yes.
# A shorter warm-up period means the initial models in the walk-forward are trained on less data. They might be less stable or more prone to fitting noise in that shorter period. If this shorter period happens to coincide with market conditions particularly favorable to the mean-reverting strategy, it could produce an unusually high Sharpe.
# A longer warm-up period generally leads to more robust parameter estimates for the model, as it's trained on more data. If the Sharpe ratio drops significantly with a longer (and arguably more robust) warm-up, it suggests the initial high Sharpe might have been less reliable or an artifact of the shorter training data.


# 12.3 Your strategy achieves a Sharpe ratio of 1.5 on a WF backtest, but a Sharpe
# ratio of 0.7 on a CV backtest. You go ahead and present only the result with
# the higher Sharpe ratio, arguing that the WF backtest is historically accurate,
# while the CV backtest is a scenario simulation, or an inferential exercise. Is this
# selection bias?

# ANSWER:

# Yes. Selection Bias: Again, the analyst is choosing the result that looks better and providing a justification that might not be fully sound.
# WF vs. CV (K-Fold type):
# Walk-Forward (WF): Preserves the chronological order of data, which is crucial for time series. It simulates how a strategy would be traded by training on past data and testing on subsequent future data. It tests a single historical path. Its claim to "historical accuracy" is in its sequential nature.
# Standard Cross-Validation (CV, like K-Fold): Typically shuffles data and creates multiple train/test splits. For time series, shuffling breaks the temporal dependence and can lead to look-ahead bias if not handled very carefully (e.g., with specialized time-series CV methods like TimeSeriesSplit in scikit-learn, or Purged K-Fold which is more advanced). If standard K-Fold CV was used naively on time series data, its results could indeed be unreliable or overly optimistic due to leakage.


# 12.4 Your strategy produces 100,000 forecasts over time. You would like to derive
# the CPCV distribution of Sharpe ratios by generating 1,000paths. What are the
# possible combinations of parameters (N,k)that will allow you to achieve that?

# ANSWER:

# We want φ[N,k] ≈ 1,000.
# Finding (N, k): This is a combinatorial problem. There isn't a single unique answer; multiple combinations of (N, k) could yield around 1,000 paths. 
# We need to find pairs (N,k) such that (k/N) * N! / (k! * (N-k)!) ≈ 1000.
# We want to calculate = (k/N) * C(N,k)

# (N=13, k=7): C(13,7) = 1716 splits. φ = (7/13) * 1716 ≈ 923.07 ≈ 924 paths. (Good candidate)