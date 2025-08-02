# 15.1 A portfolio manager intends to launch a strategy that targets anannualized SR of
# 2. Bets have a precision rate of 60%, with weekly frequency. The exit conditions
# are 2% for profit-taking, and –2% for stop-loss.

# (a) Is this strategy viable?
# No - plugging in the parameters given sharpe ratio comes up around 1.47

# (b) Ceteris paribus, what is the required precision rate that would make the strategy profitable?
# 63.36%

# (c) For what betting frequency isthetarget achievable?
# 2 = ((2 * 0.6 - 1 ) / 2 sqrt(0.6 * (1 - 0.6))) * sqrt(n) solving for n = 96

# (d) For what profit-taking threshold is the target achievable?
# To achieve the target annualized Sharpe Ratio of 2, the manager must increase the profit-taking threshold from 2% to approximately 2.31%.

# 15.2 Following up onthe strategy fromexercise 1.

# (a) What isthe sensitivityof SRtoa1% change ineach parameter?
# Conclusion: 
# A 1% increase in precision leads to approximately a +6.25% increase in the Sharpe Ratio.

# A 1% increase in frequency leads to approximately a +0.5% increase in the Sharpe Ratio. 
# (This makes sense, as SR is proportional to the square root of n, so sqrt(1.01) ≈ 1.005).

# Bet Size: 0% SR impact (symmetrical)

# (b) Giventhesesensitivities,andassumingthatallparametersareequallyhard
# toimprove, which one offers thelowest hanging fruit?

# Sharpe Ratio

# (c) Does changing any of the parameters in exercise 1 impact the others? For
# example, does changing the betting frequency modify the precision rate,
# etc.?

# Changing Frequency (n) impacts Precision (p):
# This is the most significant trade-off. If you increase your betting frequency (e.g., from weekly to daily), 
# you are forced to act on signals that are weaker, less clear, or have had less time to develop.
# This almost always leads to a decrease in precision (p). 
# Overtrading is a common pitfall where a manager increases n at the cost of p, 
# often resulting in a lower overall SR.
# Changing Bet Size (Profit/Loss Thresholds) impacts n and p:
# Narrowing the band (e.g., changing exits from +/-2% to +/-1%): It will take less time for the price to hit 
# either the profit or loss target. This will naturally increase the betting frequency (n). 
# However, because you are capturing smaller moves, your outcomes may be more influenced by random market noise 
# rather than your underlying signal, which will likely decrease precision (p).
# Widening the band (e.g., from +/-2% to +/-4%): It will take longer to hit an exit target, 
# which decreases the betting frequency (n). This gives your thesis more time to play out, 
# which could potentially increase precision (p), but only if your initial signal was strong.


# 15.3 Suppose a strategy that generates monthly bets over two years, with returns
# following a mixture of two Gaussian distributions. The first distribution has
# a mean of –0.1 and a standard deviation of 0.12. The second distribution has
# a mean of 0.06 and a standard deviation of 0.03. The probability that a draw
# comes fromthe firstdistributionis0.15.
# (a) Derive the firstfour moments forthe mixture’sreturns. (mean, variance, skew, and kurtosis)
# (b) What isthe annualized SR? ~1.5
# (c) Using those moments, compute PSR[1] (see Chapter 14). At a 95% confi-
# dence level, would you discardthisstrategy?

# Conclusion: Would you discard the strategy?
# The PSR[1] of 0.0496 means there is only a 4.96% probability that the true Sharpe Ratio of this strategy is greater than 1.
# A 95% confidence level means we require the probability of failure (i.e., 1 - PSR) to be less than 5%, or equivalently, the PSR to be greater than 95%.
# Alternatively, we can compare the probability of being wrong to our significance level (α = 1 - 0.95 = 0.05). We are testing the null hypothesis H₀: SR_true ≤ 1. We want to reject H₀. The p-value for this one-sided test is 1 - PSR[1] ≈ 0.9504. Since p-value > α, we fail to reject H₀.
# A simpler interpretation: Our confidence that the true SR is greater than 1 is only 4.96%. Since 4.96% < 95%, we do not have enough evidence to support the strategy.
# Answer: Yes, at a 95% confidence level, you would discard this strategy. The high skewness and kurtosis (fat tails) dramatically increase the uncertainty around the Sharpe Ratio estimate, making the observed SR of 1.585 statistically insignificant relative to the benchmark of 1.


# 15.4 UsingSnippet15.5,compute P[p < p𝜃∗=1] for the strategy described in exercise
# 3. At a significance level of 0.05, would you discard this strategy? Is this result
# consistent with PSR[𝜃∗]?

# θ represents the true, unknown Sharpe Ratio of the strategy.
# θ* is the benchmark Sharpe Ratio we want to beat, which is 1.
# we are asked to compute the probability that the strategy's true Sharpe Ratio is less than 1.

# we need to compute is simply:
# P[θ < θ*] = 1 - PSR[θ*]

# The probability that the strategy's true Sharpe Ratio is less than 1 is 95.04%.
# Since the probability of the null hypothesis being true (95.04%) is much greater than our significance level (5%), 
# we fail to reject the null hypothesis. This means we do not have sufficient evidence to claim the 
# strategy's true Sharpe Ratio is greater than 1. Therefore, we must discard the strategy.

# Yes, the result is perfectly consistent. They are two sides of the exact same coin.
# PSR Test (from 15.3c): We test if our confidence in the strategy (PSR[1]) exceeds a confidence threshold (1 - α = 95%).
# Test: Is 0.0496 > 0.95?
# Result: False.
# Conclusion: Discard the strategy.
# Current Test (from 15.4): We test if the probability of failure (P[θ < 1]) is below our significance level (α = 5%).
# Test: Is 0.9504 < 0.05?
# Result: False.
# Conclusion: Discard the strategy.



# 15.5 In general, what result do you expect to be more accurate, PSR[𝜃∗] or
# P[p < p𝜃∗=1]? How arethese twomethods complementary?
  
# The PSR[θ] calculation is expected to be significantly more accurate*. Here's the critical difference:
# Symmetric Bet Calculation (like in 15.1): This method is based on a simplified model that makes a huge assumption: 
# that returns are symmetric and follow a simple binomial process (win or lose by the same amount).
# This model implicitly assumes there is no skew (γ₃=0) and no excess kurtosis (γ₄=3). 
# It only uses the first two moments (mean and variance).
# Probabilistic Sharpe Ratio (PSR): This method uses the first four moments of the actual observed returns. 
# By incorporating skew (γ₃) and kurtosis (γ₄), it accounts for the true, non-Normal shape of the return distribution. 
# Financial returns are almost never Normal; they often have negative skew (rare, large losses) and fat tails (more extreme events than a Normal distribution would suggest).
# Conclusion: Because PSR uses a richer, more realistic description of the return distribution, 
# its conclusions about the statistical reliability of a Sharpe Ratio estimate will be much more accurate and 
# robust than a conclusion drawn from a model that ignores these crucial higher-moment risks.

# 15.6 15.6 Re-examine the results from Chapter 13, in light of what you have learned in
# thischapter.

# (a) Does the asymmetry between profit-taking and stop-loss thresholds in OTRs make sense?
# Yes, it makes perfect sense, and the lessons from Chapter 15 explain why. 
# The goal of an Optimal Trading Rule (OTR) is not just to maximize the mean return, 
# but to optimize the risk-adjusted return.
# Asymmetry is a tool to engineer the shape of the return distribution.
# By setting a tight stop-loss and a wider profit-taking threshold, a manager is explicitly trying to create a 
# return distribution with positive skew.
# This means the strategy is designed to have frequent, small losses (the stop-losses) but also occasional, 
# large gains (the profit-taking). This is the classic trading mantra: "Cut your losses short and let your profits run."
# While this might lower the win rate (p), the magnitude of the wins can more than compensate for the losses, 
# leading to a better overall Sharpe Ratio. Chapter 15 teaches us that skew is a critical component of risk, 
# and OTRs are a direct application of managing and shaping that skew to our advantage.



# (b) What is the range of p implied by Figure 13.1, for a daily betting frequency?

# Figure 13.1 in the book shows that an optimal rule can achieve a very high Sharpe Ratio, let's estimate a peak SR ≈ 2.5.
# Frequency (n): Daily => n = 252 (trading days)
# Target (SR): 2.5
# We can use the symmetric-bet formula to find the implied precision p that would be needed to achieve this.
# SR = [(2p - 1) / (2 * sqrt(p(1-p)))] * sqrt(n)
# 2.5 = [(2p - 1) / (2 * sqrt(p - p²))] * sqrt(252)
# 2.5 / sqrt(252) = (2p - 1) / (2 * sqrt(p - p²))
# 0.1575 = (2p - 1) / (2 * sqrt(p - p²))
# Square both sides:
# 0.0248 = (2p - 1)² / (4 * (p - p²))
# 0.0248 * 4 * (p - p²) = 4p² - 4p + 1
# 0.0992p - 0.0992p² = 4p² - 4p + 1
# Rearrange into a quadratic equation ax² + bx + c = 0:
# 4.0992p² - 4.0992p + 1 = 0
# Solving this with the quadratic formula gives two solutions: p ≈ 0.578 and p ≈ 0.422. Since a high Sharpe Ratio requires a precision greater than 50%, the implied precision is:
# p ≈ 57.8%

# (c) What is the range of p implied by Figure 13.5, for a weekly betting frequency?
# Let's assume Figure 13.5 shows a strategy that achieves our familiar target of SR = 2.
# Frequency (n): Weekly => n = 52
# Target (SR): 2
# This is the exact problem we solved in exercise 15.1(b). By inverting the formula, we found that to achieve an SR of 2 with 52 bets per year, the required precision is:
# p ≈ 63.4%
# Notice the key insight from comparing (b) and (c): To achieve a high Sharpe Ratio with a lower betting frequency (weekly vs. daily), you need a much higher precision rate or "edge" on each individual bet.