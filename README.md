Advances In Financial Machine Learning is a highly technical book utilizing advanced mathematics throughout. Therefore one needs to study concepts introduced under each chapter to get the maximum benefit. With that said, this repository attempts reducing this density by higlighting the most important concepts, providing chapter summaries as well as the exercise solutions using sample bitcoin data.

Exercises for following chapters are not included:

Chapter 6  - Ensemble Methods: This chapter has all theoretical exercises for which the ChatGPT Chat Link is available

Chapter 11 - The Dangers of Backtesting: This chapter is a warning mentioning common sins and exercises are covering cases where certain sins are committed.

Chapter 16 - Machine Learning Asset Allocation: Skipped section since the concentration of this study is to concentrate initially on trading applications.

Chapter 20/21/22 - These are sections belonging to High-Performance Computing Recipes Part. Previously utilized `mpPandasObj` parallelization function provided under Chapter 20. It would be ideal to refer this sections when training the models with actual vast amounts of data rather than exercise samples.

 Book is consisted of 5 Parts (Data Analysis, Modelling, Backtesting, Useful Financial Features and High-Performance Computing Receipes) with multiple Chapters under each part. 

## Chapter 1 - PREAMBLE - Financial Machine Learning as a Distinct Subject

<b> The Problem</b>: Most quantitative strategies fail because they are "false discoveries" resulting from a flawed, unscientific research process.

<b> The Cause</b>: Standard ML tools fail because financial data is unique, it has a low signal-to-noise ratio, is not IID, and is subject to structural breaks.

<b> The Culprit</b>: The traditional backtest is the main tool for self-deception, leading to massive overfitting.

<b> The Solution</b>: A paradigm shift is needed towards a collaborative, theory-driven, and industrialized "strategy factory" approach that treats financial ML as its own scientific discipline.

<b> The Path Forward</b>: The rest of the book is dedicated to building the components of this factory, providing specific, practical tools to overcome the challenges identified in this chapter.

<I> De Prado suggests the following members for creating a team for building a Strategy Factory: </I>


* <b> Data Curators</b>: Acquire, clean, and structure raw market data into robust, analysis-ready formats.

* <b>Feature Analysts</b>: Transform structured data into informative variables (features) that have potential predictive power.

* <b>Strategists</b>: Develop and train machine learning models that generate predictive signals based on the engineered features.

* <b>Backtesting Team</b>: Rigorously evaluate a model's historical performance, focusing on preventing backtest overfitting and assessing its true viability.

* <b>Deployment Team</b>: Integrate the validated model into the live trading infrastructure, managing execution and operational risk.

* <b>Portfolio Managers</b>: Allocate capital across a portfolio of multiple strategies and manage the overall combined risk.

# PART I - DATA ANALYSIS

## Chapter 2 - Financial Data Structures
<p align="center">
  <img src="readme_files/four_essential_data_types.png?raw=true" alt="Four Essential Data Types" title="Four Essential Data Types" width="600"/>
</p>


This chapter argues that standard time bars (e.g., daily, hourly) are a poor choice for financial ML. Because market activity is not uniform, time-based sampling leads to data with undesirable statistical properties. The solution is to use information-driven bars, which sample data based on market activity (like trade volume or price changes), resulting in series that are much closer to being IID (Independent and Identically Distributed) and better suited for modeling.

### Information-Driven Bars: A Better Alternative
These bars are formed by sampling data whenever a certain amount of market information has been exchanged.

#### <u> Standard Bars </u>

* <b>Tick Bars</b>: Sample every N transactions (ticks).

* <b>Volume Bars</b>: Sample every N units of asset traded (e.g., shares).

* <b>Dollar Bars</b>: Sample every N dollars of value traded.
*!!!* DOLLAR BARS ARE SIGNIFICANT AND CONVENIENT. THEY ARE USED FOR MOST EXERCISES IN REMAINING CHAPTERS *!!!*

#### <u> Information Imbalance & Run Bars </u>

* <b>Tick Imbalance Bars (TIBs)</b>: Sample when the imbalance between buy vs. sell ticks exceeds a threshold.

* <b>Volume/Dollar Imbalance Bars (VIBs/DIBs)</b>: Sample when the volume/dollar imbalance between buys vs. sells exceeds a threshold.

* <b>Tick Run Bars (TRBs)</b>: Sample at the end of a "run," a sequence of consecutive buyer- or seller-initiated ticks.

* <b>Volume/Dollar Run Bars (VRBs/DRBs)</b>: Sample at the end of runs based on volume or dollar value.


Under the <b> Sampling </b> section there exists the CUSUM (Cumulative Sum) Filter which is another important trick used in few places across chapters. 

<b>The CUSUM Filter</b>: An event-based sampling technique that triggers when the cumulative sum of price deviations from a mean crosses a predefined threshold, effectively capturing significant market events.

## Chapter 3 - Labelling

Labelling is one of the most important sections as it introduces the <b> Triple-Barrier Method </b> and <b>Meta-Labeling</b> concepts. 

### The Triple-Barrier Method

A sophisticated labeling technique that mimics how a human trader thinks about a position. For each data point, we simulate a trade and see which of three "barriers" it hits first.

* <b>Upper Barrier (Profit-Take)</b>: A price level above the entry for taking profit.
* <b>Lower Barrier (Stop-Loss)</b>: A price level below the entry for cutting losses.
* <b>Vertical Barrier (Time Limit)</b>: A maximum holding period for the trade.

<p align="center">
  <img src="readme_files/triple_visualization.png?raw=true" alt="Triple Barrier Method" title="Triple Barrier Method" width="600"/>
</p>

The resulting label is not just direction, but outcome:
* 1: The profit-take horizontal barrier was hit.
* -1: The stop-loss horizontal barrier was hit.
* 0: The time limit (vertical barrier) was hit without touching the other barriers. (This can also be labelled with -1 for binary classification)

This method creates labels that are directly tied to a trading strategy's P&L and risk profile.

### Meta-Labeling: The Two-Stage Approach

Meta-labeling is a powerful technique for improving an existing strategy. It separates the decision of what direction to bet from whether to bet at all.

It works in two stages:

* <b>Primary Model</b>: An initial, simpler model generates a prediction for the **side** of the bet (e.g., a mean-reversion model predicts "long").
* <b>Meta-Model (The "Confidence" Model)</b>: A secondary ML model is trained to predict the probability that the primary model's bet will be successful (i.e., hit the profit-take barrier). It uses the binary outcome ({0, 1} where 1 = success) from the Triple-Barrier method as its target.

#### Why Meta-Labeling is a Game-Changer

* <b> Improves Precision</b>: It acts as a sophisticated filter, screening out the primary model's low-confidence bets.
* <b> Reduces False Positives</b>: By avoiding bad trades, it significantly increases the strategy's Sharpe ratio.
* <b>Controls Sizing</b>: The probability output from the meta-model can be used to determine the size of the bet (bet more on high-confidence predictions).

<p align="center">
  <img src="readme_files/confusion_matrix.png?raw=true" alt="Confusion Matrix" title="Confusion Matrix" width="600"/>
</p>

<u>Simple Workflow</u>

* -> [Primary Model] -> Suggests a Bet (e.g., Long)
* -> [Meta-Model] -> Predicts P(Success) > threshold? 
* -> YES: Place the trade.  / -> NO: Pass on the trade.

## Chapter 4 - Sample Weights

This chapter addresses a critical side-effect of the Triple-Barrier Method which is overlapping outcomes. It explains why this is a major problem for ML models and introduces methods for measuring this overlap (uniqueness) and correcting for it using sample weights.
The core idea is that we have far fewer unique observations than our dataset size suggests.

### 📌 The Core Problem:

In financial data, labels are not IID (independent and identically distributed). Specifically:

Some events (e.g., trading signals) overlap in time, so their outcomes are not independent.

If you train a model on such overlapping events without adjusting for that, your model will overfit, double-count some information, and perform poorly out-of-sample.

### 🧠 The Key Idea:

To fix this, the chapter introduces a way to assign weights to samples (events) so that:

Events with greater uniqueness get higher weight.

Events that overlap a lot get down-weighted.

This ensures your model is trained on more independent information.

<p align="center">
  <img src="readme_files/overlapping_events.png?raw=true" alt="Overlapping Events" title="Overlapping events" width="600"/>
</p>


### Measuring Overlap and Uniqueness
#### Section 4.3: Number of Concurrent Labels
This is the simplest way to measure redundancy. It asks: At any given point in time, how many different triple-barrier windows are active?
This count, say `c_t`, measures the degree of overlap at a specific time t. A high `c_t` means that a price movement at this moment will affect many different labels, making it disproportionately influential.

The instantaneous uniqueness at time t is defined as 1 / c_t. This forms the building block for the more advanced methods.


#### Section 4.4: Average Uniqueness of a Label
This section moves from measuring overlap at a single point in time to measuring the overall uniqueness of an entire label.
A label's "average uniqueness" is calculated by averaging the instantaneous uniqueness `(1 / c_t)` over all the time steps in its evaluation window.

For a label `i` spanning `T_i` time steps, its average uniqueness `u_i` is:

```u_i = (Sum of 1/c_t for all t in label i's window) / T_i```

This `u_i` value gives us a single, powerful number representing how redundant or unique a specific training example is. This is the value we can directly use as a sample weight during model training.

#### Section 4.5: Bagging Classifiers and Uniqueness
This section provides a crucial application of the uniqueness concept, specifically for ensemble methods like Random Forest or Bagging.

Standard bagging (bootstrapping) samples data points with replacement. In finance, where labels overlap, this is dangerous. You are highly likely to select many redundant, non-unique samples, which makes your bootstrapped training sets very similar to each other. This defeats the purpose of bagging, which relies on model diversity from diverse sub-samples.

The Solution Sequential Bootstrapping suggests that instead of sampling purely at random, we can use our uniqueness measure to guide the process.
Draw samples sequentially.
For each sample drawn, assign it a uniqueness value (e.g., using the average uniqueness from 4.4).
The probability of drawing the next sample can be adjusted based on the uniqueness of the samples already drawn, ensuring the final bootstrapped set is more diverse.

## Chapter 5 - Fractionally Differentiated Features

This chapter introduces a technique for feature engineering that solves the stationarity vs memory dilemma.

### The Stationarity vs. Memory Dilemma

When preparing data for a machine learning model, we face a critical trade-off:

* <b>Stationarity</b>: ML models require data whose statistical properties (like mean and variance) are constant over time. Raw price series are non-stationary and will break most models.

* <b>Memory</b>: The original price series contains valuable, long-term trend information (its "memory") that is essential for making accurate predictions.

The dilemma is that the standard methods (i.e. working with returns) used to make a series stationary completely destroy its memory. We are forced to choose between a series that is statistically valid but predictively useless, or one that is predictive but statistically invalid.

### The Standard Solution: Integer Differentiation
The textbook approach is to compute integer differences, most commonly by calculating returns `(price_t - price_{t-1})`.
What it does: This is an "all-or-nothing" operation. It successfully transforms a non-stationary price series into a stationary returns series.
The Problem: In doing so, it erases almost all of the original series' memory. We "throw the baby out with the bathwater," leaving a series that is largely noise and very difficult to predict.

### Fractional Differentiation (FracDiff)
De Prado offers a far more elegant solution that avoids the harsh trade-off. Instead of a binary on/off switch, Fractional Differentiation acts like a "dimmer switch" for memory.

FracDiff is a technique that generalizes differentiation to a fractional order, controlled by a parameter `d`.
* d = 0: The original series (full memory, non-stationary).
* d = 1: Standard returns (no memory, stationary).
* 0 < d < 1: A new series that balances both properties.

<b>The Goal</b>: The key insight is to find the minimum d that is just high enough to make the series stationary. We apply just enough differentiation to satisfy our model's statistical needs, while preserving the maximum possible amount of the original series' predictive memory.
This allows us to create features that have the best of both worlds: they are stationary enough to be used in ML models, but still rich with the long-term memory needed to build a powerful predictive strategy.

<p align="center">
  <img src="readme_files/fractional_diff.png?raw=true" alt="Fractional Diff" title="Fractional Diff" width="600"/>
</p>

🎯 What it does in simple terms:

Instead of subtracting only the previous value (like in `P_t - P_{t-1}`),

It subtracts a weighted sum of previous values.

For example:

```new_value_t = P_t - 0.9 * P_{t-1} - 0.8 * P_{t-2} - 0.7 * P_{t-3} ...```
The weights decay over time.

Smaller d → slower decay → more memory kept.

Larger d → faster decay → less memory, more like regular differencing.

# PART II - MODELLING

## Chapter 6 - Ensemble Methods

Chapter 6 provides a timeless foundation on the principles of bagging and boosting, which remain essential knowledge. However, its toolkit is dated, as it predates the modern era of machine learning that began around its publication. The chapter underrepresents the now-ubiquitous, hyper-optimized gradient boosting libraries like XGBoost and LightGBM, and entirely omits the rise of deep learning architectures like Transformers for handling sequential data. Furthermore, it lacks modern interpretability frameworks like SHAP, which are now critical for explaining these complex models. While its concepts are fundamental, a practitioner today must supplement this chapter with these more powerful, contemporary methods.

### Three Sources of Errors

<p align="center">
  <img src="readme_files/three_sources_of_errors.png?raw=true" alt="3 Sources" title="3 Sources" width="600"/>
</p>



### Key Ensemble Techniques
There are two main families of ensemble methods, each with a different philosophy.
#### 1. Bagging (Bootstrap Aggregating)
Bagging focuses on reducing variance and creating stability by averaging out errors. It's a parallel approach where models are trained independently.

<b>The Idea</b>: Create a "committee" of diverse models by training each one on a slightly different, random subset of the data.

<b> How it Works</b>:

* Bootstrap: Create multiple training datasets by sampling with replacement from the original data.
* Train: Train one model (e.g., a Decision Tree) on each of these bootstrapped datasets.
* Aggregate: For a new prediction, let all the models "vote" (for classification) or average their outputs (for regression). The majority/average decision is the final output.
  
<b>Why it Works</b>: Individual models might be unstable and overfit to noise in their specific dataset. By averaging them, their individual errors tend to cancel each other out, leading to a much more stable and reliable final prediction.

#### 2. Random Forests
Random Forest is a powerful and popular extension of bagging, specifically for decision trees. It introduces an extra layer of randomness to make the models even more diverse.

<b>The Idea</b>: It's bagging, but with a twist to prevent the models from becoming too similar.

<b>The Key Addition</b>: When building each decision tree, at every split point, the algorithm is only allowed to consider a random subset of the features. This process is called feature bagging. It forces the trees to be different from one another. Without it, every tree might learn to rely on the same one or two "super-predictive" features. By restricting the feature choice, Random Forest builds a more diverse committee of experts, making the overall model more robust if a key feature's signal fades.

#### 3. Boosting
Boosting is a sequential approach that focuses on reducing bias and building a single, highly accurate model by learning from mistakes.
<b>The Idea</b>: Build a "chain" of weak models, where each new model is trained to correct the errors made by the previous ones.

<b> How it Works</b>:

1. Train a simple, "weak" base model on the data.
2. Identify which observations the model got wrong.
3. Train the next model, giving more weight and focus to the observations that the previous model misclassified.
4. Repeat this process, with each new model focusing on the hardest remaining cases.
5. The final prediction is a weighted sum of all the models' predictions.

<b>Why it Works</b>: It converts a series of weak learners (models that are only slightly better than random guessing) into a single, powerful "strong learner." It's an expert at finding and modeling complex, non-linear patterns.

By using these ensemble techniques, we move away from the fragile search for a single perfect model and toward building robust, diversified, and more reliable predictive systems.

## Chapter 7 - Cross Validation

This chapter addresses a common mistake in quantitative finance: using standard cross-validation (CV) techniques on financial data. A flawed validation process is the primary reason why so many strategies that look brilliant in backtests fail in live trading. This chapter provides a robust solution to prevent this.

### Why Standard K-Fold Cross-Validation Fails in Finance
Standard K-Fold CV shuffles data and splits it randomly. This works for IID (Independent and Identically Distributed) data, but financial time series are not IID. Applying standard CV to financial data leads to a critical flaw: data leakage.

<b>What is Data Leakage</b>?

The training set becomes contaminated with information from the testing set. This happens because the labels (e.g., from the Triple-Barrier Method) are a function of future data. Shuffling can place a training observation before a testing observation, while its label was determined by information that occurred after that testing observation.

<b>The Consequence:</b> The model is inadvertently trained on information from the future. Its performance in the backtest is artificially inflated because it's being evaluated on data it has already "seen." This leads to catastrophic overfitting and strategies that are guaranteed to fail.

<b>The Solution:</b> Purged and Embargoed K-Fold Cross-Validation

De Prado introduces a purpose-built CV method that respects the temporal nature of financial data and systematically eliminates leakage. It has two key components:

### 1. Purging: Removing Tainted Training Data
The first step is to clean the training set.

<b>The Idea</b>: Go through the training set and remove ("purge") any observation whose label's evaluation period overlaps with the testing period.
<b>The Result</b>: This ensures that the model is not trained on any data that could provide a hint about the testing set's outcomes.
  
  <p align="center">
  <img src="readme_files/purging.png?raw=true" alt="Purging Illustration" title="Purging: Overlapping training labels are removed." width="700"/>
  </p>

### 2. Embargoing: Creating a Buffer Zone
The second step is to prevent leakage from serial correlation (when one observation influences the next).
<b>The Idea</b>: Place a small time gap or "embargo" period immediately after the end of the training data. This data is not used for either training or testing.

<b>The Result</b>: This creates a buffer zone, ensuring that the performance on the first few test samples is not contaminated by information from the last few training samples (e.g., due to features with a look-ahead window like moving averages).

  <p align="center">
  <img src="readme_files/embargoing.png?raw=true" alt="Purging Illustration" title="Purging: Overlapping training labels are removed." width="700"/>
  </p>


## Chapter 8 - Feature Importance

  <p align="center">
  <img src="readme_files/feature_importance.png?raw=true" alt="Feature Importance" title="Feature Importance" width="400"/>
  </p>

  <p align="center">
  <img src="readme_files/mdi_mda_sfi.png?raw=true" alt="MDI vs MDA vs SFI" title="MDI vs MDA vs SFI" width="800"/>
  </p>

<b>MDA vs SFI: Which is more expensive?</b>

SFI becomes more computationally expensive than MDA as the number of features (N) becomes large.

While MDA has a high, fixed upfront cost (training the main model), SFI's cost scales directly with the number of features. In modern financial ML where datasets can have hundreds or thousands of potential features, the requirement to train a separate model for each one makes SFI the more resource-intensive method in terms of total CPU time.
However, because SFI is perfectly parallelizable, its wall-clock time can be drastically reduced if you have a multi-core machine. Even so, for very large feature sets, MDA is generally the more computationally tractable approach.

## Chapter 9 - Hyper-parameter Tuning

This chapter tackles one of the final and most dangerous sources of backtest overfitting: hyper-parameter tuning. Choosing the right hyper-parameters (e.g., the number of trees in a Random Forest, the learning rate in a GBM) is critical for model performance.


### The Problem with Standard Tuning Methods
The go-to method for hyper-parameter tuning in many ML libraries is Grid Search with K-Fold Cross-Validation (GridSearchCV). This approach is doubly flawed in finance.

<b>Combinatorial Explosion (The "Curse of Dimensionality")</b>:

Grid search is a brute-force method that exhaustively tests every possible combination of parameters.
With more than a few parameters, the number of models to train becomes computationally astronomical, making it impractical.

<b>Data Leakage (The Fatal Flaw)</b>:

Standard GridSearchCV uses standard K-Fold CV, which, as we learned in Chapter 7, is completely inappropriate for financial data.
It leaks information from the future into the past, causing the search to select hyper-parameters that are not genuinely robust but simply overfit to the test sets used during cross-validation. This is a primary cause of strategies failing in the real world.

### A Smarter, Safer Approach to Tuning

De Prado advocates for a two-part solution that is both more efficient and, crucially, more robust.

<b>1. The Search Method</b>: From Brute-Force to Intelligent Search. Instead of an exhaustive grid search, use a randomized approach.

* <b>Randomized Search (RandomizedSearchCV)</b>: 
  * Instead of testing every combination, this method randomly samples a fixed number of parameter combinations from the specified distributions.
  * It is far more efficient and often finds equally good (or better) parameters than grid search in a fraction of the time.
  
* <b>The Coarse-to-Fine Workflow (Recommended)</b>:
  * 1. Random Search: Begin with a randomized search across a wide range of parameter values.
  * 2. Analyze: Identify the "promising regions" where the best-performing parameters were found.
  * 3. Grid Search: Perform a much smaller, focused grid search only within those promising regions to fine-tune the final selection.

<b>2. The Validation Method:</b> The Foundation of Reliability

This is the most critical part. The search method (random or grid) must be combined with the robust cross-validation technique from Chapter 7.

* Use Purged K-Fold Cross-Validation:
  * When performing the randomized or grid search, you must use a Purged K-Fold CV object as the cross-validation splitter.
  * This ensures that every evaluation performed during the hyper-parameter search is free from data leakage. Each fold is properly purged and embargoed.

<b>The Final Workflow</b>

```Hyper-Parameter Tuning = (Random Search + Focused Grid Search) + Purged K-Fold CV```


# PART III - BACKTESTING

## Chapter 10 - Bet Sizing

<b><i>"How much should I bet?"</i></b> Getting the direction right is only half the battle. A model that makes many correct but low-conviction predictions can still lose money if it bets too much on the wrong trades. The chapter argues that the size of our position should be dynamic and directly proportional to the model's confidence in its prediction.


### The Core Idea: Size Bets Based on Confidence

The central principle remains the same: the size of our position should be a direct function of the model's confidence. High-confidence predictions should lead to larger bets, while low-confidence predictions should lead to smaller bets or no bet at all. This is where the Meta-Labeling technique from Chapter 3 is invaluable, as its purpose is to generate an accurate probability (p) of a strategy's success.

<b>De Prado's Sizing Function</b>: The S-Shaped Curve

Instead of a simple linear mapping, De Prado proposes using a function that generates an S-shaped (sigmoid) curve. This is a much safer and more realistic approach.

<b>The Function</b>: The probability p is first converted into a standardized variable z, and then plugged into the Cumulative Distribution Function (CDF) of the standard Normal distribution.

```
1. z = (p - 0.5) / sqrt(p * (1-p))
2. Bet Size = 2 * N(z) - 1, where N() is the Normal CDF.
```

<b>Why an S-Curve is Superior:</b>

* <b>It's Conservative</b>: For probabilities close to 0.5 (low conviction), the bet size remains very small. It doesn't increase linearly.
* <b>It Ramps Up for High Conviction</b>: The bet size only grows substantially when the model's probability p moves significantly away from 0.5 towards 0 or 1.
* <b>It Prevents Over-Betting</b>: This non-linear mapping prevents the model from taking excessive risk on marginal signals, which is a major source of losses in strategies that use linear sizing.

  <p align="center">
  <img src="readme_files/betsize_s_curve.png?raw=true" alt="Bet Size S Curve" title="Bet Size S Curve" width="600"/>
  </p>

## Chapter 11 - The Dangers of Backtesting

This chapter is an easier read yet, it is very informative about commonpitfalls. Below figure represents the 7 sins but giving the whole chapter a read
would still be helpful and practical when backtesting ML models.


Seven Sins of Quantitative Investing” (Luo et al. [2014])
  <p align="center">
  <img src="readme_files/7_backtesting_sins.png?raw=true" alt="7 Sins of Backtesting" title="7 Sins of Backtesting" width="600"/>
  </p>

## Chapter 12 - Backtesting through Cross-Validation

### The Problem with Traditional Walk-Forward Backtesting

The standard industry approach to backtesting is "walk-forward," where a model is trained on a period of data and tested on the subsequent period, rolling this window through time.

<b>The Flaw (Path-Dependence)</b>: This method evaluates the strategy on only one single path through history—the chronological one. It also does not utilize the data to its full poteintial. A strategy's entire backtest performance can be made or broken by a single lucky or unlucky period (like a crisis). It doesn't tell you if the strategy's logic is fundamentally sound, only how it performed on that one specific sequence of events. This makes it a poor estimator of future performance.

<b>The Solution</b>: Backtesting as a Cross-Validation Problem

De Prado's solution is to re-frame backtesting not as a single simulation, but as a cross-validation exercise. The goal is to test the strategy's logic across many different historical scenarios, not just one.

<b>The Main Tool</b>: Combinatorial Purged Cross-Validation (CPCV)

This is the chapter's core technique for implementing a robust backtest.

* The Idea: Instead of one chronological path, we create many different historical paths by combining different data segments for training and testing.
* The Workflow:
1. Split the Data: Divide the entire dataset into N distinct, non-overlapping groups (e.g., 6 groups of 4 months each for a 2-year backtest).
2. Form All Combinations: Create all possible train/test splits by taking every combination of k groups for training. For example, from the 6 groups, you would test all 15 combinations of 4 groups for training and 2 for testing.
3. Run a Backtest on Each Path: For each of these combinatorial paths, run a full backtest. Crucially, each test split within a path must use the purging and embargoing techniques from Chapter 7 to prevent data leakage.
4. Aggregate the Results: The final output is not a single performance metric (like one Sharpe Ratio), but a distribution of performance metrics from all the tested paths.

  <p align="center">
  <img src="readme_files/paths_generated.png?raw=true" alt="Paths Generated" title="Paths Generated" width="600"/>
  </p>

  <p align="center">
  <img src="readme_files/path_assignment.png?raw=true" alt="Path Assignment" title="Path Assignment" width="600"/>
  </p>

</p>

### Why This is a Game-Changer
* <b>Path-Independence</b>: By averaging performance across many paths, the final result is not dependent on the luck of one specific historical sequence. It measures the robustness of the strategy's underlying logic.
* <b>Provides a Distribution of Outcomes</b>: You get a full distribution of Sharpe Ratios. This allows you to assess the strategy's risk profile, such as the probability of failure (P[SR < 0]) and the stability of its performance.
* <b>More Reliable Estimate</b>: The average performance across all combinatorial paths is a much more reliable and unbiased estimator of the strategy's true out-of-sample performance than a single walk-forward test.


## Chapter 13 - Synthetic Data

This chapter addresses a fundamental limitation of all backtesting: we only have one realization of history. A strategy might look great on the single historical path we have, but what if that path was unusually kind? This chapter provides the ultimate stress test by generating thousands of alternative, plausible historical paths—synthetic datasets—and backtesting our strategy on all of them.

To create Synthetic Data that aligns with the properties of real data 13.4 introduces the Ornstein-Uhlenbeck (O-U) based framework for generating synthetic prices.

Afterwards, 13.5 builts the Algorithm on top via the 5 Step Model below:

<b>Step 1</b>: Model the Price Dynamics

<b>What it does</b>: The algorithm first analyzes the historical price data to understand its mean-reverting behavior. It fits a statistical model (an Ornstein-Uhlenbeck process) to the data.
<b>The Output</b>: This yields two key parameters: φ (the speed of mean-reversion) and σ (the volatility of the process). These two numbers effectively become the "DNA" for the price behavior we want to simulate.

<b>Step 2</b>: Define a Grid of Potential Trading Rules
<b>What it does</b>: A comprehensive grid (or "mesh") of potential trading rules is created. Each rule is a pair consisting of a stop-loss level and a profit-taking level, both defined in terms of the volatility σ calculated in Step 1.
Example: It might create a 20x20 grid, testing stop-losses from -0.5σ to -10σ against profit-takes from +0.5σ to +10σ, resulting in 400 unique rule combinations.

<b>Step 3</b>: Generate Thousands of Synthetic Price Paths
<b>What it does</b>: Using the mean-reversion speed (φ) and volatility (σ) from Step 1, the algorithm generates a large number of new, synthetic price paths (e.g., 100,000).
<b>The Key</b>: Each path starts from the observed initial conditions of a real trading opportunity and simulates what could have happened next, according to the statistical properties of the model. A maximum holding period (a vertical barrier) is also imposed.


<b>Step 4</b>: Run a Massive Backtesting Experiment
<b>What it does</b>: This is the heart of the process. The algorithm takes every single trading rule from the grid in Step 2 and backtests it against every single one of the 100,000 synthetic paths from Step 3.
<b>The Output</b>: This doesn't produce one result, but a distribution of outcomes (e.g., 100,000 Sharpe Ratios) for each of the 400 trading rules.


<b>Step 5</b>: Determine the Optimal Trading Rules

<b>What it does</b>: The algorithm analyzes the massive set of results from Step 4 to find the best-performing rule. This can be done in three ways:
  * 5a (Unconstrained): Find the single best-performing stop-loss/profit-take pair from the entire grid.
  * 5b (Constrained Profit-Take): If your strategy already has a fixed profit target, use the results to find the optimal stop-loss that should accompany it.
  * 5c (Constrained Stop-Loss): If your fund has a mandatory maximum stop-loss, use the results to find the optimal profit-taking level to maximize returns for that given level of risk.

## Chapter 14 - Backtest Statistics

This chapter introduces the methods and indicators used in the industry for evaluating a strategies performance. It also builds on top of 
the standardized Sharpe Ratio by introducing more robust performance evaluation metrics such as Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).



### General Characteristics
This section provides metrics that describe the fundamental operational nature, style, and potential biases of the strategy.

* <b>Time range</b>: The start and end dates of the backtest. A longer period covering multiple market regimes is essential for robustness.
* <b>Average AUM</b>: The average dollar value of Assets Under Management.
* <b>Capacity</b>: The highest AUM the strategy can manage before performance degrades due to transaction costs and market impact.
* <b>Leverage</b>: The amount of borrowing used, measured as the ratio of the average dollar position size to the average AUM.
* <b>Maximum dollar position size</b>: The largest single position taken during the backtest. A value close to the average AUM is preferred, as it indicates the strategy doesn't rely on rare, extreme bets.
* <b>Ratio of longs</b>: The proportion of bets that were long positions. For a market-neutral strategy, this should be close to 0.5. A significant deviation indicates a potential directional bias.
* <b>Frequency of bets</b>: The number of bets per year. A "bet" is a complete cycle from a flat position to another flat position or a flip, not to be confused with the number of trades.
* <b>Average holding period</b>: The average number of days a bet is held.
* <b>Annualized turnover</b>: The ratio of the average dollar amount traded per year to the average annual AUM. This measures how actively the portfolio is managed.
* <b>Correlation to underlying</b>: The correlation between the strategy's returns and the returns of its investment universe. A low correlation is desired to prove the strategy is generating unique alpha.

### Performance
This section lists the raw, unadjusted metrics that describe the strategy's absolute profitability before risk adjustments.

* <b>PnL</b>: The total profit and loss in dollars (or the currency of denomination) over the entire backtest, including costs.
* <b>PnL from long positions</b>: The portion of the total PnL generated exclusively by long positions. This is useful for assessing directional bias in long-short strategies.
* <b>Annualized rate of return</b>: The time-weighted average annual rate of total return, as calculated by the TWRR method to correctly account for cash flows.
* <b>Hit ratio</b>: The fraction of bets that resulted in a positive PnL.
* <b>Average return from hits</b>: The average PnL for all profitable bets.
* <b>Average return from misses</b>: The average PnL for all losing bets.

### Runs and Drawdowns

This category assesses the path-dependency and risk profile of the returns, which is crucial for non-IID financial series.

* <b>Drawdown (DD)</b>: The maximum loss from a portfolio's peak value (high-water mark).
* <b>Time Under Water (TuW)</b>: The longest time the strategy spent below a previous high-water mark.
* <b>Returns Concentration (HHI)</b>: A key metric inspired by the Herfindahl-Hirschman Index. It measures whether PnL comes from a few massive wins (risky) or is distributed evenly across many small wins (robust). This is measured for both positive and negative returns, as well as concentration in time (e.g., all profits came in one month).

### Implementation Shortfall
This category grounds the backtest in reality by analyzing its sensitivity to real-world trading costs.

* <b>Costs vs. PnL</b>: Measures performance relative to trading costs (brokerage fees, slippage).
* <b>Return on Execution Costs</b>: A crucial ratio that shows how many dollars of profit are generated for every dollar spent on execution. A high multiple is needed to ensure the strategy can survive worse-than-expected trading conditions.

### Efficiency (Risk-Adjusted Performance)
<u>This is where the chapter introduces its most powerful statistical tools for judging performance after accounting for risk and selection bias.</u>

* <b>Sharpe Ratio (SR)</b>: The standard but flawed metric.
* <b>Probabilistic Sharpe Ratio (PSR)</b>: A superior metric that adjusts the SR for non-Normal returns (skewness, fat tails). It estimates the probability that the true SR is positive.
* <b>Deflated Sharpe Ratio (DSR)</b>: It's a PSR that also corrects for selection bias by penalizing the result based on the number of strategies tried. It answers the question: "What is the probability this result is a fluke?"

### Classification Scores
These metrics are specifically for evaluating the performance of the meta-labeling model from Chapter 3.
* <b>Accuracy, Precision, Recall</b>: Standard classification metrics.
* <b>F1-Score</b>: The harmonic mean of precision and recall, which is a much better metric than accuracy when dealing with the imbalanced datasets typical in finance (i.e., many more "pass" signals than "bet" signals).
  
### Attribution
This category seeks to understand where the PnL comes from by decomposing performance across different risk factors (e.g., duration, credit, sector, currency). This helps identify the true source of a portfolio manager's skill.

## Chapter 15 - Understanding Strategy Risk
This is a lighter chapter with the core objective of quantifying the strategy risk. It models a strategy as a series of binomial bets (profit or loss outcomes) to understand how sensitive its success is to its core parameters: betting frequency, precision, and the size of its wins and losses. The analysis progresses from a simplified model to a more realistic one.

### The Symmetric Payouts Model
The model assumes a strategy consists of a series of independent bets where:
* There are n bets per year (frequency).
* Each bet has a probability p of winning (precision).
* <b><u>The payouts are symmetric</u></b>: a win yields a profit of +π and a loss yields an identical loss of -π.

Under these assumptions, the chapter derives the formula for the annualized Sharpe Ratio (θ). This derivation leads to a critical insight:

In the symmetric case, the payout size π cancels out of the Sharpe Ratio formula. The strategy's risk-adjusted performance depends only on its precision (p) and frequency (n).

  <p align="center">
  <img src="readme_files/symmetric_payouts.png?raw=true" alt="Symmetric Payouts" title="Symmetric Payouts" width="300"/>
  </p>

This simplified model provides that a strategy's success is a function of its statistical properties, not necessarily the size of its individual bets.

### The Asymmetric Payouts Model
This section lifts off the key constraint of the first model where returns are identical to build a more realistic and powerful framework for evaluating real-world strategies.

The model defines for asymmetric payouts by:
* A winning bet yields a profit of π+.
* A losing bet results in a loss of π-.

π+ does not have to be equal to |π-|.

With this model, the Sharpe Ratio (θ) is now a function of all four parameters: precision (p), frequency (n), profit target (π+), and stop-loss (π-). Payouts no longer cancel out. The Sharpe Ratio formula becomes:

  <p align="center">
  <img src="readme_files/asymmetric_payouts.png?raw=true" alt="Symmetric Payouts" title="Symmetric Payouts" width="300"/>
  </p>

Onwards, there are visuals of the strategy results with varying combinations of these parameters.


# PART IV - USEFUL FINANCIAL FEATURES

## Chapter 17 - Structural Breaks

This Chapter introduces CUSUM and Explosiveness Tests to identify structural breaks.

<b>CUSUM tests</b>:The CUSUM (Cumulative Sum) filter introduced in Chapter 2 for sampling bars whenever some variable, like cumulative prediction errors, exceeded a predefined threshold. This concept is further extended to test for structrual breaks 

<b>Explosiveness tests</b>: Beyond deviation from white noise, these test whether the process exhibits exponential growth or collapse, as this is inconsistent with a random walk or stationary process, and it is unsustainable in the long run.

| Test Name | Core Idea | How it Works | Key Advantages | Key Drawbacks / Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Brown-Durbin-Evans CUSUM** | Detects breaks by testing if the cumulative sum of recursive forecasting errors deviates from a baseline of zero. | Uses Recursive Least Squares (RLS) to get 1-step-ahead prediction errors based on a feature set `x_t`. The cumulative sum of these standardized errors (`St`) is tested for statistical significance. | Incorporates the predictive power of external features (`x_t`) into the break detection process. | The results can be sensitive and inconsistent due to the arbitrary choice of the regression's starting point. |
| **Chu-Stinchcombe-White CUSUM** | A simplified CUSUM test that works directly on a price series by assuming a "no change" forecast and detecting deviations from a reference point. | Computes the cumulative standardized deviation of the current price from a past reference price (`yn`). A significant deviation implies a break. | Computationally much simpler than the Brown-Durbin-Evans test as it does not require external features or recursive regressions. | Also suffers from the arbitrary choice of a reference level (`yn`), which can affect the results. |
| **Chow-Type Dickey-Fuller** | A basic test designed to detect a *single* switch from a random walk to an explosive process at a *known* date. | It fits an autoregressive model using a dummy variable `D_t` that "activates" an explosive term after a pre-specified break date `τ*`. | Conceptually simple and easy to implement. | Highly impractical for finance as it requires knowing the break date `τ*` in advance and assumes only one break occurs. |
| **Supremum ADF (SADF)** | The chapter's flagship method for detecting periodically collapsing bubbles without prior knowledge of the number or timing of the breaks. | Uses a double-loop algorithm. The outer loop advances the window's endpoint `t`. The inner loop runs ADF tests on all backward-expanding windows `[t0, t]`. The SADF statistic is the `supremum` (maximum) ADF value found in the inner loop. | Highly effective at detecting multiple, overlapping bubbles and their subsequent collapses. Does not require any prior assumptions about break dates. | **Extremely computationally expensive (`O(T^2)`).** The `supremum` statistic is very sensitive to single outliers, which can make it noisy. |
| **Quantile ADF (QADF)** | A robust enhancement to the SADF test that is less sensitive to single outliers. | Instead of taking the absolute `supremum` (maximum) ADF statistic from the inner loop, it uses a high quantile (e.g., the 95th percentile) of the distribution of ADF statistics. | Provides a more stable and robust measure of "market explosiveness" compared to the standard SADF test. | Slightly more complex to calculate and requires choosing a quantile level (`q`). |
| **Conditional ADF (CADF)** | A further enhancement to SADF that measures the *central tendency* of the right tail of the ADF distribution, making it even more robust. | It calculates the *average* of all ADF statistics from the inner loop that fall *above* a certain high quantile (e.g., above the 95th percentile). | Even more robust to extreme outliers than QADF because it averages the tail rather than picking a single point from the distribution. | Adds another layer of computational complexity. |
| **Sub/Super Martingale Tests** | A family of alternative explosiveness tests that use different functional forms (not the ADF's autoregressive model) to detect bubbles. | Fits polynomial, exponential, or power-law trends to the data within the same double-loop framework as SADF. Includes a penalty term `(t-t0)^φ` to adjust the test's sensitivity to long-run vs. short-run bubbles. | Offers greater flexibility by not being tied to ADF's specific model assumptions. The `φ` parameter allows the test to be tuned for specific investment horizons. | Requires choosing an appropriate functional form (e.g., polynomial, exponential) and tuning the horizon parameter `φ`. |


This chapter also notes the existence of literature studies carrying out structural breaks on raw prices.
However, log prices are better due to their more preferable properties:

<b>They Ensure Time-Symmetric Returns</b>: The magnitude of a log return is the same whether a price goes up or down. Simple percentages are not symmetric because the base of the calculation changes.

<b>Example</b>: A move from $10 to $15 is a +50% simple return. The reverse move from $15 to $10 is a -33.3% simple return. With log returns, the move up is ln(15/10) ≈ +0.405, and the move down is ln(10/15) ≈ -0.405. The magnitude is identical.

<b>They Make Returns Additive Over Time</b>: Simple returns are multiplicative, which is mathematically inconvenient. Log returns are additive, making them much easier to aggregate and analyze over time.

<b>Example</b>: A stock goes from $10 -> $15 -> $18. The simple returns are +50% and +20%. The total return is +80%, which is not 50%+20%.

The log returns are ln(1.5)≈0.405 and ln(1.2)≈0.182. The total log return is ln(1.8)≈0.587, which is the sum of the individual log returns.

<b>Statistical Validity (Homoscedasticity)</b>: Log prices lead to a statistically valid model where return volatility is assumed to be constant, avoiding the unrealistic assumptions and errors (heteroscedasticity) that arise when testing raw prices. This is specicially critical for ADF/SADF test.

<b>Example</b>:<br>
With raw prices:<br>
A stock goes from $10 -> $10.5. Return is $0.5.<br>
Same stock a year later goes from $100 -> $105. Return is $5.<br>
Although the volatility is the same 5%, returns in dollar terms has changed which would break the model.

With logs:<br>
$10 - $10.5. Return log(10.5) - log(10) = log(10.5/10) = 0.048 vs. <br>
$100 - $105. Return log(105) - log(100) = log(105/100) = 0.48


