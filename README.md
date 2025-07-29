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
* Chapter 2 - Financial Data Structures
* Chapter 3 - Labelling
* Chapter 4 - Sample Weights
* Chapter 5 - Fractionally differentiated Features

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

