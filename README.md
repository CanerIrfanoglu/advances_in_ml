# INTRODUCTION

Advances In Financial Machine Learning is consisted of 5 Parts (Data Analysis, Modelling, Backtesting, Useful Financial Features and High-Performance Computing Receipes) with multiple Chapters under each part. The book is highly technical utilizing advanced mathematical equations frequently. Therefore one needs to study concepts introduced under each chapter to get the maximum benefit. With that said, this repository attempts reducing this density by higlighting the most important concepts, providing chapter summaries as well as the exercise solutions using sample bitcoin data.

Exercises for following chapters are not included:

Chapter 11 - The Dangers of Backtesting: This chapter is a warning mentioning common sins and exercises are covering cases where certain sins are committed.

Chapter 16 - Machine Learning Asset Allocation: Skipped section since the concentration of this study is to concentrate initially on trading applications.

Chapter 20/21/22 - These are sections belonging to High-Performance Computing Recipes Part. Previously utilized `mpPandasObj` parallelization function provided under Chapter 20. It would be ideal to refer this sections when training the models with actual vast amounts of data rather than exercise samples.

# Chapter 1 - PREAMBLE - Financial Machine Learning as a Distinct Subject

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

# PART I -  DATA ANALYSIS

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




