from sklearn.datasets import make_classification
import pandas as pd
import datetime
# Using the function getTestData from Chapter 8, form a synthetic dataset of
# 10,000 observations with10 features, where 5 are informative and 5 are noise.
def getTestData(n_features=10, n_informative=5, n_redundant=0, n_samples=10000):
    # generate a random dataset for a classification problem
    trnsX_data, cont_data = make_classification( # Renamed to avoid conflict with DataFrame trnsX
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=0,
        shuffle=False
    )

    # 1. Create the DatetimeIndex and store it in df0_index
    df0_index = pd.date_range(
        end=datetime.datetime.today(),
        periods=n_samples,
        freq=pd.tseries.offsets.BDay() # Or freq='B'
    )

    # 2. Use df0_index when creating the DataFrame and Series
    # Also, note the tuple assignment: trnsX, cont = (DataFrame(...), Series(...).to_frame())
    # It's clearer to assign them separately if they don't depend on each other in one line.
    trnsX = pd.DataFrame(trnsX_data, index=df0_index)
    cont = pd.Series(cont_data, index=df0_index).to_frame('bin')

    # 3. Create column names and store them in a separate variable
    # Use range() instead of xrange()
    feature_column_names = ['I_' + str(i) for i in range(n_informative)] + \
                           ['R_' + str(i) for i in range(n_redundant)]
    feature_column_names += ['N_' + str(i) for i in range(n_features - len(feature_column_names))]
    trnsX.columns = feature_column_names

    cont['w'] = 1. / cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index) # This creates a column t1 with the index values
    return trnsX, cont

X, y_df = getTestData()

# (a) Use GridSearchCV on 10-fold CV to find the C, 
# gamma optimal hyper-parameters on a SVC with RBF kernel, where param_grid={'C':[1E-
# 2,1E-1,1,10,100],'gamma':[1E-2,1E-1,1,10,100]} and the scoring function is neg_log_loss.
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("\n--- Part (a): GridSearchCV for SVC ---")

# Define the parameter grid
param_grid = {
    'svc__C': [1E-2, 1E-1, 1, 10, 100],         # Note: 'svc__' prefix for pipeline
    'svc__gamma': [1E-2, 1E-1, 1, 10, 100]
}

# SVCs are sensitive to feature scaling. It's best practice to include scaling in the pipeline.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', probability=True, random_state=42)) # probability=True is essential for neg_log_loss
])

# Define the 10-fold Cross-Validation strategy
# For time series data, PurgedKFold from previous exercises would be better,
# but the problem simply states "10-fold CV". Standard KFold is fine for i.i.d. data.
# Since make_classification with shuffle=False might have some order, KFold with shuffle=True is safer.
cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_log_loss', # The specified scoring function
    cv=cv_strategy,         # 10-fold cross-validation
    n_jobs=-1,              # Use all available CPU cores
    verbose=1               # Show progress
)

# Prepare data for fitting
y_target = y_df['bin'] # This is the Series of 0s and 1s

print("Starting GridSearchCV for SVC...")
# Fit GridSearchCV to the data
# X is already a DataFrame, y_target is a Series
grid_search.fit(X, y_target)

# Print the best parameters and the corresponding score
print("\nBest parameters found by GridSearchCV:")
# {'svc__C': 10, 'svc__gamma': 0.1} # took ~5 mins
print(grid_search.best_params_)
print("\nBest neg_log_loss score achieved:")
# -0.23390148682216455
print(grid_search.best_score_)

# You can also access the best estimator (the pipeline with best hyperparameters)
best_svc_pipeline = grid_search.best_estimator_
print("\nBest SVC pipeline found:")
print(best_svc_pipeline)

# To see all results (optional):
# cv_results_df = pd.DataFrame(grid_search.cv_results_)
# print("\nFull CV results:")
# print(cv_results_df[['param_svc__C', 'param_svc__gamma', 'mean_test_score', 'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score').head())

from sklearn.datasets import make_classification
import pandas as pd
import datetime
import numpy as np # For 1E-2 notation
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time # For timing

# Adjusted getTestData for the new problem
def getTestData(n_features=10, n_informative=5, n_redundant=0, n_samples=10000, random_state=0):
    trnsX_data, cont_data = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=n_redundant,
        n_clusters_per_class=1, random_state=random_state, shuffle=False
    )
    df0_index = pd.date_range(
        end=datetime.datetime.today(), periods=n_samples,
        freq=pd.tseries.offsets.BDay()
    )
    trnsX = pd.DataFrame(trnsX_data, index=df0_index)
    cont = pd.Series(cont_data, index=df0_index).to_frame('bin')
    feature_column_names = []
    if n_informative > 0: feature_column_names += ['I_' + str(i) for i in range(n_informative)]
    if n_redundant > 0: feature_column_names += ['R_' + str(i) for i in range(n_redundant)]
    n_noise_features = n_features - len(feature_column_names)
    if n_noise_features > 0: feature_column_names += ['N_' + str(i) for i in range(n_noise_features)]
    if len(feature_column_names) < n_features:
         feature_column_names += [f'F_{i}' for i in range(n_features - len(feature_column_names))]
    trnsX.columns = feature_column_names[:n_features]
    cont['w'] = 1. / cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index)
    return trnsX, cont

X, y_df = getTestData(n_features=10, n_informative=5, n_redundant=0, n_samples=10000, random_state=42)

# --- Part (a) setup ---
param_grid = {
    'svc__C': [1E-2, 1E-1, 1, 10, 100],
    'svc__gamma': [1E-2, 1E-1, 1, 10, 100]
}
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', probability=True, random_state=42))
])
cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipeline, param_grid=param_grid,
    scoring='neg_log_loss', cv=cv_strategy,
    n_jobs=-1, verbose=1
)
y_target = y_df['bin']

# --- Timing the fit for question (d) ---
print("Starting GridSearchCV for SVC...")
start_time = time.time()
grid_search.fit(X, y_target)
end_time = time.time()
duration_seconds = end_time - start_time
print(f"GridSearchCV fitting completed in {duration_seconds:.2f} seconds.")

# --- Answering the follow-up questions ---

# (b) How many nodes are there in the grid?
num_C_values = len(param_grid['svc__C'])
num_gamma_values = len(param_grid['svc__gamma'])
num_nodes_in_grid = num_C_values * num_gamma_values
print(f"\n(b) Number of nodes (hyperparameter combinations) in the grid: {num_nodes_in_grid}")

# (c) How many fits did it take to find the optimal solution?
num_cv_folds = cv_strategy.get_n_splits() # Gets n_splits from the CV object
total_fits = num_nodes_in_grid * num_cv_folds
print(f"(c) Total number of model fits performed: {total_fits}")

# (d) How long did it take to find this solution?
# This was already calculated and printed above after the .fit() call.
print(f"(d) Time taken to find the solution: {duration_seconds:.2f} seconds.")

# (e) How can you access the optimal result?
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_ # This is the pipeline refit on all data with best_params_
print("(e) Accessing the optimal result:")
print(f"  - Best hyperparameters: {best_params}")
print(f"  - Best estimator (pipeline): {best_estimator}")

# (f) What is the CV score of the optimal parameter combination?
best_cv_score = grid_search.best_score_
print(f"(f) CV score (neg_log_loss) of the optimal parameter combination: {best_cv_score:.4f}")

# (g) How can you pass sample weights to the SVC?
print("(g) How to pass sample weights to SVC via GridSearchCV:")
print("  You pass sample weights to the `fit` method of GridSearchCV.")
print("  The parameter name must be prefixed with the estimator's name in the pipeline, like `svc__sample_weight`.")
print("  Example: `grid_search.fit(X, y_target, svc__sample_weight=sample_weights_array)`")
print("  Where `sample_weights_array` is a NumPy array or list of the same length as X, containing the weight for each sample.")
print(f"  In our case, if we wanted to use the weights from y_df['w'], it would be: `grid_search.fit(X, y_target, svc__sample_weight=y_df['w'].values)`")

print("\n--- What are sample weights in this context? ---")
print("""
Sample weights allow you to assign different levels of importance to different training samples
during the model fitting process. When a model (like SVC) calculates its loss function,
the error contribution of each sample is multiplied by its weight.

Reasons to use sample weights:
1.  **Imbalanced Classes:** If one class has far fewer samples than another (e.g., fraud detection),
    you can give a higher weight to samples from the minority class. This helps prevent the
    model from being biased towards the majority class and simply ignoring the minority.
    `SVC` has a `class_weight='balanced'` option that does this automatically, but explicit
    `sample_weight` offers more fine-grained control.

2.  **Data Quality/Reliability:** If you know some data points are more reliable or 'cleaner'
    than others, you can assign them higher weights.

3.  **Importance of Observations (e.g., in Finance):**
    *   **Time Decay:** Older data might be less relevant than recent data. Weights could decrease for older samples.
    *   **Uniqueness/Information Content:** As seen in Lopez de Prado's work (like Chapter 8 exercises),
      weights can be derived based on the uniqueness of labels or information overlap,
      giving more importance to observations that provide new, non-redundant information.
      The `cont['w']` column in the `getTestData` function was initialized to uniform weights
      (1/N), meaning all samples were initially considered equally important. If this column
      contained varying weights, those could be passed.

How SVC uses them:
The SVC algorithm tries to find a hyperplane that best separates the classes while maximizing
the margin. The optimization problem involves minimizing a loss function that includes terms
for misclassifications. When sample weights are provided, these terms in the loss function
are scaled by the corresponding sample's weight. Thus, the algorithm will try harder to
correctly classify (or reduce the margin error for) samples with higher weights.
""")



print("-----------------------------------------------------------\n")
best_params_gridsearch = grid_search.best_params_
best_cv_score_gridsearch = grid_search.best_score_
print(f"Exercise 1 GridSearchCV - Best Params: {best_params_gridsearch}")
print(f"Exercise 1 GridSearchCV - Best Score: {best_cv_score_gridsearch:.4f}")
# 9.2
# (a) Use RandomizedSearchCV on 10-fold CV to find the C,
# gamma optimal hyper-parameters on an SVC with RBF kernel,
# where param_distributions={'C':logUniform(a=1E-2,b=
# 1E2),'gamma':logUniform(a=1E-2,
# (a) Use RandomizedSearchCV
print("--- Part (a): RandomizedSearchCV for SVC ---")
from scipy.stats import loguniform
import time
# Define the parameter distributions
param_distributions = {
    'svc__C': loguniform(1E-2, 1E2),       # Samples from a log-uniform distribution
    'svc__gamma': loguniform(1E-2, 1E2)
}

# Pipeline remains the same
pipeline_random = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', probability=True, random_state=42)) # probability=True for neg_log_loss
])

# CV strategy remains the same
cv_strategy_random = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize RandomizedSearchCV
# n_iter=25: It will sample 25 different combinations of parameters.
# random_state in RandomizedSearchCV ensures reproducibility of which combinations are chosen.
random_search = RandomizedSearchCV(
    estimator=pipeline_random,
    param_distributions=param_distributions,
    n_iter=25,                  # Number of parameter settings that are sampled
    scoring='neg_log_loss',
    cv=cv_strategy_random,
    n_jobs=-1,
    verbose=1,
    random_state=42             # For reproducibility of the random sampling of parameters
)

print("Starting RandomizedSearchCV for SVC...")
start_time_random = time.time()
random_search.fit(X, y_target)
end_time_random = time.time()
duration_seconds_random = end_time_random - start_time_random
print(f"RandomizedSearchCV fitting completed in {duration_seconds_random:.2f} seconds.")

# --- Answering the follow-up questions for RandomizedSearchCV ---

# (b) How long did it take to find this solution?
print(f"\n(b) Time taken for RandomizedSearchCV: {duration_seconds_random:.2f} seconds.")

# (c) Is the optimal parameter combination similar to the one found in exercise 1?
best_params_randomsearch = random_search.best_params_
print("\n(c) Comparison of optimal parameters:")
print(f"  - GridSearchCV (Exercise 1) Best Params: {best_params_gridsearch}")
print(f"  - RandomizedSearchCV Best Params:      {best_params_randomsearch}")
# You'll need to manually compare these. They might be similar if RandomizedSearch happened
# to sample near the grid points, or they could be different, potentially even better if
# the true optimum lies between the grid points of GridSearchCV.

# (d) What is the CV score of the optimal parameter combination?
best_cv_score_randomsearch = random_search.best_score_
print(f"\n(d) CV score (neg_log_loss) from RandomizedSearchCV: {best_cv_score_randomsearch:.4f}")
print(f"    For comparison, CV score from GridSearchCV:       {best_cv_score_gridsearch:.4f}")

print("\n--- Analysis of Results (c) & (d) ---")
if best_cv_score_randomsearch > best_cv_score_gridsearch: # neg_log_loss, so higher (closer to 0) is better
    print("RandomizedSearchCV found a better or equal CV score.")
    if np.isclose(best_cv_score_randomsearch, best_cv_score_gridsearch):
         print("The scores are very close, suggesting both methods performed similarly well.")
    else:
         print("RandomizedSearchCV potentially explored a more promising region of the hyperparameter space.")
else:
    print("GridSearchCV found a better CV score in this instance.")
    if np.isclose(best_cv_score_randomsearch, best_cv_score_gridsearch):
         print("The scores are very close, suggesting both methods performed similarly well.")
    else:
        print("This could be due to RandomizedSearchCV not sampling the 'best' region within its 25 iterations, or GridSearchCV's grid happened to cover the optimum well.")

print("\nParameter comparison detail:")
c_grid = best_params_gridsearch.get('svc__C')
gamma_grid = best_params_gridsearch.get('svc__gamma')
c_random = best_params_randomsearch.get('svc__C')
gamma_random = best_params_randomsearch.get('svc__gamma')

if c_grid is not None and c_random is not None:
    print(f"  C: Grid={c_grid:.4f}, Random={c_random:.4f}")
    if np.isclose(c_grid, c_random, rtol=0.5): # Allow 50% relative tolerance for "similar"
        print("     C values are somewhat similar.")
    else:
        print("     C values are different.")

if gamma_grid is not None and gamma_random is not None:
    print(f"  Gamma: Grid={gamma_grid:.4f}, Random={gamma_random:.4f}")
    if np.isclose(gamma_grid, gamma_random, rtol=0.5):
        print("     Gamma values are somewhat similar.")
    else:
        print("     Gamma values are different.")

print("\nNote: 'Similarity' is subjective. RandomizedSearchCV samples from continuous distributions,")
print("so exact matches to GridSearchCV's discrete points are unlikely. We look for values in the same order of magnitude or region.")





import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score # For scoring='accuracy'
# (Assuming all previous code from your script, including getTestData, GridSearchCV, RandomizedSearchCV setups are present)

# --- Helper function for Sharpe Ratio ---
def calculate_sharpe_ratio(returns_series, periods_per_year=252):
    """
    Calculates the (annualized) Sharpe ratio for a series of returns.
    If periods_per_year is None, returns the non-annualized Sharpe.
    """
    if returns_series.std() == 0: # Avoid division by zero if all returns are the same (e.g., all zero)
        return 0.0 if returns_series.mean() == 0 else np.inf * np.sign(returns_series.mean())
    sharpe = returns_series.mean() / returns_series.std()
    if periods_per_year is not None:
        sharpe *= np.sqrt(periods_per_year)
    return sharpe

def generate_predictions_and_returns(model, X_data, y_true_binary, assumed_positive_return=0.01, assumed_negative_return=-0.01):
    """
    Generates model predictions, maps them to positions, and calculates strategy returns.
    y_true_binary: Series of 0s and 1s (actual outcomes)
    """
    # Get in-sample predictions (binary 0 or 1)
    # Note: For SVC with probability=True, predict_proba gives probabilities, predict gives class labels
    in_sample_preds_binary = model.predict(X_data)

    # Convert binary predictions to positions: +1 for class 1, -1 for class 0
    positions = pd.Series(np.where(in_sample_preds_binary == 1, 1, -1), index=X_data.index)

    # Simulate actual market returns based on y_true_binary
    # If true outcome was 1 (e.g. price up), market_return is positive
    # If true outcome was 0 (e.g. price down), market_return is negative
    actual_market_returns = pd.Series(np.where(y_true_binary == 1, assumed_positive_return, assumed_negative_return), index=X_data.index)

    # Calculate strategy returns
    strategy_returns = positions * actual_market_returns
    return in_sample_preds_binary, strategy_returns

# --- Retrieve existing models from Exercises 9.1 and 9.2 ---
# Model from Exercise 9.1 (GridSearchCV, neg_log_loss)
best_model_ex1_nll = grid_search.best_estimator_ # Assuming 'grid_search' is the fitted GridSearchCV object

# Model from Exercise 9.2 (RandomizedSearchCV, neg_log_loss)
best_model_ex2_nll = random_search.best_estimator_ # Assuming 'random_search' is the fitted RandomizedSearchCV object

# --- Exercise 9.3 ---
print("\n\n--- Exercise 9.3 ---")
# (a) Compute the Sharpe ratio of the resulting in-sample forecasts, from point 1.a
_, strategy_returns_ex1_nll = generate_predictions_and_returns(best_model_ex1_nll, X, y_target)
sharpe_ex1_nll = calculate_sharpe_ratio(strategy_returns_ex1_nll, periods_per_year=None) # Non-annualized for direct comparison
print(f"(a) Sharpe Ratio (GridSearchCV, neg_log_loss): {sharpe_ex1_nll:.4f}")

# (b) Repeat point 1.a, this time with accuracy as the scoring function.
print("\n(b) Repeating GridSearchCV with 'accuracy' scoring...")
grid_search_acc = GridSearchCV(
    estimator=pipeline, # Same pipeline definition as before
    param_grid=param_grid,
    scoring='accuracy', # Changed scoring
    cv=cv_strategy,
    n_jobs=-1,
    verbose=0 # Less verbose for this run
)
grid_search_acc.fit(X, y_target)
best_model_ex1_acc = grid_search_acc.best_estimator_
print(f"  Best params (accuracy scoring): {grid_search_acc.best_params_}")
print(f"  Best CV accuracy: {grid_search_acc.best_score_:.4f}")

_, strategy_returns_ex1_acc = generate_predictions_and_returns(best_model_ex1_acc, X, y_target)
sharpe_ex1_acc = calculate_sharpe_ratio(strategy_returns_ex1_acc, periods_per_year=None)
print(f"  Sharpe Ratio (GridSearchCV, accuracy): {sharpe_ex1_acc:.4f}")

# (c) What scoring method leads to higher (in-sample) Sharpe ratio?
print("\n(c) Comparison for Exercise 9.3:")
if sharpe_ex1_nll > sharpe_ex1_acc:
    print(f"  neg_log_loss scoring led to a higher Sharpe ratio ({sharpe_ex1_nll:.4f} vs {sharpe_ex1_acc:.4f}).")
elif sharpe_ex1_acc > sharpe_ex1_nll:
    print(f"  accuracy scoring led to a higher Sharpe ratio ({sharpe_ex1_acc:.4f} vs {sharpe_ex1_nll:.4f}).")
else:
    print(f"  Both scoring methods led to a similar Sharpe ratio ({sharpe_ex1_nll:.4f}).")

# --- Exercise 9.4 ---
print("\n\n--- Exercise 9.4 ---")
# (a) Compute the Sharpe ratio of the resulting in-sample forecasts, from point 2.a
_, strategy_returns_ex2_nll = generate_predictions_and_returns(best_model_ex2_nll, X, y_target)
sharpe_ex2_nll = calculate_sharpe_ratio(strategy_returns_ex2_nll, periods_per_year=None)
print(f"(a) Sharpe Ratio (RandomizedSearchCV, neg_log_loss): {sharpe_ex2_nll:.4f}")

# (b) Repeat point 2.a, this time with accuracy as the scoring function.
print("\n(b) Repeating RandomizedSearchCV with 'accuracy' scoring...")
random_search_acc = RandomizedSearchCV(
    estimator=pipeline_random, # Same pipeline definition as before
    param_distributions=param_distributions,
    n_iter=25,
    scoring='accuracy', # Changed scoring
    cv=cv_strategy_random,
    n_jobs=-1,
    verbose=0, # Less verbose for this run
    random_state=42
)
random_search_acc.fit(X, y_target)
best_model_ex2_acc = random_search_acc.best_estimator_
print(f"  Best params (accuracy scoring): {random_search_acc.best_params_}")
print(f"  Best CV accuracy: {random_search_acc.best_score_:.4f}")


_, strategy_returns_ex2_acc = generate_predictions_and_returns(best_model_ex2_acc, X, y_target)
sharpe_ex2_acc = calculate_sharpe_ratio(strategy_returns_ex2_acc, periods_per_year=None)
print(f"  Sharpe Ratio (RandomizedSearchCV, accuracy): {sharpe_ex2_acc:.4f}")

# (c) What scoring method leads to higher (in-sample) Sharpe ratio?
print("\n(c) Comparison for Exercise 9.4:")
if sharpe_ex2_nll > sharpe_ex2_acc:
    print(f"  neg_log_loss scoring led to a higher Sharpe ratio ({sharpe_ex2_nll:.4f} vs {sharpe_ex2_acc:.4f}).")
elif sharpe_ex2_acc > sharpe_ex2_nll:
    print(f"  accuracy scoring led to a higher Sharpe ratio ({sharpe_ex2_acc:.4f} vs {sharpe_ex2_nll:.4f}).")
else:
    print(f"  Both scoring methods led to a similar Sharpe ratio ({sharpe_ex2_nll:.4f}).")


# --- Exercise 9.5 ---
print("\n\n--- Exercise 9.5 ---")
print("(a) Why is the scoring function neg_log_loss defined as the negative log loss?")
print("   Scikit-learn's GridSearchCV and RandomizedSearchCV are designed to *maximize* a score.")
print("   Log loss (cross-entropy) is a loss function, meaning lower values are better.")
print("   To fit this into a maximization framework, the negative of the log loss is used.")
print("   Maximizing 'neg_log_loss' is equivalent to minimizing 'log_loss'.")

print("\n(b) What would be the outcome of maximizing the log loss, rather than the negative log loss?")
print("   If we were to (incorrectly) try to maximize log loss directly, the optimization")
print("   process would favor models that are very *confident and wrong*.")
print("   Log loss heavily penalizes confident incorrect predictions (e.g., predicting P(class_1)=0.99 when true is class_0).")
print("   Maximizing it would seek out these highly penalized, poorly calibrated models, which is the opposite of what we want.")

# --- Exercise 9.6 ---
print("\n\n--- Exercise 9.6 ---")
print("Consider an investment strategy that sizes its bets equally, regardless of the forecast’s confidence.")
print("In this case, what is a more appropriate scoring function for hyper-parameter tuning, accuracy or cross-entropy loss?")
print("   If bets are sized equally (e.g., +1 for predicted up, -1 for predicted down, 0 for flat),")
print("   the primary concern is getting the *direction* of the forecast correct, as the magnitude")
print("   of the bet does not change with the model's confidence.")
print("   **Accuracy** directly measures the proportion of correctly classified instances (correct direction).")
print("   Cross-entropy loss (log loss) penalizes based on the confidence of the prediction.")
print("   While a well-calibrated model (good log loss) is generally desirable, if the bet sizing")
print("   mechanism completely ignores the confidence, then optimizing directly for accuracy might be")
print("   more aligned with the strategy's utility function for this specific fixed-size betting scheme.")
print("   However, a model with very poor calibration (high confidence in wrong predictions) might still")
print("   perform poorly even if its raw accuracy is decent, as it might make many small errors but some")
print("   very confident large errors that lead to overall poor performance if the underlying asset returns vary significantly.")
print("   In a strict fixed-size betting scenario focused *only* on direction, accuracy seems more direct.")
print("   But generally, a model with good log-loss is more robust as it implies better probability estimates.")
print("   If the choice is strictly between these two for this scenario: Accuracy.")