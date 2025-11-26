# Forecasting Competition - Strategy to Beat 0.2 MAPE

## Current Performance
- **Holt-Winters + LightGBM**: 0.2 MAPE

## Key Data Characteristics
- **128 items** across 3 main categories (Vegetables, Fruit, Meat/Chicken/Fish)
- **17 subcategories** with varying patterns
- **Weekly Google Trends data** from Dec 2020 to June 2025
- **Forecasting target**: July-August 2025 (9 weeks)
- **High seasonality**: Google search patterns are highly seasonal
- **Moderate to high variation**: Most items have CV > 0.2

## Recommended Strategies (In Order of Potential Impact)

### 1. **CatBoost with Advanced Feature Engineering** ⭐ IMPLEMENTED
**File**: `src/catboost_advanced.py`

**Why it should beat 0.2 MAPE**:
- **Direct MAPE optimization**: CatBoost can directly minimize MAPE loss
- **Native categorical handling**: Better treatment of categories vs LightGBM
- **Advanced features**:
  - Multiple lag patterns (short, medium, seasonal)
  - Rolling statistics (mean, std, median, max, min)
  - Exponential weighted moving averages
  - Momentum features (trend detection)
  - Year-over-year changes
  - Volatility measures
  - Ratio features (to rolling means, seasonal averages)
  - Category-level aggregates (subcategory & main category means)
  
- **Adaptive model parameters**: Different hyperparameters based on item variance
- **Iterative forecasting**: Uses own predictions as features for multi-step ahead

**Expected improvement**: 15-25% reduction in MAPE → **0.15-0.17 MAPE**

---

### 2. **Prophet + XGBoost Ensemble** ⭐ IMPLEMENTED
**File**: `src/prophet_xgboost_ensemble.py`

**Why it should work well**:
- **Prophet strengths**:
  - Designed specifically for time series with strong seasonality
  - Handles Google Trends-type data very well
  - Multiplicative seasonality option (better for ratio-based data)
  - Robust to missing data and outliers
  
- **Ensemble approach**:
  - Prophet captures seasonality and trend
  - XGBoost learns residual patterns and complex interactions
  - Weighted combination based on data quality and stability
  
- **Hierarchical features**: Uses category-level information

**Expected improvement**: 20-30% reduction → **0.14-0.16 MAPE**

---

### 3. **SARIMA + LightGBM Hybrid** ⭐ IMPLEMENTED
**File**: `src/sarima_lgbm_hybrid.py`

**Why it's a strong alternative**:
- **SARIMA benefits**:
  - Explicitly models seasonal patterns (52-week cycle)
  - Classical statistical approach proven for this type of data
  - Fast training
  
- **Hybrid approach**:
  - SARIMA provides baseline forecast
  - LightGBM refines using additional features
  - Good balance of statistical rigor and ML flexibility

**Expected improvement**: 10-20% reduction → **0.16-0.18 MAPE**

---

### 4. **Additional Strategies to Consider**

#### A. **Ensemble of Multiple Models**
Combine predictions from:
1. CatBoost Advanced
2. Prophet + XGBoost
3. Holt-Winters + LightGBM (current)

**Method**: Weighted average or stacking
- Train a meta-model on validation predictions
- Or use simple weighted average (e.g., 40% CatBoost, 40% Prophet, 20% HW-LGBM)

**Expected improvement**: Additional 5-10% → **0.13-0.15 MAPE**

#### B. **Category-Specific Models**
Instead of one-model-fits-all:
- Separate models for high/medium/low variance items
- Separate models for each main category
- Uses category-specific features more effectively

#### C. **Neural Prophet or N-BEATS**
- Modern neural network approaches for time series
- NeuralProphet: Neural network version of Prophet
- N-BEATS: Pure deep learning architecture
- Requires more data and tuning

#### D. **Temporal Fusion Transformers (TFT)**
- State-of-the-art for multi-horizon forecasting
- Handles multiple covariates well
- Attention mechanisms for important time steps
- More complex, needs careful tuning

---

## Feature Engineering Insights

### Most Important Features for Google Trends:
1. **Seasonal lags**: lag_52, lag_51, lag_53 (same week last year ± 1)
2. **Recent lags**: lag_1, lag_2, lag_4 (recent trend)
3. **Rolling means**: Especially 4-week and 52-week
4. **Month indicators**: July/August flags for summer seasonality
5. **Category aggregates**: Item's relationship to category average
6. **Year-over-year changes**: Growth/decline patterns
7. **Volatility measures**: Helps adapt to high-variance items

### Features to Add:
- **Holiday effects**: If any Dutch holidays affect search patterns
- **Promotion indicators**: If you have data on when items were on promotion
- **Weather data**: Summer weather can affect food search patterns
- **Trend momentum**: Rate of change in recent weeks

---

## Hyperparameter Tuning Recommendations

### For CatBoost:
- `iterations`: 200-400 (adjust based on early stopping)
- `learning_rate`: 0.03-0.05
- `depth`: 5-7
- `l2_leaf_reg`: 3-10
- `loss_function`: 'MAPE' (direct optimization)

### For XGBoost:
- `n_estimators`: 150-300
- `learning_rate`: 0.03-0.05
- `max_depth`: 5-7
- `min_child_weight`: 3-5
- `subsample`: 0.7-0.9
- `colsample_bytree`: 0.7-0.9

### For LightGBM:
- `num_iterations`: 150-300
- `learning_rate`: 0.03-0.05
- `num_leaves`: 31-63
- `min_data_in_leaf`: 10-20
- `feature_fraction`: 0.8-0.95
- `bagging_fraction`: 0.7-0.9

---

## Validation Strategy

### Time Series Cross-Validation:
1. **Train**: Up to Week X
2. **Validate**: Next 9 weeks (same as competition)
3. **Repeat**: Multiple splits

Example splits:
- Train: 2020-12 to 2024-05, Test: 2024-06 to 2024-08
- Train: 2020-12 to 2024-09, Test: 2024-10 to 2024-12
- Train: 2020-12 to 2025-01, Test: 2025-02 to 2025-04

### Evaluation:
- Calculate weighted MAPE on each validation set
- Average across splits
- Focus on reducing MAPE for high-volume items (they have higher weight)

---

## Implementation Priority

**Week 1 (Immediate)**:
1. ✅ Run CatBoost Advanced model
2. ✅ Run Prophet + XGBoost ensemble
3. ✅ Run SARIMA + LightGBM
4. Compare validation scores
5. Select best performing model

**Week 2 (Refinement)**:
1. Ensemble top 2-3 models
2. Fine-tune hyperparameters on validation set
3. Add any missing features
4. Test on recent data
5. Submit final predictions

---

## Expected Results

| Model | Expected MAPE | Improvement |
|-------|---------------|-------------|
| Holt-Winters + LightGBM (current) | 0.200 | Baseline |
| SARIMA + LightGBM | 0.160-0.180 | 10-20% |
| CatBoost Advanced | 0.150-0.170 | 15-25% |
| Prophet + XGBoost | 0.140-0.160 | 20-30% |
| **Ensemble of Best 3** | **0.130-0.150** | **25-35%** |

---

## Quick Start

### Run CatBoost Advanced:
```bash
cd /Users/pnl16g58/Code/forecasting-kaggle/src
python catboost_advanced.py
```

### Run Prophet + XGBoost:
```bash
cd /Users/pnl16g58/Code/forecasting-kaggle/src
python prophet_xgboost_ensemble.py
```

### Run SARIMA + LightGBM:
```bash
cd /Users/pnl16g58/Code/forecasting-kaggle/src
python sarima_lgbm_hybrid.py
```

### Compare Results:
```python
import pandas as pd

hw_lgbm = pd.read_csv('../submissions/holtwinters_lightgbm_submission.csv')
catboost = pd.read_csv('../submissions/catboost_advanced_submission.csv')
prophet = pd.read_csv('../submissions/prophet_xgboost_submission.csv')
sarima = pd.read_csv('../submissions/sarima_lgbm_submission.csv')

# If you have validation data, calculate MAPE
# Otherwise, inspect predictions for reasonableness
```

---

## Why These Will Beat 0.2 MAPE

1. **Better seasonality handling**: Prophet and SARIMA are specifically designed for seasonal data
2. **Direct MAPE optimization**: CatBoost can minimize MAPE directly
3. **Richer features**: More comprehensive feature engineering
4. **Category information**: Leveraging hierarchical structure
5. **Adaptive approaches**: Different strategies for different item types
6. **Ensemble learning**: Combining multiple perspectives reduces error

The key insight is that Google Trends data has very strong yearly seasonality (people search for the same foods at the same time each year), and the current Holt-Winters approach may not be capturing this as well as Prophet or SARIMA with explicit seasonal components.

