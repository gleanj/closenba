# NBA "Both Teams Lead by 5+" Prediction Model

## 🎯 Project Overview

This project builds a machine learning model to predict whether **both teams will lead by 5 or more points at any point during an NBA game**. This is a unique prediction target that focuses on game volatility, momentum swings, and competitive balance rather than traditional win/loss outcomes.

## 🔑 Why This Target Matters

- **Unique Market Inefficiency**: Traditional models focus on game winners and spreads. This target exploits a different dimension
- **Live Betting Value**: Games with lead changes create more live betting opportunities  
- **Momentum Analysis**: Requires modeling game flow and momentum, not just final outcomes
- **Competitive Balance Indicator**: Measures true competitiveness beyond final score

## 📁 Project Structure

```
closenba/
├── data/
│   ├── raw/              # Raw data from nba_api
│   ├── processed/        # Cleaned and feature-engineered data
│   └── labels/           # Historical game labels (both teams led by 5+)
├── notebooks/
│   └── 01_data_collection.ipynb  # Data collection workflow
├── src/
│   ├── data/
│   │   ├── collectors.py     # nba_api data collection
│   │   ├── labelers.py       # Label games (both teams led by 5+)
│   │   ├── validation.py     # ✨ Data validation and preprocessing
│   │   └── processors.py     # Data cleaning and preprocessing
│   ├── features/
│   │   ├── game_features.py  # Game-level features
│   │   ├── team_features.py  # ✨ Enhanced: Team stats, Four Factors, pace, volatility
│   │   └── momentum_features.py  # Momentum and volatility features
│   ├── models/
│   │   ├── baseline.py       # ✨ Enhanced: Logistic Regression with CV
│   │   ├── ensemble.py       # ✨ NEW: Random Forest, XGBoost with tuning
│   │   ├── pipeline.py       # ✨ NEW: End-to-end training pipeline
│   │   └── deep_learning.py  # Neural networks if needed later
│   └── utils/
│       ├── config.py         # Configuration and constants
│       ├── rate_limiter.py   # API rate limiting
│       └── metrics.py        # Custom evaluation metrics
├── models/               # Saved model files
├── outputs/             # Predictions and reports
├── requirements.txt
└── config.yaml
```

## ✨ Recent Accuracy Improvements

**Major enhancements implemented to maximize prediction accuracy:**

### 1. **Critical Bug Fixes**
- ✅ Fixed deprecated pandas `fillna(method='ffill')` in labelers.py
- ✅ Improved score parsing accuracy for play-by-play data
- ✅ Enhanced error handling throughout codebase

### 2. **Advanced Feature Engineering**
- ✅ **Pace Calculations**: Possessions per 48 minutes for tempo normalization
- ✅ **Offensive/Defensive Ratings**: Points per 100 possessions
- ✅ **Enhanced Volatility Features**:
  - Coefficient of variation (normalized volatility)
  - Close game percentage (games within 5 points)
  - Blowout percentage (games won/lost by 15+)
  - Scoring trends and streaks
- ✅ **Competitive Balance Indicators**:
  - Evenly matched teams detection
  - Win percentage differentials
  - Combined volatility metrics
- ✅ **Momentum Features**:
  - Current win/loss streaks
  - Points trend (recent vs season average)
  - Offensive vs defensive matchups

### 3. **Model Improvements**
- ✅ **Cross-Validation**: Stratified K-Fold CV for robust evaluation
- ✅ **Hyperparameter Tuning**: RandomizedSearchCV for optimal parameters
- ✅ **Ensemble Models**: Random Forest and XGBoost with auto-tuning
- ✅ **Class Imbalance Handling**: Proper scaling for imbalanced datasets
- ✅ **Feature Importance Analysis**: Identify most predictive features

### 4. **Data Quality**
- ✅ **Data Validation**: Automated checks for missing values, outliers, duplicates
- ✅ **Preprocessing Pipeline**: Robust handling of missing data and outliers
- ✅ **Feature Selection**: Remove noise and improve generalization
- ✅ **Time-Based Splitting**: Chronological split prevents look-ahead bias

### 5. **Comprehensive Training Pipeline**
- ✅ **End-to-End Automation**: From data prep to model selection
- ✅ **Multiple Model Comparison**: Automatically compare all models
- ✅ **Best Model Selection**: Choose optimal model based on validation metrics
- ✅ **Results Tracking**: Save all metrics and models for reproducibility

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Project

Review and customize `config.yaml`:
- Seasons to collect
- Rate limiting settings (CRITICAL to avoid IP bans)
- Feature engineering parameters
- Model hyperparameters

### 3. Data Collection

⚠️ **IMPORTANT**: Data collection takes 2-3 hours per season due to mandatory rate limiting.

```python
from src.data.collectors import NBADataCollector
from src.utils.config import get_config

config = get_config()
collector = NBADataCollector(config)

# Collect data for a season (takes 2-3 hours)
saved_files = collector.collect_season_data('2023-24', 'data/raw/2023-24')
```

Or use the notebook: `notebooks/01_data_collection.ipynb`

### 4. Label Games

```python
from src.data.labelers import GameLabeler

labeler = GameLabeler(threshold=5)
labels = labeler.label_game_from_pbp(play_by_play_df)
# Returns: {'both_teams_led_5plus': True/False, ...}
```

### 5. Feature Engineering

```python
from src.features.team_features import FourFactorsCalculator, TeamFeatureEngineer

# Calculate Dean Oliver's Four Factors
factors = FourFactorsCalculator.calculate_all_factors(team_stats)

# Create rolling features
engineer = TeamFeatureEngineer(rolling_windows=[10, 20])
rolling_features = engineer.create_rolling_features(team_game_logs)
```

### 6. Train Models (New Enhanced Pipeline)

**Option A: Use the automated pipeline (Recommended)**

```python
from src.models.pipeline import ModelPipeline
from pathlib import Path

# Create pipeline
pipeline = ModelPipeline(output_dir=Path('outputs/models'))

# Run complete training pipeline
# This will:
# - Validate data quality
# - Preprocess and split data (time-based)
# - Train baseline models (Logistic Regression, Naive Bayes)
# - Train ensemble models (Random Forest, XGBoost)
# - Perform hyperparameter tuning
# - Select best model
# - Evaluate on test set
# - Save results

results = pipeline.run_full_pipeline(
    df=features_df,  # Your feature DataFrame
    target_col='both_teams_led_5plus',
    date_col='game_date',
    train_baseline=True,
    train_ensemble=True,
    tune_hyperparameters=True
)

print(f"Best model: {pipeline.best_model_name}")
print(f"Test accuracy: {results['best_model']['test_metrics']['accuracy']:.3f}")
print(f"Test ROC-AUC: {results['best_model']['test_metrics']['roc_auc']:.3f}")
```

**Option B: Train individual models**

```python
from src.models.baseline import BaselineModel
from src.models.ensemble import EnsembleModel

# Baseline: Logistic Regression with Cross-Validation
lr_model = BaselineModel(model_type='logistic', scale_features=True)
lr_model.fit(X_train, y_train)
cv_results = lr_model.cross_validate(X_train, y_train, cv=5)
metrics = lr_model.evaluate(X_test, y_test)
print(f"Logistic Regression - Accuracy: {metrics['accuracy']:.3f}")

# Ensemble: XGBoost with Hyperparameter Tuning
xgb_model = EnsembleModel(model_type='xgboost', tune_hyperparameters=True)
xgb_model.fit(X_train, y_train, X_val, y_val)
metrics = xgb_model.evaluate(X_test, y_test)
print(f"XGBoost - Accuracy: {metrics['accuracy']:.3f}")

# Get feature importance
importance = xgb_model.get_feature_importance(top_n=15)
print("\nTop 15 Most Important Features:")
print(importance)
```

## 📊 Key Features

### Core Statistical Features
- **Dean Oliver's Four Factors** (eFG%, TOV%, ORB%, FTR)
- Pace-adjusted offensive/defensive ratings
- Recent form (last 5/10/20 games)

### Volatility & Momentum Features (Unique to This Target)
- Standard deviation of point differential
- Lead change frequency
- Average largest lead in recent games
- Comeback frequency metrics

### Contextual Features
- Rest differential (back-to-back vs rested)
- Home/away splits
- Head-to-head history
- Injury impact scores

## ⚠️ Critical Implementation Notes

### 1. **MANDATORY Rate Limiting**

The stats.nba.com API **aggressively blocks IPs** that make too many requests. Our system implements:

- **700ms delay between ALL requests** (1.4 requests/second)
- Exponential backoff on errors
- Browser-like headers to avoid bot detection
- Proper timeout handling

**DO NOT** disable or reduce rate limiting. This will result in:
- HTTP 429 errors (Too Many Requests)
- ReadTimeout errors
- Temporary or permanent IP bans

### 2. **Cloud Deployment Requires Proxies**

From research: stats.nba.com **blocks cloud platform IPs** (AWS, Heroku, GitHub Actions).

If deploying to cloud:
```python
collector = NBADataCollector(config)
# Use proxy parameter (acquire your own proxy service)
```

### 3. **Data Collection Time**

- **~2-3 hours per season** (1,230+ games)
- Run overnight or in multiple sessions
- Use checkpointing to resume if interrupted

## 🎯 Model Development Strategy

Based on comprehensive research literature:

### Phase 1: Baseline Models
1. **Logistic Regression** (most interpretable, computationally efficient)
2. **Gaussian Naive Bayes**

From research: "Simple models often outperform complex ones" due to high noise in sports data.

### Phase 2: Ensemble Models
3. **Random Forest** (robust to overfitting)
4. **XGBoost** (highest accuracy in research)
5. **XGBoost + SHAP** (interpretability)

### Phase 3: Advanced (If Justified)
- Recurrent Neural Networks (momentum modeling)
- Graph Neural Networks (player interactions)
- Reinforcement Learning (betting policy)

**Key Principle**: Start simple. Only add complexity if justified by performance gains.

## 📈 Success Metrics

- **Primary**: Accuracy in predicting "both teams lead by 5+" outcome
- **Secondary**: Precision and Recall for positive class
- **Tertiary**: ROC-AUC, calibration plots
- **Real-world**: Backtested ROI if applied to betting markets

## 🔬 Research Foundation

This project is built on comprehensive analysis of:

1. **nba_api Python Library**: Complete technical analysis of architecture, operation, and risk mitigation
2. **Machine Learning in NBA Betting**: Analysis of 48 academic papers and real-money projects covering:
   - Supervised learning (LR, NB, RF, XGBoost, Neural Networks)
   - Feature engineering (Four Factors, momentum, volatility)
   - Reinforcement learning for betting policy
   - Market efficiency and limitations

Key findings applied:
- Feature engineering > algorithm choice
- Four Factors are foundational
- Contextual features (rest, fatigue) are critical
- Simple models often outperform complex ones
- Rate limiting is non-negotiable

## 🔄 Real-Time Predictions (NEW!)

**CRITICAL**: Your rolling features (L5, L10, L20 games) require **daily updates** for accurate predictions.

### Why Today's Data Matters

If you're predicting tomorrow's LAL vs GSW game:
- **Without today's update**: Features use data from 2+ days ago (stale)
- **With today's update**: Features include tonight's results (fresh)

**Impact on Accuracy:**
- Stale features can reduce accuracy by 10-15%
- Recent momentum/form changes are missed
- Injuries from today's game aren't reflected

### Daily Update Workflow

**Step 1: Update Dataset with Latest Results**

Run daily (e.g., via cron at 2 AM after games finish):

```bash
# Update with yesterday's results
python scripts/daily_update.py

# Or include today's results (for evening updates)
python scripts/daily_update.py --include-today
```

This fetches:
- Yesterday's game results
- Play-by-play data for labeling
- Boxscores for features
- Updates rolling averages, streaks, etc.

**Step 2: Generate Tomorrow's Predictions**

```bash
python scripts/predict_tomorrow.py --model models/best_model_xgboost.joblib
```

This:
1. Updates data with latest results (optional: use `--no-update` to skip)
2. Recalculates features with fresh data
3. Fetches tomorrow's schedule
4. Generates predictions
5. Saves to `outputs/predictions/tomorrow.csv`

**Step 3: Automate with Cron** (Linux/Mac)

```bash
# Edit crontab
crontab -e

# Add this line to run at 2 AM daily
0 2 * * * cd /path/to/closenba && /path/to/venv/bin/python scripts/daily_update.py
```

Or use Windows Task Scheduler for automation.

### Manual Usage

```python
from src.data.daily_updater import DailyUpdater, TomorrowPredictor
from pathlib import Path

# Update data
updater = DailyUpdater(data_dir=Path('data/raw'))
summary = updater.run_daily_update(include_today=True)
print(f"Updated {summary['games_updated']} games")

# Generate predictions
predictor = TomorrowPredictor(
    model_path=Path('models/best_model_xgboost.joblib'),
    data_dir=Path('data/raw')
)
predictions = predictor.predict_tomorrow(update_data=True)
```

### Data Freshness Best Practices

1. **Train on historical data** (one-time): Use `pipeline.run_full_pipeline()`
2. **Update daily**: Run `daily_update.py` every morning
3. **Predict daily**: Run `predict_tomorrow.py` for next day's games
4. **Retrain monthly**: Update model with new month's data

## 📝 Development Roadmap

- [x] Project structure and configuration
- [x] Data collection module with rate limiting
- [x] Game labeling system (both teams led by 5+)
- [x] Feature engineering (Four Factors, volatility)
- [x] Baseline models (Logistic Regression, Naive Bayes)
- [x] ✨ Advanced ensemble models (Random Forest, XGBoost)
- [x] ✨ Cross-validation and hyperparameter tuning
- [x] ✨ Data validation and preprocessing pipeline
- [x] ✨ Enhanced feature engineering (pace, ratings, advanced volatility)
- [x] ✨ Feature importance analysis
- [x] ✨ Comprehensive training pipeline
- [x] ✨ Daily data updates and real-time predictions
- [ ] Model interpretability (SHAP)
- [ ] Backtesting framework
- [ ] Full feature engineering integration for daily predictions
- [ ] Deployment with monitoring

## 🤝 Contributing

This is a research project. Feel free to:
- Experiment with new features
- Try different models
- Improve data collection efficiency
- Add new analytical notebooks

## ⚖️ License & Disclaimer

This is an educational and research project.

**Disclaimer**: This project is for educational purposes only. Sports betting involves risk. Past performance does not guarantee future results. Use responsibly and in compliance with local laws.

## 📚 Documentation

Detailed documentation is available in the `/docs` folder:
- API usage and rate limiting best practices
- Feature engineering methodology
- Model selection criteria
- Error handling and troubleshooting

## 🙏 Acknowledgments

Built using:
- [nba_api](https://github.com/swar/nba_api) - Community-maintained NBA stats API client
- Research insights from 48+ academic papers and real-money projects
- Dean Oliver's Four Factors framework

---

**Last Updated**: 2025-01-09

For questions or issues, please open a GitHub issue or contact the maintainers.
