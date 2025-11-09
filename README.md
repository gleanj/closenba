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
│   │   └── processors.py     # Data cleaning and preprocessing
│   ├── features/
│   │   ├── game_features.py  # Game-level features
│   │   ├── team_features.py  # Team statistics and metrics
│   │   └── momentum_features.py  # Momentum and volatility features
│   ├── models/
│   │   ├── baseline.py       # Logistic Regression baseline
│   │   ├── ensemble.py       # Random Forest, XGBoost
│   │   └── deep_learning.py  # Neural networks if needed
│   └── utils/
│       ├── config.py         # Configuration and constants
│       ├── rate_limiter.py   # API rate limiting
│       └── metrics.py        # Custom evaluation metrics
├── models/               # Saved model files
├── outputs/             # Predictions and reports
├── requirements.txt
└── config.yaml
```

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

### 6. Train Models

```python
from src.models.baseline import BaselineModel

# Train Logistic Regression
model = BaselineModel(model_type='logistic')
model.fit(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.3f}")
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

## 📝 Development Roadmap

- [x] Project structure and configuration
- [x] Data collection module with rate limiting
- [x] Game labeling system (both teams led by 5+)
- [x] Feature engineering (Four Factors, volatility)
- [x] Baseline models (Logistic Regression, Naive Bayes)
- [ ] Advanced ensemble models (Random Forest, XGBoost)
- [ ] Model interpretability (SHAP)
- [ ] Backtesting framework
- [ ] Real-time prediction pipeline
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
