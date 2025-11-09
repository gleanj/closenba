"""
Test script to verify the environment is set up correctly
"""
import sys
from pathlib import Path

print("Testing CloseNBA Setup...")
print(f"Python version: {sys.version}")
print()

# Test imports
try:
    import pandas as pd
    print(f"[OK] pandas: {pd.__version__}")
except ImportError as e:
    print(f"[FAIL] pandas: {e}")

try:
    import numpy as np
    print(f"[OK] numpy: {np.__version__}")
except ImportError as e:
    print(f"[FAIL] numpy: {e}")

try:
    from nba_api.stats.static import teams, players
    print(f"[OK] nba_api: imported successfully")
except ImportError as e:
    print(f"[FAIL] nba_api: {e}")

try:
    import yaml
    print(f"[OK] pyyaml: imported successfully")
except ImportError as e:
    print(f"[FAIL] pyyaml: {e}")

try:
    from sklearn import __version__
    print(f"[OK] scikit-learn: {__version__}")
except ImportError as e:
    print(f"[FAIL] scikit-learn: {e}")

try:
    import xgboost
    print(f"[OK] xgboost: {xgboost.__version__}")
except ImportError as e:
    print(f"[FAIL] xgboost: {e}")

print()

# Test configuration loading
try:
    from src.utils.config import get_config
    config = get_config()
    print(f"[OK] Configuration loaded successfully")
    print(f"   Seasons: {config.data_collection['seasons']}")
except Exception as e:
    print(f"[FAIL] Configuration: {e}")

print()

# Test NBA API connection with a simple query
try:
    from nba_api.stats.static import teams
    nba_teams = teams.get_teams()
    print(f"[OK] NBA API connection successful")
    print(f"   Retrieved {len(nba_teams)} teams")
    print(f"   Sample team: {nba_teams[0]['full_name']}")
except Exception as e:
    print(f"[FAIL] NBA API: {e}")

print()
print("Setup test complete!")
