"""
Feature Engineering for NBA Prediction Model

Implements Dean Oliver's Four Factors and volatility/momentum features.
Based on research showing these are most predictive for NBA outcomes.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FourFactorsCalculator:
    """
    Calculate Dean Oliver's Four Factors of Basketball Success.
    
    From research: These are the most correlated stats with winning:
    1. eFG% (40% weight) - Effective Field Goal %
    2. TOV% (25% weight) - Turnover %
    3. ORB% (20% weight) - Offensive Rebound %
    4. FTR (15% weight) - Free Throw Rate
    """
    
    @staticmethod
    def calculate_efg_pct(fgm: float, fga: float, fg3m: float) -> float:
        """
        Calculate Effective Field Goal Percentage.
        
        Formula: eFG% = (FGM + 0.5 * 3PM) / FGA
        
        Accounts for the extra value of 3-pointers.
        """
        if fga == 0:
            return 0.0
        return (fgm + 0.5 * fg3m) / fga
    
    @staticmethod
    def calculate_tov_pct(tov: float, fga: float, fta: float) -> float:
        """
        Calculate Turnover Percentage.
        
        Formula: TOV% = TOV / (FGA + 0.44 * FTA + TOV)
        
        Measures team's ability to protect the ball.
        """
        possessions = fga + 0.44 * fta + tov
        if possessions == 0:
            return 0.0
        return tov / possessions
    
    @staticmethod
    def calculate_orb_pct(orb: float, drb_opp: float) -> float:
        """
        Calculate Offensive Rebound Percentage.
        
        Formula: ORB% = ORB / (ORB + Opponent's DRB)
        
        Measures team's ability to get second-chance opportunities.
        """
        total_rebounds = orb + drb_opp
        if total_rebounds == 0:
            return 0.0
        return orb / total_rebounds
    
    @staticmethod
    def calculate_ftr(ftm: float, fga: float) -> float:
        """
        Calculate Free Throw Rate.
        
        Formula: FTR = FTM / FGA
        
        Measures team's ability to get to the foul line.
        """
        if fga == 0:
            return 0.0
        return ftm / fga
    
    @classmethod
    def calculate_all_factors(cls, team_stats: pd.Series, opp_stats: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate all four factors for a team.
        
        Args:
            team_stats: Series with team box score stats
            opp_stats: Series with opponent stats (needed for ORB%)
        
        Returns:
            Dictionary with all four factors
        """
        factors = {
            'eFG%': cls.calculate_efg_pct(
                team_stats.get('FGM', 0),
                team_stats.get('FGA', 0),
                team_stats.get('FG3M', 0)
            ),
            'TOV%': cls.calculate_tov_pct(
                team_stats.get('TOV', 0),
                team_stats.get('FGA', 0),
                team_stats.get('FTA', 0)
            ),
            'FTR': cls.calculate_ftr(
                team_stats.get('FTM', 0),
                team_stats.get('FGA', 0)
            )
        }
        
        # ORB% requires opponent data
        if opp_stats is not None:
            factors['ORB%'] = cls.calculate_orb_pct(
                team_stats.get('OREB', 0),
                opp_stats.get('DREB', 0)
            )
        else:
            factors['ORB%'] = 0.0
        
        return factors


class TeamFeatureEngineer:
    """
    Creates team-level features for prediction.
    
    Features include:
    - Four Factors (offense and defense)
    - Rolling averages (recent form)
    - Volatility metrics
    - Contextual features (rest, home/away)
    """
    
    def __init__(self, rolling_windows: List[int] = [10, 20]):
        self.rolling_windows = rolling_windows
        self.four_factors_calc = FourFactorsCalculator()
    
    def create_rolling_features(self, team_game_logs: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling average features for recent team form.
        
        Args:
            team_game_logs: DataFrame with team's game-by-game stats
            windows: List of window sizes (e.g., [10, 20] for last 10/20 games)
        
        Returns:
            DataFrame with rolling average features
        """
        if windows is None:
            windows = self.rolling_windows
        
        # Ensure chronological order
        team_game_logs = team_game_logs.sort_values('GAME_DATE')
        
        rolling_features = pd.DataFrame(index=team_game_logs.index)
        
        # Stats to calculate rolling averages for
        stats_to_roll = ['PTS', 'FGA', 'FGM', 'FG3M', 'FTA', 'FTM', 
                         'OREB', 'DREB', 'AST', 'TOV', 'STL', 'BLK', 'PF']
        
        for window in windows:
            for stat in stats_to_roll:
                if stat in team_game_logs.columns:
                    col_name = f'{stat}_L{window}'
                    rolling_features[col_name] = team_game_logs[stat].rolling(
                        window=window, min_periods=1
                    ).mean()
        
        return rolling_features
    
    def calculate_four_factors_rolling(self, team_game_logs: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        Calculate Four Factors with rolling averages.
        
        Args:
            team_game_logs: Team's game logs
            window: Rolling window size
        
        Returns:
            DataFrame with Four Factors rolling averages
        """
        # Calculate Four Factors for each game
        factors_list = []
        
        for idx, row in team_game_logs.iterrows():
            factors = self.four_factors_calc.calculate_all_factors(row)
            factors_list.append(factors)
        
        factors_df = pd.DataFrame(factors_list, index=team_game_logs.index)
        
        # Calculate rolling averages
        rolling_factors = pd.DataFrame(index=factors_df.index)
        
        for factor in ['eFG%', 'TOV%', 'ORB%', 'FTR']:
            col_name = f'{factor}_L{window}'
            rolling_factors[col_name] = factors_df[factor].rolling(
                window=window, min_periods=1
            ).mean()
        
        return rolling_factors
    
    def calculate_volatility_features(self, team_game_logs: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        Calculate volatility and momentum features.
        
        These are CRITICAL for our target: predicting competitive, volatile games.
        
        Features:
        - std_point_diff: Standard deviation of point differential
        - avg_largest_lead: Average of team's largest lead per game
        - comeback_frequency: How often team came back from deficits
        
        Args:
            team_game_logs: Team's game logs with WIN/LOSS and point differential
            window: Rolling window size
        
        Returns:
            DataFrame with volatility features
        """
        volatility_features = pd.DataFrame(index=team_game_logs.index)
        
        # Calculate point differential for each game
        if 'PLUS_MINUS' in team_game_logs.columns:
            point_diff = team_game_logs['PLUS_MINUS']
        else:
            # Calculate from PTS and opponent PTS if available
            point_diff = pd.Series(0, index=team_game_logs.index)
        
        # Standard deviation of point differential (volatility)
        volatility_features['std_point_diff'] = point_diff.rolling(
            window=window, min_periods=1
        ).std()
        
        # Win streak features
        if 'WL' in team_game_logs.columns:
            wins = (team_game_logs['WL'] == 'W').astype(int)
            volatility_features['win_pct_L10'] = wins.rolling(
                window=window, min_periods=1
            ).mean()
        
        return volatility_features
    
    def create_contextual_features(self, game_info: pd.Series) -> Dict[str, float]:
        """
        Create contextual features for a specific game matchup.
        
        Features:
        - rest_days: Days of rest since last game
        - back_to_back: Binary indicator for back-to-back games
        - home_away: Binary indicator (1=home, 0=away)
        
        Args:
            game_info: Series with game information
        
        Returns:
            Dictionary of contextual features
        """
        features = {}
        
        # Home/Away
        features['is_home'] = 1 if game_info.get('MATCHUP', '').find('@') == -1 else 0
        
        # Rest days (if previous game date available)
        # This would require additional data - placeholder for now
        features['rest_days'] = 0
        features['back_to_back'] = 0
        
        return features


class MatchupFeatureEngineer:
    """
    Creates matchup-level features comparing two teams.
    
    This is where we combine features from both teams to predict
    the likelihood of a competitive, volatile game.
    """
    
    def __init__(self):
        self.team_engineer = TeamFeatureEngineer()
    
    def create_matchup_features(
        self,
        home_team_features: pd.Series,
        away_team_features: pd.Series
    ) -> pd.Series:
        """
        Create matchup features comparing home and away teams.
        
        Features include:
        - Differentials (home - away) for all stats
        - Offensive/Defensive matchups (home offense vs away defense)
        - Competitive balance indicators
        
        Args:
            home_team_features: Home team's recent features
            away_team_features: Away team's recent features
        
        Returns:
            Series with matchup features
        """
        matchup_features = {}
        
        # Differentials for Four Factors
        for factor in ['eFG%_L10', 'TOV%_L10', 'ORB%_L10', 'FTR_L10']:
            if factor in home_team_features and factor in away_team_features:
                matchup_features[f'{factor}_diff'] = (
                    home_team_features[factor] - away_team_features[factor]
                )
        
        # Competitive balance score
        # Lower = more evenly matched = more likely both teams lead
        if 'PTS_L10' in home_team_features and 'PTS_L10' in away_team_features:
            matchup_features['competitive_balance'] = abs(
                home_team_features['PTS_L10'] - away_team_features['PTS_L10']
            )
        
        # Combined volatility (sum of both teams' volatility)
        if 'std_point_diff' in home_team_features and 'std_point_diff' in away_team_features:
            matchup_features['combined_volatility'] = (
                home_team_features['std_point_diff'] + away_team_features['std_point_diff']
            )
        
        return pd.Series(matchup_features)
