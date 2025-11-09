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
    def calculate_possessions(fga: float, fta: float, tov: float, oreb: float, dreb_opp: float) -> float:
        """
        Calculate team possessions (Dean Oliver's formula).

        Formula: Poss = FGA - OREB + TOV + 0.44 * FTA
        More accurate: Poss = 0.5 * ((Tm FGA + 0.4 * Tm FTA - 1.07 * (Tm OREB / (Tm OREB + Opp DREB)) *
                                     (Tm FGA - Tm FGM) + Tm TOV) + (Opp FGA + 0.4 * Opp FTA - 1.07 * ...))

        Simplified version for single team:
        """
        return fga - oreb + tov + 0.44 * fta

    @staticmethod
    def calculate_pace(poss: float, minutes: float = 48.0) -> float:
        """
        Calculate pace (possessions per 48 minutes).

        Higher pace = faster game = more possessions = more scoring opportunities
        Critical for normalizing stats across different game tempos.
        """
        if minutes == 0:
            return 0.0
        return (poss / minutes) * 48.0

    @staticmethod
    def calculate_offensive_rating(pts: float, poss: float) -> float:
        """
        Calculate offensive rating (points per 100 possessions).

        Normalizes scoring by pace - allows comparison across different tempo games.
        """
        if poss == 0:
            return 0.0
        return (pts / poss) * 100.0

    @staticmethod
    def calculate_defensive_rating(pts_allowed: float, poss: float) -> float:
        """
        Calculate defensive rating (points allowed per 100 possessions).

        Lower is better.
        """
        if poss == 0:
            return 0.0
        return (pts_allowed / poss) * 100.0
    
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
        - cv_point_diff: Coefficient of variation (normalized volatility)
        - close_game_pct: Percentage of games decided by <= 5 points
        - comeback_pct: Percentage of wins when trailing
        - blowout_pct: Percentage of games won/lost by 15+ points

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
        elif 'PTS' in team_game_logs.columns:
            # If we have PTS but not opponent PTS, we can't calculate this
            # Will be handled when opponent stats are merged
            point_diff = pd.Series(0, index=team_game_logs.index)
        else:
            point_diff = pd.Series(0, index=team_game_logs.index)

        # Standard deviation of point differential (volatility)
        volatility_features['std_point_diff'] = point_diff.rolling(
            window=window, min_periods=1
        ).std()

        # Coefficient of variation (normalized volatility)
        mean_diff = point_diff.rolling(window=window, min_periods=1).mean()
        volatility_features['cv_point_diff'] = (
            volatility_features['std_point_diff'] / (abs(mean_diff) + 1)  # +1 to avoid division by zero
        )

        # Close game percentage (games within 5 points)
        close_games = (abs(point_diff) <= 5).astype(int)
        volatility_features['close_game_pct'] = close_games.rolling(
            window=window, min_periods=1
        ).mean()

        # Blowout percentage (games won/lost by 15+ points)
        blowouts = (abs(point_diff) >= 15).astype(int)
        volatility_features['blowout_pct'] = blowouts.rolling(
            window=window, min_periods=1
        ).mean()

        # Win streak features
        if 'WL' in team_game_logs.columns:
            wins = (team_game_logs['WL'] == 'W').astype(int)
            volatility_features['win_pct'] = wins.rolling(
                window=window, min_periods=1
            ).mean()

            # Win/loss streaks (consecutive wins/losses)
            win_streak = self._calculate_current_streak(team_game_logs['WL'])
            volatility_features['current_streak'] = win_streak

        # Scoring volatility (std of points scored)
        if 'PTS' in team_game_logs.columns:
            volatility_features['std_pts_scored'] = team_game_logs['PTS'].rolling(
                window=window, min_periods=1
            ).std()

            # Scoring trend (recent average vs season average)
            recent_avg = team_game_logs['PTS'].rolling(window=window, min_periods=1).mean()
            season_avg = team_game_logs['PTS'].expanding(min_periods=1).mean()
            volatility_features['pts_trend'] = recent_avg - season_avg

        return volatility_features

    def _calculate_current_streak(self, wl_series: pd.Series) -> pd.Series:
        """
        Calculate current win/loss streak.

        Positive = win streak, Negative = loss streak
        """
        streak = pd.Series(0, index=wl_series.index)
        current = 0

        for i, result in enumerate(wl_series):
            if result == 'W':
                current = current + 1 if current >= 0 else 1
            elif result == 'L':
                current = current - 1 if current <= 0 else -1
            else:
                current = 0

            streak.iloc[i] = current

        return streak
    
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
        - Combined volatility metrics

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

        # Competitive balance score (CRITICAL for our target)
        # Lower = more evenly matched = more likely both teams lead by 5+
        if 'PTS_L10' in home_team_features and 'PTS_L10' in away_team_features:
            matchup_features['competitive_balance'] = abs(
                home_team_features['PTS_L10'] - away_team_features['PTS_L10']
            )

            # Win percentage difference
            if 'win_pct' in home_team_features and 'win_pct' in away_team_features:
                matchup_features['win_pct_diff'] = abs(
                    home_team_features['win_pct'] - away_team_features['win_pct']
                )

                # Competitive matchup indicator (both teams similar strength)
                # TRUE if win percentages within 0.15 of each other
                win_pct_close = abs(home_team_features['win_pct'] - away_team_features['win_pct']) < 0.15
                matchup_features['evenly_matched'] = 1.0 if win_pct_close else 0.0

        # Combined volatility (sum of both teams' volatility)
        # Higher = both teams play volatile games = more likely for lead swings
        if 'std_point_diff' in home_team_features and 'std_point_diff' in away_team_features:
            matchup_features['combined_volatility'] = (
                home_team_features['std_point_diff'] + away_team_features['std_point_diff']
            )

        # Combined close game percentage
        if 'close_game_pct' in home_team_features and 'close_game_pct' in away_team_features:
            matchup_features['combined_close_game_pct'] = (
                home_team_features['close_game_pct'] + away_team_features['close_game_pct']
            ) / 2.0

        # Pace matchup (fast vs slow)
        # High combined pace = more possessions = more opportunities for leads
        if 'pace_L10' in home_team_features and 'pace_L10' in away_team_features:
            matchup_features['combined_pace'] = (
                home_team_features['pace_L10'] + away_team_features['pace_L10']
            ) / 2.0

            matchup_features['pace_diff'] = abs(
                home_team_features['pace_L10'] - away_team_features['pace_L10']
            )

        # Offensive vs Defensive matchups
        if 'offensive_rating_L10' in home_team_features and 'defensive_rating_L10' in away_team_features:
            # Home offense vs Away defense
            matchup_features['home_off_vs_away_def'] = (
                home_team_features['offensive_rating_L10'] - away_team_features['defensive_rating_L10']
            )

        if 'offensive_rating_L10' in away_team_features and 'defensive_rating_L10' in home_team_features:
            # Away offense vs Home defense
            matchup_features['away_off_vs_home_def'] = (
                away_team_features['offensive_rating_L10'] - home_team_features['defensive_rating_L10']
            )

        # Blowout tendency differential
        if 'blowout_pct' in home_team_features and 'blowout_pct' in away_team_features:
            # Low combined blowout % suggests competitive game
            matchup_features['combined_blowout_pct'] = (
                home_team_features['blowout_pct'] + away_team_features['blowout_pct']
            ) / 2.0

        # Momentum differential (streaks)
        if 'current_streak' in home_team_features and 'current_streak' in away_team_features:
            matchup_features['momentum_diff'] = (
                home_team_features['current_streak'] - away_team_features['current_streak']
            )

        return pd.Series(matchup_features)
