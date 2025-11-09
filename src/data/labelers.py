"""
Game Labeler: Determine if both teams led by 5+ points at any point in the game.

This is the core module that defines our unique prediction target.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class GameLabeler:
    """
    Labels NBA games based on whether both teams led by 5+ points at any point.
    
    This target is different from traditional win/loss or spread predictions.
    It measures game volatility, competitiveness, and momentum swings.
    """
    
    def __init__(self, threshold: int = 5):
        """
        Args:
            threshold: Minimum lead in points (default 5)
        """
        self.threshold = threshold
        
    def label_game_from_pbp(self, play_by_play_df: pd.DataFrame) -> Dict:
        """
        Analyze play-by-play data to determine if both teams led by 5+.
        
        Args:
            play_by_play_df: Play-by-play DataFrame with columns:
                - SCOREMARGIN (point differential as string, e.g., "LAL 5", "TIE", "BOS 3")
                - Or SCORE (home score) and VISITORSCORE (visitor score)
        
        Returns:
            dict with:
                - both_teams_led_5plus: bool (our target variable)
                - home_max_lead: int
                - away_max_lead: int
                - total_lead_changes: int
                - game_volatility_score: float
        """
        if play_by_play_df.empty:
            logger.warning("Empty play-by-play data")
            return self._get_default_label()
        
        # Calculate point differential for each play
        scores = self._extract_scores(play_by_play_df)
        
        if scores is None:
            return self._get_default_label()
        
        home_scores, away_scores = scores
        point_diff = home_scores - away_scores  # Positive = home leading
        
        # Find maximum leads for each team
        home_max_lead = point_diff.max()
        away_max_lead = abs(point_diff.min())
        
        # Check if both teams led by threshold at any point
        home_led_by_threshold = home_max_lead >= self.threshold
        away_led_by_threshold = away_max_lead >= self.threshold
        both_teams_led = home_led_by_threshold and away_led_by_threshold
        
        # Calculate additional volatility metrics
        lead_changes = self._count_lead_changes(point_diff)
        volatility_score = self._calculate_volatility(point_diff)
        
        return {
            'both_teams_led_5plus': both_teams_led,
            'home_max_lead': int(home_max_lead),
            'away_max_lead': int(away_max_lead),
            'total_lead_changes': lead_changes,
            'game_volatility_score': volatility_score,
            'threshold': self.threshold
        }
    
    def _extract_scores(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract home and away scores from play-by-play data.
        
        Handles different formats:
        1. SCORE and VISITORSCORE columns
        2. SCOREMARGIN column
        """
        # Method 1: Direct score columns
        if 'SCORE' in df.columns and 'VISITORSCORE' in df.columns:
            home_scores = pd.to_numeric(df['SCORE'], errors='coerce').fillna(method='ffill').fillna(0)
            away_scores = pd.to_numeric(df['VISITORSCORE'], errors='coerce').fillna(method='ffill').fillna(0)
            return home_scores.values, away_scores.values
        
        # Method 2: Parse SCOREMARGIN column
        elif 'SCOREMARGIN' in df.columns:
            return self._parse_score_margin(df['SCOREMARGIN'])
        
        # Method 3: Try to derive from other columns
        else:
            logger.warning("Could not find score columns in play-by-play data")
            return None
    
    def _parse_score_margin(self, score_margin: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse SCOREMARGIN column to extract point differential.
        
        Examples: "LAL 5", "TIE", "BOS 3"
        """
        # This is a simplified parser - may need adjustment based on actual data format
        point_diffs = []
        
        for margin in score_margin:
            if pd.isna(margin) or margin == '' or margin == 'TIE':
                point_diffs.append(0)
            else:
                # Extract number from string like "LAL 5"
                try:
                    parts = str(margin).split()
                    if len(parts) >= 2:
                        diff = int(parts[-1])
                        # Determine sign based on team abbreviation (simplified)
                        # In real implementation, need to know which team is home
                        point_diffs.append(diff)
                    else:
                        point_diffs.append(0)
                except:
                    point_diffs.append(0)
        
        # Convert to numpy array
        point_diff_array = np.array(point_diffs)
        
        # Derive scores (starting from 0-0)
        # This is approximate - for precise scores, use Method 1
        home_scores = np.maximum(0, point_diff_array).cumsum()
        away_scores = np.maximum(0, -point_diff_array).cumsum()
        
        return home_scores, away_scores
    
    def _count_lead_changes(self, point_diff: np.ndarray) -> int:
        """Count number of times the lead changed hands."""
        # A lead change occurs when sign of point_diff changes (excluding ties)
        non_zero_diff = point_diff[point_diff != 0]
        if len(non_zero_diff) <= 1:
            return 0
        
        sign_changes = np.diff(np.sign(non_zero_diff)) != 0
        return int(sign_changes.sum())
    
    def _calculate_volatility(self, point_diff: np.ndarray) -> float:
        """
        Calculate game volatility score.
        
        Higher volatility = more competitive/back-and-forth game
        Based on standard deviation of point differential.
        """
        if len(point_diff) == 0:
            return 0.0
        
        return float(np.std(point_diff))
    
    def _get_default_label(self) -> Dict:
        """Return default label when data is unavailable."""
        return {
            'both_teams_led_5plus': None,
            'home_max_lead': None,
            'away_max_lead': None,
            'total_lead_changes': None,
            'game_volatility_score': None,
            'threshold': self.threshold
        }
    
    def label_multiple_games(self, game_pbp_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Label multiple games at once.
        
        Args:
            game_pbp_dict: Dictionary mapping game_id to play-by-play DataFrame
        
        Returns:
            DataFrame with game_id and all label columns
        """
        results = []
        
        for game_id, pbp_df in game_pbp_dict.items():
            label = self.label_game_from_pbp(pbp_df)
            label['game_id'] = game_id
            results.append(label)
        
        return pd.DataFrame(results)


def calculate_label_statistics(labels_df: pd.DataFrame) -> Dict:
    """
    Calculate statistics about the labeled dataset.
    
    Args:
        labels_df: DataFrame with labeled games
    
    Returns:
        Dictionary with statistics
    """
    total_games = len(labels_df)
    both_teams_led = labels_df['both_teams_led_5plus'].sum()
    
    stats = {
        'total_games': total_games,
        'both_teams_led_count': int(both_teams_led),
        'both_teams_led_pct': both_teams_led / total_games if total_games > 0 else 0,
        'avg_home_max_lead': labels_df['home_max_lead'].mean(),
        'avg_away_max_lead': labels_df['away_max_lead'].mean(),
        'avg_lead_changes': labels_df['total_lead_changes'].mean(),
        'avg_volatility': labels_df['game_volatility_score'].mean(),
    }
    
    logger.info(f"Label Statistics:")
    logger.info(f"  Total games: {stats['total_games']}")
    logger.info(f"  Both teams led by 5+: {stats['both_teams_led_count']} "
                f"({stats['both_teams_led_pct']:.1%})")
    logger.info(f"  Avg lead changes: {stats['avg_lead_changes']:.1f}")
    logger.info(f"  Avg volatility: {stats['avg_volatility']:.1f}")
    
    return stats
