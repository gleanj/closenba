"""
Daily Data Updater for Real-Time Predictions

Fetches yesterday's/today's games and updates the dataset for tomorrow's predictions.
Critical for maintaining feature accuracy (rolling averages, streaks, etc.)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path
from datetime import datetime, timedelta
import time

from .collectors import NBADataCollector
from .labelers import GameLabeler
from ..utils.config import get_config

logger = logging.getLogger(__name__)


class DailyUpdater:
    """
    Fetches and processes recent games to keep the dataset current.

    Critical for real-time predictions:
    - Rolling features (L5, L10, L20) need latest game results
    - Momentum features (streaks, trends) change daily
    - Stale data = poor predictions
    """

    def __init__(self, data_dir: Path, config=None):
        """
        Args:
            data_dir: Directory containing existing data
            config: Configuration object
        """
        self.data_dir = Path(data_dir)
        self.config = config or get_config()
        self.collector = NBADataCollector(self.config)
        self.labeler = GameLabeler(threshold=self.config.target_threshold)

    def get_yesterday_games(self) -> pd.DataFrame:
        """
        Fetch all games from yesterday.

        Returns:
            DataFrame with yesterday's games
        """
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_str = yesterday.strftime('%Y-%m-%d')

        logger.info(f"Fetching games from {yesterday_str}...")

        try:
            # Get current season string (e.g., '2024-25')
            current_year = datetime.now().year
            current_month = datetime.now().month

            if current_month >= 10:  # NBA season starts in October
                season = f"{current_year}-{str(current_year + 1)[2:]}"
            else:
                season = f"{current_year - 1}-{str(current_year)[2:]}"

            # Fetch all games for the season
            all_games = self.collector.get_season_games(season)

            # Filter for yesterday's games
            all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
            yesterday_games = all_games[
                all_games['GAME_DATE'].dt.date == yesterday.date()
            ]

            logger.info(f"Found {len(yesterday_games)} games from yesterday")

            return yesterday_games

        except Exception as e:
            logger.error(f"Error fetching yesterday's games: {e}")
            return pd.DataFrame()

    def get_today_games(self) -> pd.DataFrame:
        """
        Fetch all games from today (for intraday updates).

        Returns:
            DataFrame with today's completed games
        """
        today = datetime.now()
        today_str = today.strftime('%Y-%m-%d')

        logger.info(f"Fetching games from {today_str}...")

        try:
            current_year = datetime.now().year
            current_month = datetime.now().month

            if current_month >= 10:
                season = f"{current_year}-{str(current_year + 1)[2:]}"
            else:
                season = f"{current_year - 1}-{str(current_year)[2:]}"

            all_games = self.collector.get_season_games(season)
            all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])

            today_games = all_games[
                all_games['GAME_DATE'].dt.date == today.date()
            ]

            logger.info(f"Found {len(today_games)} games from today")

            return today_games

        except Exception as e:
            logger.error(f"Error fetching today's games: {e}")
            return pd.DataFrame()

    def fetch_game_details(self, game_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch play-by-play and boxscore for games.

        Args:
            game_ids: List of game IDs to fetch

        Returns:
            Dictionary mapping game_id to {pbp, boxscore, labels}
        """
        game_data = {}

        logger.info(f"Fetching details for {len(game_ids)} games...")

        for idx, game_id in enumerate(game_ids, 1):
            try:
                # Fetch play-by-play
                pbp = self.collector.get_play_by_play(game_id)

                # Fetch boxscore
                boxscore = self.collector.get_boxscore(game_id)

                # Label the game
                labels = self.labeler.label_game_from_pbp(pbp)

                game_data[game_id] = {
                    'play_by_play': pbp,
                    'boxscore': boxscore,
                    'labels': labels
                }

                if idx % 10 == 0:
                    logger.info(f"Progress: {idx}/{len(game_ids)} games fetched")

            except Exception as e:
                logger.error(f"Error fetching game {game_id}: {e}")
                continue

        return game_data

    def update_dataset(
        self,
        games_df: pd.DataFrame,
        save: bool = True
    ) -> Dict[str, Path]:
        """
        Update the main dataset with new games.

        Args:
            games_df: DataFrame with new games to add
            save: Whether to save updated data

        Returns:
            Dictionary of saved file paths
        """
        if games_df.empty:
            logger.warning("No games to update")
            return {}

        logger.info(f"Updating dataset with {len(games_df)} new games...")

        # Fetch game details
        game_ids = games_df['GAME_ID'].unique()
        game_details = self.fetch_game_details(game_ids)

        if not game_details:
            logger.error("Failed to fetch any game details")
            return {}

        saved_files = {}

        if save:
            # Save to incremental directory
            update_dir = self.data_dir / 'incremental' / datetime.now().strftime('%Y-%m-%d')
            update_dir.mkdir(parents=True, exist_ok=True)

            # Save play-by-play
            pbp_dir = update_dir / 'play_by_play'
            pbp_dir.mkdir(exist_ok=True)

            # Save boxscores
            boxscore_dir = update_dir / 'boxscores'
            boxscore_dir.mkdir(exist_ok=True)

            # Save labels
            labels_list = []

            for game_id, data in game_details.items():
                # Save PBP
                pbp_file = pbp_dir / f'{game_id}.csv'
                data['play_by_play'].to_csv(pbp_file, index=False)

                # Save boxscore
                player_file = boxscore_dir / f'{game_id}_players.csv'
                team_file = boxscore_dir / f'{game_id}_team.csv'
                data['boxscore']['player_stats'].to_csv(player_file, index=False)
                data['boxscore']['team_stats'].to_csv(team_file, index=False)

                # Collect labels
                label = data['labels'].copy()
                label['game_id'] = game_id
                labels_list.append(label)

            # Save labels
            if labels_list:
                labels_df = pd.DataFrame(labels_list)
                labels_file = update_dir / 'labels.csv'
                labels_df.to_csv(labels_file, index=False)
                saved_files['labels'] = labels_file

            saved_files['update_dir'] = update_dir

            logger.info(f"Saved {len(game_details)} games to {update_dir}")

        return saved_files

    def run_daily_update(self, include_today: bool = False) -> Dict:
        """
        Run the daily update process.

        Args:
            include_today: Whether to include today's games (for evening updates)

        Returns:
            Dictionary with update summary
        """
        logger.info("=" * 60)
        logger.info("STARTING DAILY UPDATE")
        logger.info("=" * 60)

        # Fetch yesterday's games
        yesterday_games = self.get_yesterday_games()

        all_games = yesterday_games

        # Optionally include today's games
        if include_today:
            today_games = self.get_today_games()
            all_games = pd.concat([yesterday_games, today_games], ignore_index=True)

        # Update dataset
        saved_files = self.update_dataset(all_games, save=True)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'games_updated': len(all_games),
            'yesterday_games': len(yesterday_games),
            'today_games': len(self.get_today_games()) if include_today else 0,
            'saved_files': {k: str(v) for k, v in saved_files.items()}
        }

        logger.info("=" * 60)
        logger.info("DAILY UPDATE COMPLETE")
        logger.info(f"Updated {summary['games_updated']} games")
        logger.info("=" * 60)

        return summary


class TomorrowPredictor:
    """
    Predicts outcomes for tomorrow's games using the most recent data.

    Workflow:
    1. Update dataset with yesterday's/today's results
    2. Recalculate features with latest data
    3. Fetch tomorrow's schedule
    4. Generate predictions
    """

    def __init__(
        self,
        model_path: Path,
        data_dir: Path,
        config=None
    ):
        """
        Args:
            model_path: Path to trained model
            data_dir: Directory containing data
            config: Configuration object
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.config = config or get_config()
        self.collector = NBADataCollector(self.config)
        self.updater = DailyUpdater(data_dir, config)

        # Load trained model
        from ..models.ensemble import EnsembleModel
        from ..models.baseline import BaselineModel

        try:
            self.model = EnsembleModel.load(model_path)
        except:
            try:
                self.model = BaselineModel.load(model_path)
            except Exception as e:
                logger.error(f"Could not load model from {model_path}: {e}")
                self.model = None

    def get_tomorrow_schedule(self) -> pd.DataFrame:
        """
        Fetch tomorrow's game schedule.

        Returns:
            DataFrame with tomorrow's scheduled games
        """
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_str = tomorrow.strftime('%Y-%m-%d')

        logger.info(f"Fetching schedule for {tomorrow_str}...")

        try:
            current_year = datetime.now().year
            current_month = datetime.now().month

            if current_month >= 10:
                season = f"{current_year}-{str(current_year + 1)[2:]}"
            else:
                season = f"{current_year - 1}-{str(current_year)[2:]}"

            all_games = self.collector.get_season_games(season)
            all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])

            tomorrow_games = all_games[
                all_games['GAME_DATE'].dt.date == tomorrow.date()
            ]

            logger.info(f"Found {len(tomorrow_games)} games scheduled for tomorrow")

            return tomorrow_games

        except Exception as e:
            logger.error(f"Error fetching tomorrow's schedule: {e}")
            return pd.DataFrame()

    def predict_tomorrow(self, update_data: bool = True) -> pd.DataFrame:
        """
        Generate predictions for tomorrow's games.

        Args:
            update_data: Whether to update with latest results first

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        logger.info("=" * 60)
        logger.info("GENERATING TOMORROW'S PREDICTIONS")
        logger.info("=" * 60)

        # Step 1: Update data with latest results
        if update_data:
            logger.info("Step 1: Updating with latest results...")
            self.updater.run_daily_update(include_today=True)

        # Step 2: Get tomorrow's schedule
        logger.info("Step 2: Fetching tomorrow's schedule...")
        tomorrow_games = self.get_tomorrow_schedule()

        if tomorrow_games.empty:
            logger.warning("No games scheduled for tomorrow")
            return pd.DataFrame()

        # Step 3: Calculate features for each game
        logger.info("Step 3: Calculating features...")
        # NOTE: This would require the full feature engineering pipeline
        # For now, this is a placeholder
        # You would need to:
        # - Load team stats with latest games included
        # - Calculate rolling features for each team
        # - Create matchup features
        # - Format as model input

        logger.warning("Feature calculation not yet implemented - placeholder")

        # Step 4: Generate predictions
        # predictions = self.model.predict_proba(features)

        logger.info("=" * 60)
        logger.info("PREDICTIONS COMPLETE")
        logger.info("=" * 60)

        return tomorrow_games
