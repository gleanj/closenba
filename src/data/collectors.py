"""
NBA Data Collector using nba_api

Collects historical game data with defensive rate limiting and error handling.
Based on comprehensive nba_api research to avoid IP bans.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
import time
from pathlib import Path
import json

# nba_api imports
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import (
    leaguegamefinder,
    playbyplayv2,
    boxscoretraditionalv2,
    teamgamelogs,
    LeagueGameFinder
)

from ..utils.rate_limiter import rate_limited, exponential_backoff
from ..utils.config import get_config

logger = logging.getLogger(__name__)


class NBADataCollector:
    """
    Collects NBA data using nba_api with proper rate limiting.
    
    Critical safeguards:
    - 700ms delay between requests (prevents IP ban)
    - Exponential backoff on errors
    - Custom headers to mimic browser
    - Timeout handling for slow queries
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.headers = self.config.api_headers
        self.timeout = self.config.api_timeout
        
        # Static data (loaded once to reduce API calls)
        logger.info("Loading static NBA data (teams, players)...")
        self.all_teams = teams.get_teams()
        self.all_players = players.get_players()
        logger.info(f"Loaded {len(self.all_teams)} teams, {len(self.all_players)} players")
    
    def get_team_id(self, team_abbr: str) -> Optional[int]:
        """Get team ID from abbreviation."""
        team = teams.find_team_by_abbreviation(team_abbr)
        return team['id'] if team else None
    
    def get_player_id(self, player_name: str) -> Optional[int]:
        """Get player ID from full name."""
        player_list = players.find_players_by_full_name(player_name)
        return player_list[0]['id'] if player_list else None
    
    @rate_limited
    @exponential_backoff(max_retries=3, base_delay=2.0)
    def get_season_games(self, season: str, season_type: str = 'Regular Season') -> pd.DataFrame:
        """
        Get all games for a season.
        
        Args:
            season: Season string (e.g., '2023-24')
            season_type: 'Regular Season', 'Playoffs', etc.
        
        Returns:
            DataFrame with game information
        """
        logger.info(f"Fetching games for season {season} ({season_type})")
        
        try:
            gamefinder = LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                headers=self.headers,
                timeout=self.timeout
            )
            
            games_df = gamefinder.get_data_frames()[0]
            
            # Remove duplicate games (each game appears twice, once per team)
            games_df = games_df.drop_duplicates(subset=['GAME_ID'], keep='first')
            
            logger.info(f"Found {len(games_df)} games for {season}")
            return games_df
            
        except Exception as e:
            logger.error(f"Error fetching games for {season}: {e}")
            raise
    
    @rate_limited
    @exponential_backoff(max_retries=3, base_delay=2.0)
    def get_play_by_play(self, game_id: str) -> pd.DataFrame:
        """
        Get play-by-play data for a specific game.
        
        This is CRITICAL for labeling our target: both teams led by 5+.
        
        Args:
            game_id: NBA game ID (e.g., '0022300001')
        
        Returns:
            DataFrame with play-by-play data
        """
        logger.debug(f"Fetching play-by-play for game {game_id}")
        
        try:
            pbp = playbyplayv2.PlayByPlayV2(
                game_id=game_id,
                headers=self.headers,
                timeout=self.timeout
            )
            
            pbp_df = pbp.get_data_frames()[0]
            return pbp_df
            
        except Exception as e:
            logger.error(f"Error fetching PBP for game {game_id}: {e}")
            raise
    
    @rate_limited
    @exponential_backoff(max_retries=3, base_delay=2.0)
    def get_boxscore(self, game_id: str) -> Dict[str, pd.DataFrame]:
        """
        Get boxscore data for a game.
        
        Args:
            game_id: NBA game ID
        
        Returns:
            Dictionary with:
                - 'player_stats': Player-level statistics
                - 'team_stats': Team-level statistics
        """
        logger.debug(f"Fetching boxscore for game {game_id}")
        
        try:
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=game_id,
                headers=self.headers,
                timeout=self.timeout
            )
            
            dfs = boxscore.get_data_frames()
            
            return {
                'player_stats': dfs[0],  # PlayerStats
                'team_stats': dfs[1],     # TeamStats
            }
            
        except Exception as e:
            logger.error(f"Error fetching boxscore for game {game_id}: {e}")
            raise
    
    @rate_limited
    @exponential_backoff(max_retries=3, base_delay=2.0)
    def get_team_game_logs(self, team_id: int, season: str) -> pd.DataFrame:
        """
        Get game-by-game logs for a team in a season.
        
        Args:
            team_id: NBA team ID
            season: Season string (e.g., '2023-24')
        
        Returns:
            DataFrame with game logs
        """
        logger.debug(f"Fetching game logs for team {team_id}, season {season}")
        
        try:
            logs = teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id,
                season_nullable=season,
                headers=self.headers,
                timeout=self.timeout
            )
            
            return logs.get_data_frames()[0]
            
        except Exception as e:
            logger.error(f"Error fetching game logs for team {team_id}: {e}")
            raise
    
    def collect_season_data(self, season: str, save_dir: Path) -> Dict[str, Path]:
        """
        Collect complete dataset for a season.
        
        This is the main orchestration method that collects:
        1. Game list
        2. Play-by-play for each game (for labeling)
        3. Boxscores for each game (for features)
        
        Args:
            season: Season string (e.g., '2023-24')
            save_dir: Directory to save raw data
        
        Returns:
            Dictionary of saved file paths
        """
        logger.info(f"=" * 60)
        logger.info(f"Starting data collection for season {season}")
        logger.info(f"=" * 60)
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Step 1: Get all games for the season
        games_df = self.get_season_games(season)
        games_file = save_dir / f'games_{season}.csv'
        games_df.to_csv(games_file, index=False)
        saved_files['games'] = games_file
        logger.info(f"Saved {len(games_df)} games to {games_file}")
        
        # Step 2: Collect play-by-play data (CRITICAL for labeling)
        logger.info(f"Collecting play-by-play data for {len(games_df)} games...")
        logger.info(f"This will take approximately {len(games_df) * 0.7 / 60:.1f} minutes")
        
        pbp_dir = save_dir / 'play_by_play'
        pbp_dir.mkdir(exist_ok=True)
        
        successful_pbp = 0
        failed_pbp = []
        
        for idx, game_id in enumerate(games_df['GAME_ID'].unique(), 1):
            try:
                pbp_df = self.get_play_by_play(game_id)
                pbp_file = pbp_dir / f'{game_id}.csv'
                pbp_df.to_csv(pbp_file, index=False)
                successful_pbp += 1
                
                if idx % 50 == 0:
                    logger.info(f"Progress: {idx}/{len(games_df)} games ({successful_pbp} successful)")
                    
            except Exception as e:
                logger.error(f"Failed to get PBP for game {game_id}: {e}")
                failed_pbp.append(game_id)
        
        saved_files['play_by_play_dir'] = pbp_dir
        logger.info(f"Play-by-play: {successful_pbp} successful, {len(failed_pbp)} failed")
        
        # Step 3: Collect boxscores
        logger.info(f"Collecting boxscore data...")
        
        boxscore_dir = save_dir / 'boxscores'
        boxscore_dir.mkdir(exist_ok=True)
        
        successful_box = 0
        
        for idx, game_id in enumerate(games_df['GAME_ID'].unique(), 1):
            if game_id in failed_pbp:
                continue  # Skip games that already failed
                
            try:
                boxscore_data = self.get_boxscore(game_id)
                
                # Save player and team stats separately
                player_file = boxscore_dir / f'{game_id}_players.csv'
                team_file = boxscore_dir / f'{game_id}_team.csv'
                
                boxscore_data['player_stats'].to_csv(player_file, index=False)
                boxscore_data['team_stats'].to_csv(team_file, index=False)
                
                successful_box += 1
                
                if idx % 50 == 0:
                    logger.info(f"Boxscores: {successful_box}/{len(games_df)} completed")
                    
            except Exception as e:
                logger.error(f"Failed to get boxscore for game {game_id}: {e}")
        
        saved_files['boxscores_dir'] = boxscore_dir
        logger.info(f"Boxscores: {successful_box} successful")
        
        # Save collection summary
        summary = {
            'season': season,
            'total_games': len(games_df),
            'successful_pbp': successful_pbp,
            'successful_boxscores': successful_box,
            'failed_games': failed_pbp,
            'collection_date': pd.Timestamp.now().isoformat()
        }
        
        summary_file = save_dir / f'collection_summary_{season}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"=" * 60)
        logger.info(f"Data collection complete for {season}")
        logger.info(f"Summary saved to {summary_file}")
        logger.info(f"=" * 60)
        
        return saved_files
