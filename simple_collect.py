"""
Simplified data collection using nba_api defaults (no custom headers)
"""

import logging
import time
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv2, boxscoretraditionalv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_simple(season='2022-23'):
    """Collect data using nba_api defaults."""

    logger.info(f"Starting simple collection for {season}")
    save_dir = Path(f'data/raw/{season}')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get games
    logger.info("Fetching games list...")
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable='Regular Season'
        )
        games_df = gamefinder.get_data_frames()[0]
        games_df = games_df.drop_duplicates(subset=['GAME_ID'], keep='first')

        games_file = save_dir / f'games_{season}.csv'
        games_df.to_csv(games_file, index=False)
        logger.info(f"✓ Saved {len(games_df)} games to {games_file}")

    except Exception as e:
        logger.error(f"Failed to get games: {e}")
        return

    # Step 2: Get play-by-play for first 10 games (test)
    logger.info(f"Fetching play-by-play for first 10 games...")
    pbp_dir = save_dir / 'play_by_play'
    pbp_dir.mkdir(exist_ok=True)

    for idx, game_id in enumerate(games_df['GAME_ID'].head(10), 1):
        try:
            logger.info(f"  Game {idx}/10: {game_id}")
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            pbp_df = pbp.get_data_frames()[0]

            pbp_file = pbp_dir / f'{game_id}.csv'
            pbp_df.to_csv(pbp_file, index=False)

            time.sleep(0.7)  # Rate limit

        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue

    logger.info(f"Collection complete! Data saved to {save_dir}")

if __name__ == '__main__':
    collect_simple('2022-23')
