"""
Quick test to see what works with the NBA API right now
"""

import logging
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test 1: Static data (this worked before)
logger.info("Test 1: Getting static teams data...")
all_teams = teams.get_teams()
logger.info(f"✓ Success: Got {len(all_teams)} teams")

# Test 2: Try a very simple API call with a short season
logger.info("\nTest 2: Trying to fetch games from 2022-23 season...")
logger.info("This will take up to 2 minutes if it times out...")

try:
    # Try with minimal parameters and shorter timeout
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable='2022-23',
        season_type_nullable='Regular Season',
        timeout=30  # Shorter timeout
    )
    games_df = gamefinder.get_data_frames()[0]
    logger.info(f"✓ Success: Got {len(games_df)} game records")
    logger.info(f"Sample game: {games_df.iloc[0]['GAME_ID']}")
except Exception as e:
    logger.error(f"✗ Failed: {type(e).__name__}: {str(e)[:100]}")
    logger.info("\nThe NBA API is currently not responding.")
    logger.info("Recommendations:")
    logger.info("1. Try again in a few hours (off-peak times work best)")
    logger.info("2. Check if stats.nba.com is accessible in your browser")
    logger.info("3. Consider using a VPN if the site is region-blocked")
