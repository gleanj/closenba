"""
Single Season Data Collection Script (for testing)

Collects data for just ONE season to test the setup.
Use this before running the full collection.
"""

import logging
from pathlib import Path
from src.data.collectors import NBADataCollector
from src.utils.config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Collect data for a single season to test the setup."""

    # Test with 2022-23 season (2023-24 is timing out)
    season = '2022-23'

    logger.info("=" * 80)
    logger.info(f"TEST DATA COLLECTION - Season {season}")
    logger.info("=" * 80)
    logger.info("This is a test run to verify the data collection pipeline.")
    logger.info(f"Estimated time: 2-3 hours for ~1,230 games")
    logger.info("=" * 80)

    # Load configuration
    config = get_config()

    logger.info(f"Rate limit: {config.rate_limit_interval:.3f} seconds between requests")
    logger.info("=" * 80)

    # Initialize collector
    collector = NBADataCollector(config)

    # Collect data
    try:
        save_dir = Path(f'data/raw/{season}')

        logger.info(f"\nStarting collection for {season}...")
        saved_files = collector.collect_season_data(season, save_dir)

        logger.info("\n" + "=" * 80)
        logger.info("TEST COLLECTION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Data saved to: {save_dir}")
        logger.info("\nNext steps:")
        logger.info("1. Review the logs to ensure no errors")
        logger.info("2. Check the saved files in data/raw/2023-24/")
        logger.info("3. If successful, run collect_data.py for all seasons")

    except Exception as e:
        logger.error(f"Error during collection: {e}", exc_info=True)
        logger.info("\nIf you see rate limiting errors, the script will automatically retry.")
        logger.info("If you see connection errors, check your internet connection.")

if __name__ == '__main__':
    main()
