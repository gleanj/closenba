"""
Data Collection Script for CloseNBA Project

Collects NBA game data for configured seasons with proper rate limiting.
Run this script to start data collection (takes 2-3 hours per season).
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
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main data collection workflow."""

    logger.info("Starting NBA Data Collection")
    logger.info("=" * 80)

    # Load configuration
    config = get_config()
    seasons = config.seasons

    logger.info(f"Configured seasons: {seasons}")
    logger.info(f"Rate limit: {config.rate_limit_interval:.3f} seconds between requests")
    logger.info(f"Estimated time: {len(seasons) * 2.5:.1f} hours total")
    logger.info("=" * 80)

    # Initialize collector
    collector = NBADataCollector(config)

    # Collect data for each season
    for season in seasons:
        try:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"SEASON: {season}")
            logger.info(f"{'=' * 80}\n")

            # Create season directory
            save_dir = Path(f'data/raw/{season}')

            # Collect data
            saved_files = collector.collect_season_data(season, save_dir)

            logger.info(f"\nSeason {season} complete!")
            logger.info(f"Files saved to: {save_dir}")

        except Exception as e:
            logger.error(f"Error collecting data for season {season}: {e}", exc_info=True)
            logger.info("Continuing with next season...")
            continue

    logger.info("\n" + "=" * 80)
    logger.info("DATA COLLECTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"All data saved to: data/raw/")
    logger.info("\nNext steps:")
    logger.info("1. Review collection logs for any errors")
    logger.info("2. Label games using src/data/labelers.py")
    logger.info("3. Engineer features using src/features/")
    logger.info("4. Train models using src/models/pipeline.py")

if __name__ == '__main__':
    main()
