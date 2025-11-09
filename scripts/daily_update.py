#!/usr/bin/env python3
"""
Daily Update Script

Run this script daily (e.g., via cron job) to:
1. Fetch yesterday's game results
2. Update the dataset
3. Regenerate features with latest data
4. Prepare for tomorrow's predictions

Usage:
    python scripts/daily_update.py
    python scripts/daily_update.py --include-today  # For evening runs
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.daily_updater import DailyUpdater
from src.utils.config import get_config
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('outputs/logs/daily_update.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Update dataset with recent games')
    parser.add_argument(
        '--include-today',
        action='store_true',
        help='Include today\'s games (for evening updates)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Data directory'
    )

    args = parser.parse_args()

    # Create updater
    config = get_config()
    updater = DailyUpdater(
        data_dir=Path(args.data_dir),
        config=config
    )

    # Run update
    try:
        summary = updater.run_daily_update(include_today=args.include_today)

        logger.info("Update Summary:")
        logger.info(f"  Games updated: {summary['games_updated']}")
        logger.info(f"  Yesterday: {summary['yesterday_games']} games")
        if args.include_today:
            logger.info(f"  Today: {summary['today_games']} games")

        logger.info("\n✅ Daily update completed successfully!")

    except Exception as e:
        logger.error(f"❌ Daily update failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
