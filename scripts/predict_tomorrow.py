#!/usr/bin/env python3
"""
Tomorrow's Game Predictions

Generates predictions for tomorrow's NBA games using:
1. Latest data (yesterday's + today's results)
2. Most recent team features (rolling averages, streaks, etc.)
3. Trained model

Usage:
    python scripts/predict_tomorrow.py --model models/best_model_xgboost.joblib
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.daily_updater import TomorrowPredictor
from src.utils.config import get_config
import logging
import argparse
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Predict tomorrow\'s NBA games')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Data directory'
    )
    parser.add_argument(
        '--no-update',
        action='store_true',
        help='Skip data update (use existing data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/predictions/tomorrow.csv',
        help='Output file for predictions'
    )

    args = parser.parse_args()

    # Create predictor
    config = get_config()
    predictor = TomorrowPredictor(
        model_path=Path(args.model),
        data_dir=Path(args.data_dir),
        config=config
    )

    # Generate predictions
    try:
        predictions_df = predictor.predict_tomorrow(
            update_data=not args.no_update
        )

        if predictions_df.empty:
            logger.warning("No games to predict for tomorrow")
            return

        # Save predictions
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(output_path, index=False)

        logger.info(f"\n✅ Predictions saved to {output_path}")

        # Display predictions
        print("\n" + "=" * 80)
        print("TOMORROW'S PREDICTIONS")
        print("=" * 80)
        print(predictions_df.to_string(index=False))
        print("=" * 80)

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
