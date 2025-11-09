"""Date and season utility functions for NBA data processing."""

from datetime import datetime, date, timedelta
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def get_current_nba_season() -> str:
    """
    Calculate the current NBA season string based on current date.

    NBA seasons run from October to April, so:
    - Oct-Dec: Current year season (e.g., Oct 2023 -> "2023-24")
    - Jan-Sep: Previous year season (e.g., Jan 2024 -> "2023-24")

    Returns:
        str: Season string in format "YYYY-YY" (e.g., "2023-24")

    Example:
        >>> # If called in November 2023
        >>> get_current_nba_season()
        '2023-24'
        >>> # If called in March 2024
        >>> get_current_nba_season()
        '2023-24'
    """
    current_year = datetime.now().year
    current_month = datetime.now().month

    if current_month >= 10:  # Oct, Nov, Dec
        season = f"{current_year}-{str(current_year + 1)[2:]}"
    else:  # Jan-Sep
        season = f"{current_year - 1}-{str(current_year)[2:]}"

    return season


def parse_season_string(season: str) -> Tuple[int, int]:
    """
    Parse NBA season string to start and end years.

    Args:
        season: Season string in format "YYYY-YY" (e.g., "2023-24")

    Returns:
        Tuple of (start_year, end_year)

    Raises:
        ValueError: If season string format is invalid

    Example:
        >>> parse_season_string("2023-24")
        (2023, 2024)
    """
    try:
        parts = season.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid season format: {season}")

        start_year = int(parts[0])

        # Handle 2-digit or 4-digit end year
        if len(parts[1]) == 2:
            end_year = int(f"{str(start_year)[:2]}{parts[1]}")
        else:
            end_year = int(parts[1])

        if end_year != start_year + 1:
            raise ValueError(f"Invalid season range: {season}")

        return start_year, end_year

    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid season string '{season}': {e}")


def get_season_for_date(game_date: date) -> str:
    """
    Get the NBA season string for a given date.

    Args:
        game_date: Date of the game

    Returns:
        str: Season string in format "YYYY-YY"

    Example:
        >>> from datetime import date
        >>> get_season_for_date(date(2023, 12, 25))
        '2023-24'
        >>> get_season_for_date(date(2024, 3, 15))
        '2023-24'
    """
    year = game_date.year
    month = game_date.month

    if month >= 10:  # Oct-Dec: part of current year's season
        return f"{year}-{str(year + 1)[2:]}"
    else:  # Jan-Sep: part of previous year's season
        return f"{year - 1}-{str(year)[2:]}"


def get_yesterday() -> date:
    """Get yesterday's date."""
    return date.today() - timedelta(days=1)


def get_tomorrow() -> date:
    """Get tomorrow's date."""
    return date.today() + timedelta(days=1)


def is_nba_season_active(check_date: date = None) -> bool:
    """
    Check if NBA regular season is typically active on given date.

    Note: This is approximate. Actual season dates vary by year.
    Regular season typically runs October through mid-April.

    Args:
        check_date: Date to check (defaults to today)

    Returns:
        bool: True if date falls in typical NBA season
    """
    if check_date is None:
        check_date = date.today()

    month = check_date.month

    # NBA season runs October (10) through April (4)
    # May-September is offseason
    return month >= 10 or month <= 4


def format_date_for_api(dt: date) -> str:
    """
    Format date for NBA API requests.

    Args:
        dt: Date to format

    Returns:
        str: Date formatted as "MM/DD/YYYY"

    Example:
        >>> from datetime import date
        >>> format_date_for_api(date(2023, 12, 25))
        '12/25/2023'
    """
    return dt.strftime("%m/%d/%Y")


def format_date_for_filename(dt: date) -> str:
    """
    Format date for use in filenames.

    Args:
        dt: Date to format

    Returns:
        str: Date formatted as "YYYY-MM-DD"

    Example:
        >>> from datetime import date
        >>> format_date_for_filename(date(2023, 12, 25))
        '2023-12-25'
    """
    return dt.strftime("%Y-%m-%d")
