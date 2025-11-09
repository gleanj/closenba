"""
Rate Limiter for nba_api

Critical utility to prevent IP bans from stats.nba.com.
Based on research showing 600ms between requests is safe threshold.
"""

import time
from functools import wraps
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter to prevent overwhelming stats.nba.com API.
    
    From research: stats.nba.com aggressively throttles and blocks IPs.
    Safe threshold: ~600ms between requests (1.4 requests/second).
    """
    
    def __init__(self, min_interval: float = 0.7):
        """
        Args:
            min_interval: Minimum seconds between API calls (default 0.7 = 700ms)
        """
        self.min_interval = min_interval
        self.last_call = 0.0
        
    def wait(self):
        """Wait until enough time has passed since last API call."""
        current_time = time.time()
        time_since_last = current_time - self.last_call
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.3f}s")
            time.sleep(wait_time)
            
        self.last_call = time.time()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to automatically rate limit function calls."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            self.wait()
            return func(*args, **kwargs)
        return wrapper


# Global rate limiter instance
rate_limiter = RateLimiter(min_interval=0.7)


def rate_limited(func: Callable) -> Callable:
    """
    Decorator for rate-limiting API calls.
    
    Usage:
        @rate_limited
        def get_player_stats(player_id):
            return playercareerstats.PlayerCareerStats(player_id=player_id)
    """
    return rate_limiter(func)


def exponential_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator for exponential backoff on failed API calls.
    
    Handles common nba_api errors:
    - ReadTimeout: Server took too long
    - JSONDecodeError: Rate limited (server sent HTML instead of JSON)
    - ConnectionError: Network issues
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles each retry)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    error_type = type(e).__name__
                    
                    # Don't retry on certain errors
                    if 'KeyError' in error_type or 'ValueError' in error_type:
                        raise
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    
                    # Calculate backoff delay
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: "
                        f"{error_type}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    
            return None  # Should never reach here
        return wrapper
    return decorator
