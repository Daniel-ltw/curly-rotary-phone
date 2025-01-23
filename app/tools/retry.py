import re
from tenacity import retry, stop_after_attempt

def parse_retry_time(exception):
    """Parse the retry time from rate limit error messages."""
    if "rate_limit_exceeded" in str(exception):
        match = re.search(r"Please try again in (\d+)m(\d+\.\d+)s", str(exception))
        if match:
            minutes = float(match.group(1))
            seconds = float(match.group(2)) + 1
            return minutes * 60 + seconds
        else:
            print(str(exception))
    else:
        print(str(exception))
    return 60

def wait_strategy(retry_state):
    """Custom wait strategy for retrying failed requests."""
    exception = retry_state.outcome.exception()
    print(f"Exception: {str(exception)}")
    wait_time = parse_retry_time(exception)
    return wait_time

def with_retry(func=None, *, max_attempts=8):
    """Decorator that adds retry logic to a function.

    Args:
        func: The function to decorate
        max_attempts: Maximum number of retry attempts (default: 8)

    Usage:
        @with_retry
        def my_function():
            pass

        # Or with custom max_attempts:
        @with_retry(max_attempts=5)
        def my_function():
            pass
    """
    if func is None:
        return lambda f: retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_strategy
        )(f)

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_strategy
    )(func)
