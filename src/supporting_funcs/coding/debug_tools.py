"""A simple set of tools for debugging and profiling code.

Functions
tic
    Start a timer with the given ID.
toc
    Return the time in seconds since the last call to tic() with the same timer_id.
toc_tic
    Convenience function to call toc() and then immediately restart the timer with tic().
"""
from time import time_ns

_TIME_START_DICT: dict[str, int] = {'None': -1}
def tic(timer_id: str = 'None'):
    """Start the timer with the given ID."""
    _TIME_START_DICT[timer_id] = time_ns()

def toc(timer_id: str = 'None') -> float:
    """Return the time in seconds since the last call to tic() with the same timer_id."""
    try:
        _time_start = _TIME_START_DICT[timer_id]
    except KeyError:
        print(f"Error: toc() called for unknown timer '{timer_id}'. " \
              "Calling tic() with the same ID now")
        tic(timer_id=timer_id)
        return -1.
    return (time_ns() - _time_start) * 1e-9

def toc_tic(timer_id: str = 'None') -> float:
    """Call toc() and then immediately restart the timer with tic()."""
    toc_time = toc(timer_id=timer_id)
    tic(timer_id=timer_id)
    return toc_time
