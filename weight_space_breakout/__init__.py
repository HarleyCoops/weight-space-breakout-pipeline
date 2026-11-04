"""Independent Breakout rules; not the original paper's environment."""
__version__ = "0.4.3"
from .game import Breakout
from .policy import Policy

__all__ = ["Breakout", "Policy"]

# Verifiers discovers exported Taskset classes. Core simulation also works
# without installing the optional evaluation framework.
import importlib.util
import sys
if sys.platform != "win32" and importlib.util.find_spec("verifiers") is not None:
    from .taskset import BreakoutTaskset
    __all__.append("BreakoutTaskset")
