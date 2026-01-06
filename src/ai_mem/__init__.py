from importlib.metadata import PackageNotFoundError, version

from .config import load_config, save_config
from .memory import MemoryManager

try:
    __version__ = version("ai-mem")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["MemoryManager", "load_config", "save_config", "__version__"]
