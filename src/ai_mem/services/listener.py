"""Listener Service - Manage observation listeners/callbacks.

This service handles registration and notification of listeners
that want to be notified when observations are added.
"""

import threading
from typing import Callable, List, TYPE_CHECKING

from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..models import Observation

logger = get_logger("services.listener")

ObservationListener = Callable[["Observation"], None]


class ListenerService:
    """Manages observation listeners.

    Thread-safe service for registering and notifying callbacks
    when observations are created.

    Usage:
        service = ListenerService()
        service.add_listener(my_callback)
        service.notify(observation)
    """

    def __init__(self):
        """Initialize listener service."""
        self._listeners: List[ObservationListener] = []
        self._lock = threading.Lock()

    def add_listener(self, listener: ObservationListener) -> None:
        """Register a new listener.

        Args:
            listener: Callback function to be called with each observation
        """
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)
                logger.debug(f"Added listener: {listener.__name__ if hasattr(listener, '__name__') else listener}")

    def remove_listener(self, listener: ObservationListener) -> bool:
        """Remove a registered listener.

        Args:
            listener: Callback function to remove

        Returns:
            True if listener was found and removed
        """
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
                logger.debug(f"Removed listener: {listener.__name__ if hasattr(listener, '__name__') else listener}")
                return True
            return False

    def notify(self, observation: "Observation") -> None:
        """Notify all listeners of a new observation.

        Args:
            observation: The observation to broadcast
        """
        with self._lock:
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(observation)
            except Exception as e:
                # Log but don't propagate - one bad listener shouldn't break others
                logger.warning(f"Listener error: {type(e).__name__}: {e}")

    def clear(self) -> None:
        """Remove all listeners."""
        with self._lock:
            count = len(self._listeners)
            self._listeners.clear()
            logger.debug(f"Cleared {count} listeners")

    @property
    def count(self) -> int:
        """Get number of registered listeners."""
        with self._lock:
            return len(self._listeners)
