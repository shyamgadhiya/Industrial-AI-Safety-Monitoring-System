"""
Asynchronous queue manager for pipeline decoupled execution.
Provides bounded queues with configurable overflow/backpressure drop policies.
"""

from __future__ import annotations
import queue
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class BoundedPipelineQueue:
    """
    Thread-safe bounded queue that drops the oldest packet on overflow
    to prevent latency build-up and video lag during compute spikes.
    """

    def __init__(self, maxsize: int = 30, drop_oldest_on_full: bool = True):
        self.maxsize = maxsize
        self.drop_oldest_on_full = drop_oldest_on_full
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped_count: int = 0

    def put(self, item: Any, block: bool = False, timeout: Optional[float] = None) -> bool:
        """
        Put an item into the queue.
        If queue is full and drop_oldest_on_full is True, evicts the oldest item.
        """
        try:
            self._queue.put(item, block=block, timeout=timeout)
            return True
        except queue.Full:
            if self.drop_oldest_on_full:
                try:
                    # Drop oldest item
                    _ = self._queue.get_nowait()
                    self._queue.task_done()
                    self.dropped_count += 1
                    self._queue.put_nowait(item)
                    return True
                except (queue.Empty, queue.Full):
                    return False
            return False

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        return self._queue.get(block=block, timeout=timeout)

    def task_done(self) -> None:
        self._queue.task_done()

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()
