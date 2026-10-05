"""Run long work (training, benchmarks, downloads) off the UI thread, with progress."""

from __future__ import annotations

import traceback
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class _Worker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def run(self) -> None:
        try:
            result = self.fn(lambda done, total, message: self.progress.emit(done, total, message))
            self.finished.emit(result)
        except Exception as error:  # noqa: BLE001 - reported to the page, never fatal
            traceback.print_exc()
            self.failed.emit(f"{type(error).__name__}: {error}")


class Job(QObject):
    """One background job. ``fn(progress)`` runs on a worker thread; signals come back on the UI thread.

        job = Job(lambda progress: work(progress))
        job.progress.connect(...); job.finished.connect(...); job.failed.connect(...)
        job.start()
    """

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.thread = QThread()
        self.worker = _Worker(fn)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress)
        self.worker.finished.connect(self._finish_ok)
        self.worker.failed.connect(self._finish_fail)
        self.running = False

    def start(self) -> None:
        self.running = True
        self.thread.start()

    def _stop(self) -> None:
        self.thread.quit()
        self.thread.wait()
        self.running = False

    def _finish_ok(self, result) -> None:
        self._stop()
        self.finished.emit(result)

    def _finish_fail(self, message: str) -> None:
        self._stop()
        self.failed.emit(message)
