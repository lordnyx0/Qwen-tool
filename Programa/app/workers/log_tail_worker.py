"""log_tail_worker.py — Monitoramento e streaming dinâmico de arquivos de log sem travamento."""
import os
import time
from typing import Optional

from PySide6.QtCore import QThread, Signal


class LogTailWorker(QThread):
    """Monitora continuamente um arquivo de log e emite novos dados assincronamente."""

    lines_appended = Signal(str)
    tail_updated = Signal(str)
    stopped = Signal()

    def __init__(self, file_path: str, poll_interval: float = 1.0, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.poll_interval = poll_interval
        self._running = True

    def run(self):
        last_size = 0
        while self._running:
            try:
                if os.path.isfile(self.file_path):
                    current_size = os.path.getsize(self.file_path)
                    if current_size > last_size:
                        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(last_size)
                            new_text = f.read()
                            last_size = f.tell()
                            if new_text:
                                self.lines_appended.emit(new_text)

                        # Emite também o bloco final formatado (últimas 30 linhas)
                        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                            self.tail_updated.emit("".join(lines[-30:]))
                    elif current_size < last_size:
                        # Arquivo foi truncado / reiniciado
                        last_size = 0
            except Exception:
                pass

            time.sleep(self.poll_interval)

        self.stopped.emit()

    def stop(self):
        self._running = False
