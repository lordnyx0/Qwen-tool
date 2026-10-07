"""command_worker.py — Execução assíncrona de comandos sem travar a interface Qt."""
import os
import subprocess
from typing import List, Optional

from PySide6.QtCore import QThread, Signal


class CommandWorker(QThread):
    """Executa um comando de terminal em subprocesso assíncrono via QThread."""

    output_line = Signal(str)
    finished = Signal(int, str)
    error = Signal(str)

    def __init__(
        self,
        cmd: List[str] | str,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.cmd = cmd
        self.cwd = cwd
        self.env = env or os.environ.copy()
        self.process: Optional[subprocess.Popen] = None
        self._is_killed = False

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as e:
            self.error.emit(f"Falha ao iniciar processo: {e}")
            self.finished.emit(-1, str(e))
            return

        lines: List[str] = []
        try:
            if self.process.stdout:
                for line in iter(self.process.stdout.readline, ""):
                    if self._is_killed:
                        break
                    lines.append(line)
                    self.output_line.emit(line)
                self.process.stdout.close()

            ret = self.process.wait()
            if not self._is_killed:
                self.finished.emit(ret, "".join(lines))
        except Exception as e:
            self.error.emit(f"Erro durante execução: {e}")
            self.finished.emit(-1, "".join(lines) + f"\n[ERRO] {e}")

    def stop(self):
        """Interrompe o processo em execução."""
        self._is_killed = True
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
