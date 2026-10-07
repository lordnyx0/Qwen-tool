"""process_manager.py — Rastreamento e ciclo de vida de processos em segundo plano (multiplataforma)."""
import os
import signal
import subprocess
import sys
from typing import Dict, Optional, Tuple


def launch_detached_process(cmd: str, log_file: str, cwd: str) -> Dict[str, Optional[int]]:
    """Lança processo em segundo plano (detached), capturando PID e redirecionando logs sem travar o app."""
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    try:
        log_f = open(log_file, "a", encoding="utf-8", errors="replace")

        creationflags = 0
        preexec_fn = None

        if sys.platform == "win32":
            # DETACHED_PROCESS (0x00000008) | CREATE_NEW_PROCESS_GROUP (0x00000200)
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008
            close_fds = True
        else:
            # POSIX setsid
            preexec_fn = os.setsid
            close_fds = True

        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
            close_fds=close_fds,
        )
        return {"return_value": 0, "pid": proc.pid, "error": None}
    except Exception as e:
        return {"return_value": -1, "pid": None, "error": str(e)}


# Alias para retrocompatibilidade
launch_wmi_detached = launch_detached_process


def is_pid_running(pid: Optional[int]) -> bool:
    """Verifica se um determinado PID ainda está em execução no sistema operacional."""
    if not pid or pid <= 0:
        return False

    if sys.platform == "win32":
        try:
            import ctypes
            import ctypes.wintypes
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
            if not handle:
                return False
            exit_code = ctypes.wintypes.DWORD()
            res = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            # STILL_ACTIVE = 259
            return res != 0 and exit_code.value == 259
        except Exception:
            # Fallback seguro para tasklist caso ctypes encontre restrição
            try:
                r = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return str(pid) in r.stdout
            except Exception:
                return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def kill_process_tree(pid: Optional[int]) -> Tuple[bool, str]:
    """Encerra uma árvore de processos de forma limpa e multiplataforma."""
    if not pid or pid <= 0:
        return False, "PID inválido ou inexistente."

    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            success = r.returncode == 0
            msg = (r.stdout + r.stderr).strip()
            return success, msg
        except Exception as e:
            return False, str(e)
    else:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            return True, f"Grupo de processos {pgid} encerrado."
        except Exception as e:
            try:
                os.kill(pid, signal.SIGKILL)
                return True, f"Processo PID {pid} encerrado."
            except Exception as e2:
                return False, f"Falha ao encerrar PID {pid}: {e2}"

