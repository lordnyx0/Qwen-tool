"""playground_worker.py — Envio assíncrono de prompt de teste para router llama-server (:8080)."""
import json
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QThread, Signal


def extract_tool_calls(text: str) -> Tuple[List[Dict[str, Any]], bool, bool]:
    """Extrai tool_calls do texto e valida conformidade XML e JSON."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    has_tag = "<tool_call>" in text
    calls = []
    all_json_valid = True

    if not matches and has_tag:
        after = text.split("<tool_call>", 1)[1]
        matches = [after.split("</tool_call>", 1)[0]]

    for m in matches:
        s = m.strip()
        try:
            parsed = json.loads(s)
            calls.append(parsed)
        except Exception:
            all_json_valid = False
            calls.append({"raw_error": s})

    return calls, has_tag, all_json_valid


class PlaygroundWorker(QThread):
    """Envia requisição para llama-server local e analisa saída de tool calling."""

    result_ready = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant equipped with specialized tools. Use them when needed.",
        base_url: str = "http://127.0.0.1:8080",
        parent=None,
    ):
        super().__init__(parent)
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.base_url = base_url.rstrip("/")

    def run(self):
        t0 = time.time()
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_bytes = resp.read()
                elapsed = time.time() - t0
                body = json.loads(raw_bytes.decode("utf-8"))
                content = ""
                if "choices" in body and body["choices"]:
                    msg = body["choices"][0].get("message", {})
                    content = msg.get("content", "")

                calls, has_tag, json_valid = extract_tool_calls(content)

                res = {
                    "raw_text": content,
                    "has_tag": has_tag,
                    "tool_calls": calls,
                    "json_valid": json_valid,
                    "elapsed_s": round(elapsed, 2),
                    "error": None,
                }
                self.result_ready.emit(res)
        except urllib.error.URLError as e:
            err_msg = (
                f"Servidor inacessível em {self.base_url} ({e.reason}).\n"
                "Certifique-se de que o llama-server está ativo (ex: serve_qwen3loop_pi.bat)."
            )
            self.error.emit(err_msg)
        except Exception as e:
            self.error.emit(f"Erro na requisição ao servidor: {e}")
