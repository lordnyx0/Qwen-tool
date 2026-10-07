"""Estado (aba 1): dataset, modelos, GPU, servidores. So leitura."""
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QTextEdit, QVBoxLayout, QWidget)

from app.core import repo_state


class EstadoTab(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn = QPushButton("Atualizar (lê do repo na hora)")
        self.btn.clicked.connect(self.refresh)
        top.addWidget(self.btn)
        top.addStretch(1)
        lay.addLayout(top)
        self.boxes = {}
        for title in ("Dataset", "Checkpoints / GGUF", "GPU + Servidores",
                      "Kaggle (sob demanda)"):
            gb = QGroupBox(title)
            txt = QTextEdit()
            txt.setReadOnly(True)
            txt.setFontFamily("Consolas")
            QVBoxLayout(gb).addWidget(txt)
            lay.addWidget(gb)
            self.boxes[title] = txt
        self.refresh(light=True)

    def refresh(self, light=False):
        try:
            s = repo_state.summary()
        except Exception as e:
            self.boxes["Dataset"].setText(f"ERRO lendo repo: {e}\n"
                                          f"ROOT={repo_state.ROOT}\n"
                                          "Defina QWEN_ROOT se necessário.")
            return
        try:
            self.boxes["Dataset"].setText(
                f"ROOT={s['root']}\nPURIFIED N = {s['purified_n']}\nSHIP: v38_ep2 "
                f"(Q8 em models/gguf/)")
        except Exception as e:
            self.boxes["Dataset"].setText(f"ERRO: {e}")
        try:
            ck = "\n".join(f"{c['name']} ({c['mb']}MB)"
                           for c in s["checkpoints"]) or "(vazio)"
            gg = "\n".join(f"{g['name']} ({g['mb']}MB)" for g in s["gguf"]) or \
                "(vazio)"
            self.boxes["Checkpoints / GGUF"].setText(
                f"[checkpoints]\n{ck}\n\n[gguf]\n{gg}")
        except Exception as e:
            self.boxes["Checkpoints / GGUF"].setText(f"ERRO: {e}")
        try:
            v = s["vram"]
            vram_s = (f"{v['used_mb']}/{v['total_mb']} MB"
                      if "used_mb" in v else v.get("error"))
            srv = "\n".join(f":{x['port']} {x['status']}" for x in s["servers"]) \
                or "nenhum"
            self.boxes["GPU + Servidores"].setText(
                f"VRAM: {vram_s}\nServidores:\n{srv}")
        except Exception as e:
            self.boxes["GPU + Servidores"].setText(f"ERRO: {e}")
        if light:
            return
        try:
            ks = repo_state.kaggle_sessions()
            if isinstance(ks, dict):
                self.boxes["Kaggle (sob demanda)"].setText(str(ks))
            else:
                self.boxes["Kaggle (sob demanda)"].setText(
                    "\n".join(f"{k['kernel'].split('/')[-1]}: "
                              f"{k.get('status', k.get('error'))}"
                              for k in ks))
        except Exception as e:
            self.boxes["Kaggle (sob demanda)"].setText(f"ERRO: {e}")
