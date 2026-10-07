"""Treinos (aba 2): fila local em segundo plano + status Kaggle + tail de log em tempo real."""
import glob
import os

from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout,
                               QLabel, QPushButton, QTextEdit, QVBoxLayout,
                               QWidget)

from app.core import config, paths, repo_state
from app.workers import (LogTailWorker, is_pid_running, kill_process_tree,
                         launch_detached_process)

ROOT = repo_state.ROOT
LOGDIR = paths.get_log_dir()

DEFAULT_JOBS = {
    "DPO smoke (8 pares, v38)": (
        "python scripts/train/train_dpo_tool.py --pairs "
        "{tmp}/smoke_dpo_r2_chosen.jsonl "
        "--rejected {tmp}/smoke_dpo_r2_rejected.jsonl "
        "--model_path {v38} --output_dir {tmp}/smoke_gui "
        "--lr 5e-7 --beta 0.1 --epochs 1",
        "smoke_gui.log"),
    "Validação Dataset PURIFIED (Gate)": (
        "python scripts/pipeline/validate_sft.py data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl",
        "validate_sft_gui.log"),
    "Validação Documentação (Gate)": (
        "python scripts/pipeline/validate_docs.py",
        "validate_docs_gui.log"),
}


def get_jobs():
    """Carrega os jobs definidos no workbench.json ou usa os padrões do projeto."""
    cfg = config.get_project_config()
    cfg_jobs = cfg.get("jobs")
    if cfg_jobs and isinstance(cfg_jobs, dict):
        out = {}
        for name, info in cfg_jobs.items():
            if isinstance(info, (list, tuple)) and len(info) >= 2:
                out[name] = (info[0], info[1])
            elif isinstance(info, dict):
                out[name] = (info.get("cmd", ""), info.get("log", f"{name}.log"))
        if out:
            return out
    return DEFAULT_JOBS


def _v38():
    snap = paths.find_hf_model_snapshot("Lordnyx/qwen3loop-0.6b-tool-calling-sft-v38")
    if snap:
        return snap
    # Fallback caso não esteja no cache HuggingFace
    return os.path.join(ROOT, "models", "qwen3loop-0.6b-tool-calling-sft-v38")


class TreinosTab(QWidget):
    def __init__(self):
        super().__init__()
        self.current_pid = None
        self.tail_worker = None
        self.jobs = get_jobs()

        lay = QVBoxLayout(self)
        gb = QGroupBox("Lançar treino/processo local (em segundo plano com PID)")
        hl = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.addItems(list(self.jobs.keys()))
        hl.addWidget(self.combo, 3)
        self.btn_launch = QPushButton("Lançar")
        self.btn_launch.clicked.connect(self.launch)
        hl.addWidget(self.btn_launch)

        self.btn_kill = QPushButton("Parar Processo")
        self.btn_kill.setEnabled(False)
        self.btn_kill.clicked.connect(self.stop_process)
        hl.addWidget(self.btn_kill)

        QVBoxLayout(gb).addLayout(hl)

        self.lbl_process_info = QLabel("Status: Nenhum processo local ativo gerenciado.")
        QVBoxLayout(gb).addWidget(self.lbl_process_info)
        lay.addWidget(gb)

        gb2 = QGroupBox("Tail de log dinâmico (logs/*.log)")
        hl2 = QHBoxLayout()
        self.logcombo = QComboBox()
        self.refresh_logs()
        hl2.addWidget(self.logcombo, 3)

        self.chk_autotail = QCheckBox("Seguir em tempo real")
        self.chk_autotail.toggled.connect(self.toggle_autotail)
        hl2.addWidget(self.chk_autotail)

        b2 = QPushButton("Últimas 30 linhas")
        b2.clicked.connect(self.tail_manual)
        hl2.addWidget(b2)
        br = QPushButton("Recarregar lista")
        br.clicked.connect(self.refresh_logs)
        hl2.addWidget(br)
        QVBoxLayout(gb2).addLayout(hl2)
        lay.addWidget(gb2)

        self.out = QTextEdit()
        self.out.setReadOnly(True)
        self.out.setFontFamily("Consolas")
        lay.addWidget(self.out, 1)

    def refresh_logs(self):
        cur = self.logcombo.currentText()
        files = sorted(glob.glob(os.path.join(LOGDIR, "*.log")),
                       key=os.path.getmtime, reverse=True)
        self.logcombo.clear()
        self.logcombo.addItems([os.path.basename(f) for f in files]
                               or ["(sem logs)"])
        if cur:
            i = self.logcombo.findText(cur)
            if i >= 0:
                self.logcombo.setCurrentIndex(i)

    def launch(self):
        name = self.combo.currentText()
        if name not in self.jobs:
            return
        cmd, log = self.jobs[name]
        cmd = cmd.format(v38=_v38(), tmp=LOGDIR.replace("\\", "/"), root=ROOT.replace("\\", "/"))
        log_fp = os.path.join(LOGDIR, log)

        res = launch_detached_process(cmd, log_fp, cwd=ROOT)
        if res.get("return_value") == 0 and res.get("pid"):
            self.current_pid = res["pid"]
            self.lbl_process_info.setText(
                f"Lançado: {name} | PID: {self.current_pid} (Em execução) | Log: {log}")
            self.btn_kill.setEnabled(True)
            self.refresh_logs()
            idx = self.logcombo.findText(log)
            if idx >= 0:
                self.logcombo.setCurrentIndex(idx)
            self.out.setText(f"[INICIADO] {name}\nPID: {self.current_pid}\nArquivo: {log_fp}\n")
            if not self.chk_autotail.isChecked():
                self.chk_autotail.setChecked(True)
        else:
            err = res.get("error") or f"código {res.get('return_value')}"
            self.lbl_process_info.setText(f"Falha ao lançar: {err}")
            self.out.setText(f"[ERRO AO INICIAR]\n{err}")

    def stop_process(self):
        if self.current_pid:
            success, msg = kill_process_tree(self.current_pid)
            if success:
                self.lbl_process_info.setText(
                    f"Processo PID {self.current_pid} encerrado com sucesso.")
                self.out.append(f"\n[INTERROMPIDO] PID {self.current_pid} encerrado.")
            else:
                self.lbl_process_info.setText(f"Erro ao encerrar PID {self.current_pid}: {msg}")
                self.out.append(f"\n[FALHA AO ENCERRAR] {msg}")
            self.btn_kill.setEnabled(False)
            self.current_pid = None

    def toggle_autotail(self, checked: bool):
        if checked:
            fp = os.path.join(LOGDIR, self.logcombo.currentText())
            if os.path.isfile(fp):
                if self.tail_worker:
                    self.tail_worker.stop()
                    self.tail_worker.wait()
                self.tail_worker = LogTailWorker(fp, poll_interval=1.0)
                self.tail_worker.tail_updated.connect(self.on_tail_updated)
                self.tail_worker.start()
        else:
            if self.tail_worker:
                self.tail_worker.stop()
                self.tail_worker = None

    def on_tail_updated(self, text: str):
        self.out.setText(text)
        sb = self.out.verticalScrollBar()
        sb.setValue(sb.maximum())

        # Atualiza status do PID se houver processo ativo
        if self.current_pid:
            if not is_pid_running(self.current_pid):
                self.lbl_process_info.setText(
                    f"Processo PID {self.current_pid} concluiu a execução.")
                self.btn_kill.setEnabled(False)
                self.current_pid = None

    def tail_manual(self):
        fp = os.path.join(LOGDIR, self.logcombo.currentText())
        try:
            with open(fp, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            self.out.setText("".join(lines[-30:]))
        except FileNotFoundError:
            self.out.setText(f"Sem log: {fp}")
