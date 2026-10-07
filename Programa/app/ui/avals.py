"""Evals (aba 3): Baterias/Sondas, histórico, comparador de regressão e playground interativo."""
import glob
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QTabWidget,
                               QTextEdit, QVBoxLayout, QWidget)

from app.core import config, paths, repo_state
from app.workers import (LogTailWorker, PlaygroundWorker, is_pid_running,
                         kill_process_tree, launch_detached_process)

ROOT = repo_state.ROOT
LOGDIR = paths.get_log_dir()

DEFAULT_PRESETS = {
    "Sonda Canônica 25 (v38 Q8, Rápida ~1-2 min)": (
        "python scripts/eval/benchmark_battery_v2.py "
        "--battery_path data/eval_probe_canonical_25.jsonl "
        "--model_path models/gguf/unrolled_modelo_qwen3loop_tool_sft_v38_ep2_q8_0.gguf "
        "--tokenizer_dir Kaggle/kaggle_kernel_output/sft_v33/qwen3loop_tool_curriculum_3000_final/epoch-2 "
        "--output_file {output_dir}/probe25_gui.json "
        "--n_gpu_layers -1",
        "probe25_gui.log"),
    "Bateria 121 (v38 Q8, Completa)": (
        "python scripts/eval/benchmark_battery_v2.py "
        "--model_path models/gguf/unrolled_modelo_qwen3loop_tool_sft_v38_ep2_q8_0.gguf "
        "--tokenizer_dir Kaggle/kaggle_kernel_output/sft_v33/qwen3loop_tool_curriculum_3000_final/epoch-2 "
        "--output_file {output_dir}/batt_gui.json "
        "--n_gpu_layers -1",
        "batt_gui.log"),
    "Holdout 90 Baseline (v38 Q8)": (
        "python scripts/eval/benchmark_tool_calling_baseline.py "
        "--model_path models/gguf/unrolled_modelo_qwen3loop_tool_sft_v38_ep2_q8_0.gguf "
        "--tokenizer_dir Kaggle/kaggle_kernel_output/sft_v33/qwen3loop_tool_curriculum_3000_final/epoch-2",
        "holdout90_gui.log"),
}


def get_eval_presets():
    """Carrega os presets definidos no workbench.json ou usa os padrões do projeto."""
    cfg = config.get_project_config()
    cfg_evals = cfg.get("evals")
    if cfg_evals and isinstance(cfg_evals, dict):
        out = {}
        for name, info in cfg_evals.items():
            if isinstance(info, (list, tuple)) and len(info) >= 2:
                out[name] = (info[0], info[1])
            elif isinstance(info, dict):
                out[name] = (info.get("cmd", ""), info.get("log", f"{name}.log"))
        if out:
            return out
    return DEFAULT_PRESETS


class EvalsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.current_pid: Optional[int] = None
        self.tail_worker: Optional[LogTailWorker] = None
        self.play_worker: Optional[PlaygroundWorker] = None
        self.presets = get_eval_presets()

        main_layout = QVBoxLayout(self)
        self.subtabs = QTabWidget()
        main_layout.addWidget(self.subtabs)

        # 1. Sub-aba Execução
        self.tab_run = QWidget()
        self._build_run_tab()
        self.subtabs.addTab(self.tab_run, "1. Executar Baterias & Sondas")

        # 2. Sub-aba Comparador e Histórico
        self.tab_hist = QWidget()
        self._build_hist_tab()
        self.subtabs.addTab(self.tab_hist, "2. Histórico & Comparador (Diff)")

        # 3. Sub-aba Playground Tool-Calling
        self.tab_play = QWidget()
        self._build_playground_tab()
        self.subtabs.addTab(self.tab_play, "3. Playground Tool-Calling (:8080)")

    # -------------------------------------------------------------------------
    # Sub-aba 1: Execução com PID e Auto-tail
    # -------------------------------------------------------------------------
    def _build_run_tab(self):
        lay = QVBoxLayout(self.tab_run)
        gb = QGroupBox("Executar avaliação (llama-cpp-python, aceleração GPU/CPU)")
        hl = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.addItems(list(self.presets.keys()))
        hl.addWidget(self.combo, 3)

        self.btn_run = QPushButton("Rodar detached")
        self.btn_run.clicked.connect(self.run_eval)
        hl.addWidget(self.btn_run)

        self.btn_kill = QPushButton("Parar Eval")
        self.btn_kill.setEnabled(False)
        self.btn_kill.clicked.connect(self.stop_eval)
        hl.addWidget(self.btn_kill)

        QVBoxLayout(gb).addLayout(hl)
        self.lbl_eval_status = QLabel("Status: Nenhuma avaliação em andamento.")
        QVBoxLayout(gb).addWidget(self.lbl_eval_status)
        lay.addWidget(gb)

        gb_log = QGroupBox("Monitor de Log em Tempo Real")
        hl_log = QHBoxLayout()
        self.chk_autotail = QCheckBox("Seguir log automaticamente")
        self.chk_autotail.toggled.connect(self.toggle_autotail)
        hl_log.addWidget(self.chk_autotail)
        hl_log.addStretch(1)
        b_tail = QPushButton("Recarregar Log Manual")
        b_tail.clicked.connect(self.tail_manual)
        hl_log.addWidget(b_tail)
        QVBoxLayout(gb_log).addLayout(hl_log)

        self.out_run = QTextEdit()
        self.out_run.setReadOnly(True)
        self.out_run.setFontFamily("Consolas")
        QVBoxLayout(gb_log).addWidget(self.out_run)
        lay.addWidget(gb_log, 1)

    def run_eval(self):
        name = self.combo.currentText()
        if name not in self.presets:
            return
        cmd_template, log = self.presets[name]
        cmd = cmd_template.format(output_dir=LOGDIR.replace("\\", "/"))
        log_fp = os.path.join(LOGDIR, log)

        res = launch_detached_process(cmd, log_fp, cwd=ROOT)
        if res.get("return_value") == 0 and res.get("pid"):
            self.current_pid = res["pid"]
            self.lbl_eval_status.setText(
                f"Avaliação rodando: {name} | PID: {self.current_pid} | Log: {log}")
            self.btn_kill.setEnabled(True)
            self.out_run.setText(f"[INICIADO] {name}\nPID: {self.current_pid}\nLog: {log_fp}\n")
            self.chk_autotail.setChecked(True)
        else:
            err = res.get("error") or f"código {res.get('return_value')}"
            self.lbl_eval_status.setText(f"Erro ao disparar eval: {err}")
            self.out_run.setText(f"[FALHA]\n{err}")

    def stop_eval(self):
        if self.current_pid:
            ok, msg = kill_process_tree(self.current_pid)
            if ok:
                self.lbl_eval_status.setText(f"Processo {self.current_pid} interrompido.")
                self.out_run.append(f"\n[INTERROMPIDO] PID {self.current_pid} encerrado.")
            else:
                self.lbl_eval_status.setText(f"Erro ao interromper: {msg}")
            self.btn_kill.setEnabled(False)
            self.current_pid = None

    def toggle_autotail(self, checked: bool):
        if checked:
            name = self.combo.currentText()
            if name not in self.presets:
                return
            _, log = self.presets[name]
            log_fp = os.path.join(LOGDIR, log)
            if self.tail_worker:
                self.tail_worker.stop()
                self.tail_worker.wait()
            self.tail_worker = LogTailWorker(log_fp, poll_interval=1.0)
            self.tail_worker.tail_updated.connect(self.on_tail_updated)
            self.tail_worker.start()
        else:
            if self.tail_worker:
                self.tail_worker.stop()
                self.tail_worker = None

    def on_tail_updated(self, text: str):
        self.out_run.setText(text)
        sb = self.out_run.verticalScrollBar()
        sb.setValue(sb.maximum())
        if self.current_pid and not is_pid_running(self.current_pid):
            self.lbl_eval_status.setText(
                f"Avaliação PID {self.current_pid} finalizada.")
            self.btn_kill.setEnabled(False)
            self.current_pid = None
            self.refresh_hist_files()

    def tail_manual(self):
        name = self.combo.currentText()
        if name not in self.presets:
            return
        _, log = self.presets[name]
        log_fp = os.path.join(LOGDIR, log)
        if os.path.isfile(log_fp):
            with open(log_fp, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            self.out_run.setText("".join(lines[-40:]))
        else:
            self.out_run.setText(f"Arquivo de log não encontrado: {log_fp}")

    # -------------------------------------------------------------------------
    # Sub-aba 2: Histórico e Comparador (Diff de Regressões)
    # -------------------------------------------------------------------------
    def _build_hist_tab(self):
        lay = QVBoxLayout(self.tab_hist)

        # Seletor Individual
        gb_single = QGroupBox("Inspecionar Resultado Único")
        hl_s = QHBoxLayout()
        self.hist_single = QComboBox()
        hl_s.addWidget(self.hist_single, 3)
        b_view = QPushButton("Ver Resumo")
        b_view.clicked.connect(self.show_single)
        hl_s.addWidget(b_view)
        b_reload = QPushButton("Atualizar Arquivos")
        b_reload.clicked.connect(self.refresh_hist_files)
        hl_s.addWidget(b_reload)
        QVBoxLayout(gb_single).addLayout(hl_s)
        lay.addWidget(gb_single)

        # Comparador Diff
        gb_diff = QGroupBox("Comparador de Runs (Diff de Acurácia & Regressões)")
        gl_diff = QVBoxLayout(gb_diff)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Run A (Baseline):"))
        self.combo_run_a = QComboBox()
        row1.addWidget(self.combo_run_a, 3)
        gl_diff.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Run B (Novo teste):"))
        self.combo_run_b = QComboBox()
        row2.addWidget(self.combo_run_b, 3)
        b_diff = QPushButton("Comparar Runs (Diff)")
        b_diff.clicked.connect(self.compare_runs)
        row2.addWidget(b_diff)
        gl_diff.addLayout(row2)
        lay.addWidget(gb_diff)

        self.out_hist = QTextEdit()
        self.out_hist.setReadOnly(True)
        self.out_hist.setFontFamily("Consolas")
        lay.addWidget(self.out_hist, 1)

        self.refresh_hist_files()

    def refresh_hist_files(self):
        files = sorted(glob.glob(os.path.join(ROOT, "data", "eval_results", "*", "*.json")))
        files += sorted(glob.glob(os.path.join(ROOT, "data", "eval_results", "*.json")))
        files += sorted(glob.glob(os.path.join(LOGDIR, "*.json")))

        self.hist_single.clear()
        self.combo_run_a.clear()
        self.combo_run_b.clear()

        items = files or ["(nenhum)"]
        self.hist_single.addItems(items)
        self.combo_run_a.addItems(items)
        self.combo_run_b.addItems(items)
        if len(items) > 1:
            self.combo_run_b.setCurrentIndex(1)

    def show_single(self):
        fp = self.hist_single.currentText()
        if not os.path.isfile(fp):
            self.out_hist.setText(f"Arquivo não existe: {fp}")
            return
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.out_hist.setText(self._format_summary(d, os.path.basename(fp)))
        except Exception as e:
            self.out_hist.setText(f"Erro lendo {fp}: {e}")

    def _format_summary(self, d: Dict[str, Any], title: str) -> str:
        if not isinstance(d, dict) or "total_passed" not in d:
            return json.dumps(d, indent=2)[:3000]

        tot = d.get("total_samples", 0)
        passed = d.get("total_passed", 0)
        acc = d.get("accuracy_pct", round(passed / tot * 100, 1) if tot else 0)
        fmt = d.get("format_pct", "?")
        undecl = d.get("undeclared_count", "?")

        lines = [
            f"=== {title} ===",
            f"Modelo: {d.get('model', 'N/A')}",
            f"Acurácia: {passed}/{tot} ({acc:.1f}%) | Formato: {fmt}% | Undeclared: {undecl}",
            "",
            "--- Categorias ---",
        ]
        cats = d.get("categories", {})
        if isinstance(cats, dict):
            for k, v in cats.items():
                if isinstance(v, dict):
                    c_pass = v.get("pass", v.get("passed", v.get("correct", 0)))
                    c_tot = v.get("total", 0)
                    pct = round(c_pass / c_tot * 100, 1) if c_tot else 0
                    lines.append(f"  • {k}: {c_pass}/{c_tot} ({pct}%)")
                else:
                    lines.append(f"  • {k}: {v}")

        det = d.get("details", [])
        if det:
            fails = [x.get("id", "?") for x in det if not x.get("overall_pass")]
            lines.append("")
            lines.append(f"--- Falhas ({len(fails)} itens) ---")
            lines.append(", ".join(fails[:30]) + ("..." if len(fails) > 30 else ""))
        return "\n".join(lines)

    def compare_runs(self):
        fp_a = self.combo_run_a.currentText()
        fp_b = self.combo_run_b.currentText()
        if not os.path.isfile(fp_a) or not os.path.isfile(fp_b):
            self.out_hist.setText("Selecione dois arquivos válidos para comparar.")
            return

        try:
            with open(fp_a, "r", encoding="utf-8") as f:
                da = json.load(f)
            with open(fp_b, "r", encoding="utf-8") as f:
                db = json.load(f)

            diff_text = self._compute_diff(da, db, os.path.basename(fp_a), os.path.basename(fp_b))
            self.out_hist.setText(diff_text)
        except Exception as e:
            self.out_hist.setText(f"Erro comparando runs: {e}")

    def _compute_diff(self, a: Dict[str, Any], b: Dict[str, Any], name_a: str, name_b: str) -> str:
        tot_a, pass_a = a.get("total_samples", 0), a.get("total_passed", 0)
        tot_b, pass_b = b.get("total_samples", 0), b.get("total_passed", 0)
        acc_a = a.get("accuracy_pct", (pass_a / tot_a * 100) if tot_a else 0)
        acc_b = b.get("accuracy_pct", (pass_b / tot_b * 100) if tot_b else 0)
        fmt_a = a.get("format_pct", 0)
        fmt_b = b.get("format_pct", 0)
        und_a = a.get("undeclared_count", 0)
        und_b = b.get("undeclared_count", 0)

        delta_acc = acc_b - acc_a
        delta_pass = pass_b - pass_a
        delta_fmt = fmt_b - fmt_a
        delta_und = und_b - und_a

        lines = [
            f"=== DIFF: [A] {name_a}  VS  [B] {name_b} ===",
            f"Acurácia:  A: {pass_a}/{tot_a} ({acc_a:.1f}%)  ->  B: {pass_b}/{tot_b} ({acc_b:.1f}%)  [Δ {delta_acc:+.1f}% | {delta_pass:+d} acertos]",
            f"Formato:   A: {fmt_a}%  ->  B: {fmt_b}%  [Δ {delta_fmt:+.1f}%]",
            f"Undeclared: A: {und_a}  ->  B: {und_b}  [Δ {delta_und:+d}]",
            "",
            "--- Comparativo por Categoria ---",
        ]

        cats_a = a.get("categories", {})
        cats_b = b.get("categories", {})
        all_cats = sorted(set(cats_a.keys()) | set(cats_b.keys()))

        for cat in all_cats:
            va = cats_a.get(cat, {})
            vb = cats_b.get(cat, {})
            pa = va.get("pass", va.get("passed", 0)) if isinstance(va, dict) else 0
            ta = va.get("total", 0) if isinstance(va, dict) else 0
            pb = vb.get("pass", vb.get("passed", 0)) if isinstance(vb, dict) else 0
            tb = vb.get("total", 0) if isinstance(vb, dict) else 0
            pct_a = (pa / ta * 100) if ta else 0
            pct_b = (pb / tb * 100) if tb else 0
            lines.append(f"  • {cat}: {pct_a:.1f}% ({pa}/{ta}) -> {pct_b:.1f}% ({pb}/{tb})  [Δ {pct_b - pct_a:+.1f}%]")

        # Mapeamento detalhado de itens
        det_a = {x["id"]: x.get("overall_pass", False) for x in a.get("details", []) if "id" in x}
        det_b = {x["id"]: x.get("overall_pass", False) for x in b.get("details", []) if "id" in x}

        common_ids = set(det_a.keys()) & set(det_b.keys())
        regressions = [i for i in common_ids if det_a[i] and not det_b[i]]
        improvements = [i for i in common_ids if not det_a[i] and det_b[i]]

        lines.append("")
        lines.append(f"--- REGRESSÕES (Passavam em A, Falharam em B): {len(regressions)} ---")
        if regressions:
            lines.append("  ⚠ " + ", ".join(sorted(regressions)))
        else:
            lines.append("  ✓ Nenhuma regressão detectada!")

        lines.append("")
        lines.append(f"--- MELHORIAS (Falhavam em A, Passaram em B): {len(improvements)} ---")
        if improvements:
            lines.append("  ★ " + ", ".join(sorted(improvements)))
        else:
            lines.append("  (Nenhuma nova melhoria pontual)")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Sub-aba 3: Playground Tool-Calling (:8080)
    # -------------------------------------------------------------------------
    def _build_playground_tab(self):
        lay = QVBoxLayout(self.tab_play)

        gb_cfg = QGroupBox("Configuração do Router Local")
        hl_cfg = QHBoxLayout()
        hl_cfg.addWidget(QLabel("Servidor llama.cpp:"))
        self.edit_host = QLineEdit("http://127.0.0.1:8080")
        hl_cfg.addWidget(self.edit_host, 2)
        hl_cfg.addWidget(QLabel("Exemplos rápidos:"))
        self.combo_presets_prompt = QComboBox()
        self.combo_presets_prompt.addItems([
            "Verificar serviço bits no Windows (PowerShell)",
            "Uso de disco no filesystem raiz (Bash)",
            "Ler arquivo de configuração (read_file)",
            "Abstenção: pergunta geral sem ferramentas",
        ])
        self.combo_presets_prompt.currentIndexChanged.connect(self._apply_preset_prompt)
        hl_cfg.addWidget(self.combo_presets_prompt, 3)
        QVBoxLayout(gb_cfg).addLayout(hl_cfg)
        lay.addWidget(gb_cfg)

        gb_input = QGroupBox("Prompt de Teste")
        vl_in = QVBoxLayout(gb_input)
        self.edit_prompt = QTextEdit()
        self.edit_prompt.setMaximumHeight(80)
        self.edit_prompt.setText("Check the status of the Background Intelligent Transfer service.")
        vl_in.addWidget(self.edit_prompt)

        hl_btn = QHBoxLayout()
        self.btn_send_play = QPushButton("Enviar Prompt para o Modelo (:8080)")
        self.btn_send_play.clicked.connect(self.send_playground)
        hl_btn.addWidget(self.btn_send_play)
        self.lbl_play_status = QLabel("Pronto.")
        hl_btn.addWidget(self.lbl_play_status)
        hl_btn.addStretch(1)
        vl_in.addLayout(hl_btn)
        lay.addWidget(gb_input)

        gb_out = QGroupBox("Resposta e Diagnóstico de Tool Calling")
        vl_out = QVBoxLayout(gb_out)
        self.out_play = QTextEdit()
        self.out_play.setReadOnly(True)
        self.out_play.setFontFamily("Consolas")
        vl_out.addWidget(self.out_play)
        lay.addWidget(gb_out, 1)

    def _apply_preset_prompt(self, idx: int):
        prompts = [
            "Check the status of the Background Intelligent Transfer service.",
            "How much disk space is free on the root filesystem?",
            "Read the application config at config/app.yaml.",
            "What is the capital of France and what language is spoken there?",
        ]
        if 0 <= idx < len(prompts):
            self.edit_prompt.setText(prompts[idx])

    def send_playground(self):
        prompt = self.edit_prompt.toPlainText().strip()
        if not prompt:
            return

        host = self.edit_host.text().strip()
        self.btn_send_play.setEnabled(False)
        self.lbl_play_status.setText("Aguardando resposta do servidor...")
        self.out_play.setText("Enviando requisição POST para " + host + "...")

        self.play_worker = PlaygroundWorker(prompt=prompt, base_url=host)
        self.play_worker.result_ready.connect(self.on_play_result)
        self.play_worker.error.connect(self.on_play_error)
        self.play_worker.start()

    def on_play_result(self, res: Dict[str, Any]):
        self.btn_send_play.setEnabled(True)
        elapsed = res.get("elapsed_s", 0)
        has_tag = res.get("has_tag", False)
        calls = res.get("tool_calls", [])
        json_ok = res.get("json_valid", False)

        tag_str = "[OK] Tag <tool_call> presente" if has_tag else "[AVISO] Sem tag <tool_call> (Abstenção/Texto puro)"
        json_str = "[OK] JSON Válido" if json_ok else "[ERRO] Falha no parsing do JSON de chamada"

        summary_lines = [
            f"=== Resposta do Modelo em {elapsed}s ===",
            f"Status do Formato: {tag_str} | {json_str}",
            f"Chamadas detectadas: {len(calls)}",
            "",
            "--- Chamadas de Ferramenta Extraídas ---",
            json.dumps(calls, indent=2, ensure_ascii=False) if calls else "(nenhuma chamada de ferramenta realizada)",
            "",
            "--- Resposta Completa do Modelo ---",
            res.get("raw_text", ""),
        ]
        self.out_play.setText("\n".join(summary_lines))
        self.lbl_play_status.setText(f"Concluído em {elapsed}s.")

    def on_play_error(self, err: str):
        self.btn_send_play.setEnabled(True)
        self.lbl_play_status.setText("Erro na comunicação.")
        self.out_play.setText(f"[FALHA NA CONEXÃO]\n{err}")
