"""Splits (aba 4): Manifesto, filtros avançados, visualizador completo de amostras e materialização."""
import json
import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QByteArray, QAbstractTableModel, QModelIndex, Qt
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QPushButton, QScrollArea,
                               QSplitter, QTableView, QTabWidget, QTextEdit,
                               QVBoxLayout, QWidget)

from app.core import paths, repo_state
from app.core.dataset_loader import (get_sample_by_uuid, get_taxonomy_categories,
                                    query_samples)
from app.workers.command_worker import CommandWorker

ROOT = repo_state.ROOT


class SamplesTableModel(QAbstractTableModel):
    """Modelo de dados virtualizado para carregamento instantâneo de milhares de amostras."""
    HEADERS = ["UUID", "Pilar", "Efeito", "Executor", "Padrão", "Tokens", "Fonte"]

    def __init__(self, records: Optional[List[Dict[str, Any]]] = None):
        super().__init__()
        self.records: List[Dict[str, Any]] = records or []

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.records)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        row = index.row()
        col = index.column()
        if not (0 <= row < len(self.records)):
            return None

        r = self.records[row]
        if col == 0:
            return r.get("uuid", "")
        elif col in (1, 2):
            sample = get_sample_by_uuid(r.get("uuid", ""))
            if col == 1:
                return sample.get("pillar", "N/A") if sample else "N/A"
            else:
                return sample.get("effect", "N/A") if sample else "N/A"
        elif col == 3:
            return r.get("executor", "")
        elif col == 4:
            return r.get("pattern") or (f"{r.get('n_calls')} calls")
        elif col == 5:
            return str(r.get("token_len", ""))
        elif col == 6:
            return r.get("source", "")
        return None

    def set_records(self, records: List[Dict[str, Any]]):
        self.beginResetModel()
        self.records = records
        self.endResetModel()


class SplitsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker: Optional[CommandWorker] = None
        self.current_records: List[Dict[str, Any]] = []

        lay = QVBoxLayout(self)

        # 1. Filtros no Topo
        gb_filters = QGroupBox("Filtros do Dataset & Manifesto")
        fl = QVBoxLayout(gb_filters)

        tax = get_taxonomy_categories()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Executor:"))
        self.combo_executor = QComboBox()
        self.combo_executor.addItems(["(Todos)"] + tax.get("executors", []))
        row1.addWidget(self.combo_executor)

        row1.addWidget(QLabel("Padrão / Calls:"))
        self.combo_calls = QComboBox()
        self.combo_calls.addItems(["(Todos)"] + tax.get("patterns", []))
        row1.addWidget(self.combo_calls)

        row1.addWidget(QLabel("Pilar / Categoria:"))
        self.combo_pillar = QComboBox()
        self.combo_pillar.addItems(["(Todos)"] + tax.get("pillars", []))
        row1.addWidget(self.combo_pillar)

        row1.addWidget(QLabel("Subpilar:"))
        self.combo_subpillar = QComboBox()
        self.combo_subpillar.addItems(["(Todos)"] + tax.get("subpillars", []))
        row1.addWidget(self.combo_subpillar)

        row1.addStretch(1)
        fl.addLayout(row1)

        # Linha 2 de Filtros (Orientada a Efeito Colateral + Busca + Ações)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Efeito Colateral:"))
        self.combo_effect = QComboBox()
        self.combo_effect.addItems([
            "(Todos)",
            "read_only (Leitura/Consulta)",
            "mutating (Criação/Modificação/Exec)",
            "none (Abstenção)",
        ])
        row2.addWidget(self.combo_effect)

        row2.addWidget(QLabel("Max Tokens:"))
        self.edit_maxlen = QLineEdit()
        self.edit_maxlen.setPlaceholderText("ex: 500")
        self.edit_maxlen.setMaximumWidth(70)
        row2.addWidget(self.edit_maxlen)

        row2.addWidget(QLabel("Busca / UUID:"))
        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("Digite um comando, texto ou UUID...")
        row2.addWidget(self.edit_search, 2)

        self.btn_filter = QPushButton("Filtrar")
        self.btn_filter.clicked.connect(self.apply_filter)
        row2.addWidget(self.btn_filter)

        self.btn_reset = QPushButton("Ver Todos / Resetar")
        self.btn_reset.clicked.connect(self.reset_filters)
        row2.addWidget(self.btn_reset)

        self.btn_export = QPushButton("Materializar View (.jsonl)")
        self.btn_export.clicked.connect(self.toggle_export)
        row2.addWidget(self.btn_export)

        fl.addLayout(row2)
        lay.addWidget(gb_filters)

        # Conexões automáticas de filtro ao alterar seleções
        self.combo_executor.currentIndexChanged.connect(self.apply_filter)
        self.combo_calls.currentIndexChanged.connect(self.apply_filter)
        self.combo_pillar.currentIndexChanged.connect(self.apply_filter)
        self.combo_subpillar.currentIndexChanged.connect(self.apply_filter)
        self.combo_effect.currentIndexChanged.connect(self.apply_filter)
        self.edit_search.returnPressed.connect(self.apply_filter)
        self.edit_maxlen.returnPressed.connect(self.apply_filter)

        # 2. Área Central com QSplitter (Tabela à esquerda, Visualizador à direita)
        splitter = QSplitter(Qt.Horizontal)

        # Lado Esquerdo: Tabela de Amostras
        w_left = QWidget()
        vl_left = QVBoxLayout(w_left)
        vl_left.setContentsMargins(0, 0, 0, 0)
        self.lbl_count = QLabel("Carregando amostras...")
        vl_left.addWidget(self.lbl_count)

        self.table_model = SamplesTableModel()
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.selectionModel().selectionChanged.connect(self.on_selection_changed)
        vl_left.addWidget(self.table)
        splitter.addWidget(w_left)

        # Lado Direito: Detalhes da Amostra
        w_right = QWidget()
        vl_right = QVBoxLayout(w_right)
        vl_right.setContentsMargins(0, 0, 0, 0)

        # Metadados
        self.gb_meta = QGroupBox("Metadados da Amostra")
        self.lbl_meta = QLabel("Selecione uma amostra na tabela à esquerda para visualizar.")
        QVBoxLayout(self.gb_meta).addWidget(self.lbl_meta)
        vl_right.addWidget(self.gb_meta)

        # Abas de Visualização
        self.sample_tabs = QTabWidget()

        # Sub-aba Diálogo
        self.out_dialogue = QTextEdit()
        self.out_dialogue.setReadOnly(True)
        self.out_dialogue.setFontFamily("Consolas")
        self.sample_tabs.addTab(self.out_dialogue, "Diálogo & Chamadas")

        # Sub-aba Ferramentas Declaradas
        self.out_tools = QTextEdit()
        self.out_tools.setReadOnly(True)
        self.out_tools.setFontFamily("Consolas")
        self.sample_tabs.addTab(self.out_tools, "Ferramentas Declaradas (Schema)")

        # Sub-aba JSON Integral
        self.out_raw = QTextEdit()
        self.out_raw.setReadOnly(True)
        self.out_raw.setFontFamily("Consolas")
        self.sample_tabs.addTab(self.out_raw, "JSON Bruto")

        # Sub-aba Renderizador SVG Dinâmico (ativa se amostra contiver SVG)
        self.svg_scroll = QScrollArea()
        self.svg_scroll.setWidgetResizable(True)
        self.svg_widget = QSvgWidget()
        self.svg_scroll.setWidget(self.svg_widget)
        self.svg_tab_idx = self.sample_tabs.addTab(self.svg_scroll, "Preview SVG (Render)")
        self.sample_tabs.setTabVisible(self.svg_tab_idx, False)

        vl_right.addWidget(self.sample_tabs, 1)
        splitter.addWidget(w_right)

        splitter.setSizes([380, 620])
        lay.addWidget(splitter, 1)

        # Status inferior
        self.lbl_status = QLabel("Pronto.")
        lay.addWidget(self.lbl_status)

        # Dispara primeiro carregamento
        self.apply_filter()

    def reset_filters(self):
        self.combo_executor.setCurrentIndex(0)
        self.combo_calls.setCurrentIndex(0)
        self.combo_pillar.setCurrentIndex(0)
        self.combo_subpillar.setCurrentIndex(0)
        self.combo_effect.setCurrentIndex(0)
        self.edit_search.clear()
        self.edit_maxlen.clear()
        self.apply_filter()

    def apply_filter(self):
        ex = self.combo_executor.currentText()
        calls = self.combo_calls.currentText()
        pillar = self.combo_pillar.currentText()
        subpillar = self.combo_subpillar.currentText()
        eff_raw = self.combo_effect.currentText()
        search_txt = self.edit_search.text()

        ex = "" if ex == "(Todos)" else ex
        calls = "" if calls == "(Todos)" else calls
        pillar = "" if pillar == "(Todos)" else pillar
        subpillar = "" if subpillar == "(Todos)" else subpillar

        if eff_raw == "(Todos)":
            eff = ""
        elif "read_only" in eff_raw:
            eff = "read_only"
        elif "mutating" in eff_raw:
            eff = "mutating"
        elif "none" in eff_raw:
            eff = "none"
        else:
            eff = eff_raw

        max_len_val = None
        if self.edit_maxlen.text().strip().isdigit():
            max_len_val = int(self.edit_maxlen.text().strip())

        self.current_records = query_samples(
            executor=ex,
            pattern_calls=calls,
            pillar=pillar,
            subpillar=subpillar,
            effect=eff,
            search_text=search_txt,
            max_len=max_len_val,
        )

        n = len(self.current_records)
        self.lbl_count.setText(f"Exibindo {n} amostras correspondentes:")
        self.table_model.set_records(self.current_records)

        if n > 0:
            self.table.selectRow(0)
        else:
            self.clear_sample_view()

    def on_selection_changed(self):
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            return
        row = indexes[0].row()
        if 0 <= row < len(self.current_records):
            rec = self.current_records[row]
            uuid = rec["uuid"]
            self.display_sample(uuid, rec)

    def display_sample(self, uuid: str, manifest_rec: Dict[str, Any]):
        sample = get_sample_by_uuid(uuid)
        if not sample:
            self.lbl_meta.setText(f"Amostra {uuid} não encontrada no PURIFIED.")
            return

        is_retra = sample.get("is_retrajectory", False)
        retra_badge = "SIM (Auto-recuperação/Retry)" if is_retra else "NÃO (Chamada direta/progressiva)"
        pat = manifest_rec.get("pattern", "N/A")

        meta_lines = [
            f"<b>UUID:</b> {uuid} | <b>Executor:</b> {sample.get('executor', manifest_rec.get('executor'))} | "
            f"<b>Pilar:</b> {sample.get('pillar', 'N/A')} | <b>Subpilar:</b> {sample.get('subpillar', 'N/A')} | "
            f"<b>Efeito:</b> {sample.get('effect', 'N/A').upper()} | <b>Tokens:</b> {manifest_rec.get('token_len', sample.get('token_length'))}",
            f"<b>Padrão de Chamada:</b> {pat.upper()} | <b>Retrajetória:</b> {retra_badge} | <b>Fonte:</b> {sample.get('source', 'N/A')}",
        ]
        self.lbl_meta.setText("<br>".join(meta_lines))

        # 1. Monta visualização do diálogo
        dialogue_text = []
        turn_idx = 1
        for msg in sample.get("messages", []):
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if role == "USER":
                dialogue_text.append(f"═══════════════════════════════════════════════════════════════════")
                dialogue_text.append(f"👤 [USER - TURNO {turn_idx}]")
                dialogue_text.append(content)
                dialogue_text.append("")
                turn_idx += 1
            elif role == "ASSISTANT":
                dialogue_text.append(f"🤖 [ASSISTANT]")
                if content:
                    dialogue_text.append(content)
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        fname = fn.get("name", "")
                        fargs = fn.get("arguments", "")
                        try:
                            if isinstance(fargs, str):
                                fargs_parsed = json.loads(fargs)
                                fargs_str = json.dumps(fargs_parsed, indent=2, ensure_ascii=False)
                            else:
                                fargs_str = json.dumps(fargs, indent=2, ensure_ascii=False)
                        except Exception:
                            fargs_str = str(fargs)
                        dialogue_text.append(f"  <tool_call>")
                        dialogue_text.append(f"  name: {fname}")
                        dialogue_text.append(f"  arguments:\n{fargs_str}")
                        dialogue_text.append(f"  </tool_call>")
                dialogue_text.append("")
            elif role == "TOOL":
                dialogue_text.append(f"⚙️ [TOOL OUTPUT - {msg.get('name', 'tool')}]")
                dialogue_text.append(content)
                dialogue_text.append("")

        self.out_dialogue.setText("\n".join(dialogue_text))

        # 2. Ferramentas declaradas
        tools = sample.get("tools", [])
        self.out_tools.setText(json.dumps(tools, indent=2, ensure_ascii=False) if tools else "(nenhuma ferramenta declarada)")

        # 3. JSON bruto
        self.out_raw.setText(json.dumps(sample, indent=2, ensure_ascii=False))

        # 4. Preview SVG dinâmico (renderiza gráfico se existir)
        svg_code = self._extract_svg(sample)
        if svg_code:
            try:
                self.svg_widget.load(QByteArray(svg_code.encode("utf-8")))
                self.sample_tabs.setTabVisible(self.svg_tab_idx, True)
            except Exception:
                self.sample_tabs.setTabVisible(self.svg_tab_idx, False)
        else:
            self.sample_tabs.setTabVisible(self.svg_tab_idx, False)

    def _extract_svg(self, sample: Dict[str, Any]) -> str:
        """Extrai código SVG de campos diretos ou do conteúdo das mensagens."""
        for k in ("svg", "svg_code", "output", "render"):
            val = sample.get(k)
            if isinstance(val, str) and "<svg" in val and "</svg>" in val:
                start = val.find("<svg")
                end = val.find("</svg>", start)
                if start != -1 and end != -1:
                    return val[start:end + 6]
        for msg in sample.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str) and "<svg" in content and "</svg>" in content:
                start = content.find("<svg")
                end = content.find("</svg>", start)
                if start != -1 and end != -1:
                    return content[start:end + 6]
        return ""

    def clear_sample_view(self):
        self.lbl_meta.setText("Nenhuma amostra correspondente.")
        self.out_dialogue.clear()
        self.out_tools.clear()
        self.out_raw.clear()
        self.sample_tabs.setTabVisible(self.svg_tab_idx, False)

    def toggle_export(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.lbl_status.setText("Cancelando exportação...")
            self.btn_export.setEnabled(False)
            return

        cmd = ["python", "scripts/pipeline/make_view.py"]
        ex = self.combo_executor.currentText()
        calls = self.combo_calls.currentText()
        pillar = self.combo_pillar.currentText()
        subpillar = self.combo_subpillar.currentText()
        eff_raw = self.combo_effect.currentText()

        ex = "" if ex == "(Todos)" else ex
        calls = "" if calls == "(Todos)" else calls
        pillar = "" if pillar == "(Todos)" else pillar
        subpillar = "" if subpillar == "(Todos)" else subpillar

        if eff_raw == "(Todos)":
            eff = ""
        elif "read_only" in eff_raw:
            eff = "read_only"
        elif "mutating" in eff_raw:
            eff = "mutating"
        elif "none" in eff_raw:
            eff = "none"
        else:
            eff = eff_raw

        if ex:
            cmd += ["--executor", ex]
        if calls:
            cmd += ["--calls", calls]
        if pillar:
            cmd += ["--pillar", pillar]
        if subpillar:
            cmd += ["--subpillar", subpillar]
        if eff:
            cmd += ["--effect", eff]
        if self.edit_maxlen.text().strip().isdigit():
            cmd += ["--max-len", self.edit_maxlen.text().strip()]

        out = os.path.join(paths.get_log_dir(), "view_gui.jsonl")
        cmd += ["--out", out]

        self.lbl_status.setText("Materializando view em background...")
        self.btn_export.setText("Cancelar")

        self.worker = CommandWorker(cmd, cwd=ROOT)
        self.worker.finished.connect(lambda ret, text: self.on_export_finished(ret, text, out))
        self.worker.start()

    def on_export_finished(self, ret: int, text: str, out_path: str):
        self.btn_export.setText("Materializar View (.jsonl)")
        self.btn_export.setEnabled(True)
        if ret == 0:
            self.lbl_status.setText(f"[OK] View exportada com sucesso -> {out_path}")
        else:
            self.lbl_status.setText(f"[ERRO] Falha na materialização (código {ret}).")
