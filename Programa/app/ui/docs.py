"""Docs (aba 5): visualizador universal de documentação markdown do projeto."""
import glob
import os
from typing import List

from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QPushButton, QTextEdit,
                               QVBoxLayout, QWidget)

from app.core import paths, repo_state


class DocsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.paths: List[str] = []
        lay = QVBoxLayout(self)

        hl = QHBoxLayout()
        self.combo = QComboBox()
        hl.addWidget(self.combo, 3)

        b_open = QPushButton("Abrir Documento")
        b_open.clicked.connect(self.show_doc)
        hl.addWidget(b_open)

        b_reload = QPushButton("Atualizar Lista")
        b_reload.clicked.connect(self.refresh_docs)
        hl.addWidget(b_reload)

        lay.addLayout(hl)

        self.out = QTextEdit()
        self.out.setReadOnly(True)
        self.out.setFontFamily("Consolas")
        lay.addWidget(self.out, 1)

        self.refresh_docs()

    def refresh_docs(self):
        root = paths.find_root()
        found_docs: List[str] = []

        # 1. Documentos dentro da pasta docs/ (recursivo)
        docs_dir = os.path.join(root, "docs")
        if os.path.isdir(docs_dir):
            found_docs.extend(glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True))

        # 2. Documentos na raiz do repositório
        for root_md in glob.glob(os.path.join(root, "*.md")):
            found_docs.append(root_md)

        # 3. Documentos em subpastas conhecidas (ex: Programa/docs, data/README.md)
        for extra in (os.path.join(root, "Programa", "docs", "ESPECIFICACAO.md"),
                      os.path.join(root, "data", "README.md")):
            if os.path.isfile(extra) and extra not in found_docs:
                found_docs.append(extra)

        # Deduplica e ordena preservando caminhos relativos
        seen = set()
        clean_docs = []
        for d in sorted(found_docs):
            norm = os.path.abspath(d)
            if norm not in seen and os.path.isfile(norm):
                seen.add(norm)
                clean_docs.append(norm)

        self.paths = clean_docs
        self.combo.clear()
        if self.paths:
            labels = [os.path.relpath(p, root).replace("\\", "/") for p in self.paths]
            self.combo.addItems(labels)
            self.show_doc()
        else:
            self.combo.addItem("(nenhum documento .md encontrado)")
            self.out.setText(f"Nenhum arquivo .md localizado em: {root}")

    def show_doc(self):
        idx = self.combo.currentIndex()
        if 0 <= idx < len(self.paths):
            fp = self.paths[idx]
            try:
                with open(fp, encoding="utf-8", errors="replace") as f:
                    self.out.setText(f.read()[:100000])
            except Exception as e:
                self.out.setText(f"Erro ao ler {fp}: {e}")

