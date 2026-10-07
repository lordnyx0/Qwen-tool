"""Qwen Control Center — main."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QTabWidget

from app.core import config
from app.ui.avals import EvalsTab
from app.ui.docs import DocsTab
from app.ui.estado import EstadoTab
from app.ui.splits import SplitsTab
from app.ui.treinos import TreinosTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.get_project_name())
        self.resize(1050, 720)

        # Menu superior
        menubar = self.menuBar()
        menu_file = menubar.addMenu("&Arquivo")

        act_open = QAction("&Abrir Pasta do Projeto...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_project_folder)
        menu_file.addAction(act_open)

        act_reload = QAction("&Recarregar Projeto Atual", self)
        act_reload.setShortcut("F5")
        act_reload.triggered.connect(self.reload_current_project)
        menu_file.addAction(act_reload)

        menu_file.addSeparator()
        act_exit = QAction("&Sair", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

        self.tabs = None
        self.rebuild_tabs()

    def rebuild_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(EstadoTab(), "1 Estado")
        self.tabs.addTab(TreinosTab(), "2 Treinos")
        self.tabs.addTab(EvalsTab(), "3 Evals")
        self.tabs.addTab(SplitsTab(), "4 Splits")
        self.tabs.addTab(DocsTab(), "5 Docs")
        self.setCentralWidget(self.tabs)
        self.setWindowTitle(config.get_project_name())

    def open_project_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta do Projeto")
        if folder:
            os.environ["WORKBENCH_PROJECT_ROOT"] = folder
            self.reload_current_project()

    def reload_current_project(self):
        from app.core import dataset_loader
        dataset_loader._MANIFEST_CACHE = None
        dataset_loader._PURIFIED_CACHE = None
        config.get_project_config(force_reload=True)
        self.rebuild_tabs()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())



if __name__ == "__main__":
    main()
