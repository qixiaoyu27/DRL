from __future__ import annotations

import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QDoubleSpinBox,
)


class WindDesigner(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wind Field Designer")
        self.resize(850, 500)

        layout = QVBoxLayout(self)
        top = QHBoxLayout()

        self.bgx = QDoubleSpinBox(); self.bgx.setRange(-20, 20); self.bgx.setValue(1.0)
        self.bgy = QDoubleSpinBox(); self.bgy.setRange(-20, 20); self.bgy.setValue(0.5)
        top.addWidget(QLabel("Background Wx")); top.addWidget(self.bgx)
        top.addWidget(QLabel("Background Wy")); top.addWidget(self.bgy)

        btn_add = QPushButton("新增涡旋")
        btn_add.clicked.connect(self.add_row)
        btn_save = QPushButton("保存JSON")
        btn_save.clicked.connect(self.save_json)
        btn_load = QPushButton("加载JSON")
        btn_load.clicked.connect(self.load_json)
        top.addWidget(btn_add); top.addWidget(btn_save); top.addWidget(btn_load)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["x(m)", "y(m)", "gamma", "core_radius(m)"])

        layout.addLayout(top)
        layout.addWidget(self.table)

    def add_row(self, v=None):
        r = self.table.rowCount()
        self.table.insertRow(r)
        vals = v or [1000.0, 1000.0, 1200.0, 120.0]
        for c, val in enumerate(vals):
            self.table.setItem(r, c, QTableWidgetItem(str(val)))

    def to_dict(self):
        vortices = []
        for r in range(self.table.rowCount()):
            vortices.append(
                {
                    "x": float(self.table.item(r, 0).text()),
                    "y": float(self.table.item(r, 1).text()),
                    "gamma": float(self.table.item(r, 2).text()),
                    "core_radius": float(self.table.item(r, 3).text()),
                }
            )
        return {"background_xy": [self.bgx.value(), self.bgy.value()], "vortices": vortices}

    def save_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存风场", "wind_custom.json", "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开风场", "", "JSON (*.json)")
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.bgx.setValue(float(data["background_xy"][0]))
        self.bgy.setValue(float(data["background_xy"][1]))
        self.table.setRowCount(0)
        for v in data.get("vortices", []):
            self.add_row([v["x"], v["y"], v["gamma"], v["core_radius"]])


if __name__ == "__main__":
    app = QApplication([])
    w = WindDesigner()
    w.show()
    app.exec()
