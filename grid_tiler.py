#!/usr/bin/env python3
"""
Grid Tiler - 画像タイリングツール

1枚の画像をNxNグリッドに並べて1枚の画像として出力する。
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QGroupBox, QComboBox
)
from PySide6.QtCore import Qt

# パスを追加
sys.path.insert(0, str(Path(__file__).parent))

from cv2_utils import load_image_as_bgra, save_image


class GridTilerWindow(QMainWindow):
    """グリッドタイリングメインウィンドウ"""

    MAX_OUTPUT_SIZE = 2400

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Grid Tiler - 画像タイリングツール')
        self.setMinimumSize(400, 300)
        self.setAcceptDrops(True)

        self.source_image: Optional[np.ndarray] = None
        self.source_path: str = ''

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # グリッドサイズ選択
        grid_group = QGroupBox('グリッドサイズ')
        grid_layout = QHBoxLayout(grid_group)
        grid_layout.addWidget(QLabel('レイアウト:'))
        self.combo_grid = QComboBox()
        self.combo_grid.addItem('2x2（4枚）', 2)
        self.combo_grid.addItem('3x3（9枚）', 3)
        self.combo_grid.setCurrentIndex(0)
        self.combo_grid.currentIndexChanged.connect(self._on_grid_changed)
        grid_layout.addWidget(self.combo_grid)
        grid_layout.addStretch()
        main_layout.addWidget(grid_group)

        # ドロップゾーン
        drop_group = QGroupBox('画像入力')
        drop_layout = QVBoxLayout(drop_group)

        self.drop_zone = QLabel('画像をここにドロップ')
        self.drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_zone.setStyleSheet(
            "background-color: #1e1e1e; border: 2px dashed #3e3e42; "
            "border-radius: 5px; padding: 30px; color: #888; font-size: 14px;"
        )
        self.drop_zone.setMinimumHeight(120)
        drop_layout.addWidget(self.drop_zone)

        self.btn_select = QPushButton('ファイルを選択...')
        self.btn_select.clicked.connect(self._select_file)
        drop_layout.addWidget(self.btn_select)

        main_layout.addWidget(drop_group)

        # 情報表示
        info_group = QGroupBox('情報')
        info_layout = QVBoxLayout(info_group)
        self.lbl_filename = QLabel('ファイル: 未選択')
        self.lbl_filename.setStyleSheet('color: #888;')
        info_layout.addWidget(self.lbl_filename)
        self.lbl_input_size = QLabel('入力サイズ: -')
        self.lbl_input_size.setStyleSheet('color: #888;')
        info_layout.addWidget(self.lbl_input_size)
        self.lbl_output_size = QLabel('出力サイズ: -')
        self.lbl_output_size.setStyleSheet('color: #888;')
        info_layout.addWidget(self.lbl_output_size)
        main_layout.addWidget(info_group)

        # 保存ボタン
        save_group = QGroupBox('出力')
        save_layout = QVBoxLayout(save_group)
        self.btn_save = QPushButton('タイリング画像を保存...')
        self.btn_save.setStyleSheet(
            'background-color: #16a34a; color: white; font-weight: bold; padding: 10px;'
        )
        self.btn_save.clicked.connect(self._save_image)
        self.btn_save.setEnabled(False)
        save_layout.addWidget(self.btn_save)
        main_layout.addWidget(save_group)

        main_layout.addStretch()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._load_image(path)

    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '画像を選択', '',
            '画像 (*.png *.jpg *.jpeg *.bmp *.webp)'
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        try:
            image = load_image_as_bgra(path)
        except Exception as e:
            QMessageBox.warning(self, 'エラー', f'画像の読み込みに失敗しました: {e}')
            return

        # Noneチェック（OpenCVは読み込み失敗時にNoneを返す場合がある）
        if image is None:
            QMessageBox.warning(self, 'エラー', '画像の読み込みに失敗しました。ファイルが破損しているか、対応していない形式です。')
            return

        self.source_image = image
        self.source_path = path

        h, w = image.shape[:2]
        filename = Path(path).name

        self.lbl_filename.setText(f'ファイル: {filename}')
        self.lbl_filename.setStyleSheet('color: #4ade80;')
        self.lbl_input_size.setText(f'入力サイズ: {w} x {h}')
        self.lbl_input_size.setStyleSheet('color: #ccc;')

        self._update_output_info()
        self.btn_save.setEnabled(True)

    def _on_grid_changed(self):
        if self.source_image is not None:
            self._update_output_info()

    def _update_output_info(self):
        if self.source_image is None:
            return

        grid_size = self.combo_grid.currentData()
        h, w = self.source_image.shape[:2]

        # タイリング後のサイズ計算
        tiled_h = h * grid_size
        tiled_w = w * grid_size

        # 2400px制限適用後のサイズ
        scale = min(self.MAX_OUTPUT_SIZE / tiled_h, self.MAX_OUTPUT_SIZE / tiled_w, 1.0)
        final_h = max(1, int(tiled_h * scale))
        final_w = max(1, int(tiled_w * scale))

        if scale < 1.0:
            self.lbl_output_size.setText(
                f'出力サイズ: {final_w} x {final_h}（{tiled_w}x{tiled_h}から縮小）'
            )
        else:
            self.lbl_output_size.setText(f'出力サイズ: {final_w} x {final_h}')
        self.lbl_output_size.setStyleSheet('color: #60a5fa;')

    def _create_tiled_image(self) -> np.ndarray:
        """タイリング画像を生成（OOM対策: 先に縮小してからタイリング）"""
        grid_size = self.combo_grid.currentData()
        h, w = self.source_image.shape[:2]

        # タイリング後のサイズ計算
        tiled_h = h * grid_size
        tiled_w = w * grid_size

        # 2400px制限のスケール計算
        scale = min(self.MAX_OUTPUT_SIZE / tiled_h, self.MAX_OUTPUT_SIZE / tiled_w, 1.0)

        # OOM対策: 縮小が必要な場合は先に元画像を縮小してからタイリング
        if scale < 1.0:
            # 各セルの目標サイズを計算
            cell_h = max(1, int(h * scale))
            cell_w = max(1, int(w * scale))
            resized = cv2.resize(self.source_image, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            tiled = np.tile(resized, (grid_size, grid_size, 1))
        else:
            # 縮小不要: そのままタイリング
            tiled = np.tile(self.source_image, (grid_size, grid_size, 1))

        return tiled

    def _save_image(self):
        if self.source_image is None:
            QMessageBox.warning(self, '警告', '画像が読み込まれていません')
            return

        # デフォルトファイル名
        grid_size = self.combo_grid.currentData()
        base_name = Path(self.source_path).stem if self.source_path else 'output'
        default_name = f'{base_name}_{grid_size}x{grid_size}.png'

        path, _ = QFileDialog.getSaveFileName(
            self, '保存先を選択', default_name,
            'PNG画像 (*.png)'
        )
        if not path:
            return

        try:
            tiled = self._create_tiled_image()
            ok = save_image(path, tiled)
            if not ok:
                raise RuntimeError('save_image returned False')
            QMessageBox.information(self, '完了', f'保存しました:\n{path}')
        except Exception as e:
            QMessageBox.warning(self, 'エラー', f'保存に失敗しました: {e}')


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Grid Tiler')
    app.setStyle('Fusion')

    app.setStyleSheet("""
        QMainWindow { background-color: #1e1e1e; }
        QWidget { background-color: #252526; color: #cccccc; }
        QPushButton { background-color: #0e639c; color: white; border: none; padding: 5px 15px; border-radius: 3px; }
        QPushButton:hover { background-color: #1177bb; }
        QPushButton:pressed { background-color: #094771; }
        QPushButton:disabled { background-color: #3c3c3c; color: #666; }
        QGroupBox { border: 1px solid #3e3e42; margin-top: 10px; padding-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        QLabel { color: #cccccc; }
        QComboBox { background-color: #3c3c3c; border: 1px solid #3e3e42; padding: 5px; min-width: 120px; }
        QComboBox:hover { border: 1px solid #007acc; }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView { background-color: #2d2d30; selection-background-color: #094771; }
    """)

    window = GridTilerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
