"""
Simple Aligner - 位置合わせ専用モジュール
ブラシやパーツ機能を省いたシンプル版
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QListWidget,
    QListWidgetItem, QGroupBox, QSplitter, QScrollArea, QProgressDialog,
    QCheckBox, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal, QPoint, QMimeData, QSettings
from PySide6.QtGui import QPixmap, QMouseEvent, QDragEnterEvent, QDropEvent

from aligner import Aligner, AlignConfig
from cv2_utils import (
    load_image,
    save_image,
    convert_to_qimage,
    compute_common_valid_rect,
    crop_image
)


class ImageItem:
    """画像アイテム"""
    def __init__(self, path: str):
        self.path = path
        self.name = Path(path).name
        self.image: Optional[np.ndarray] = None
        self.aligned_image: Optional[np.ndarray] = None
        self.aligned_valid_mask: Optional[np.ndarray] = None  # 255=有効領域
        self.alignment_success: bool = False
        self.alignment_score: float = 0.0
        self.alignment_method: str = ""
        self.alignment_inliers: int = 0
        self.alignment_total_matches: int = 0
        self.alignment_error: str = ""


class SimplePreviewWidget(QWidget):
    """シンプルなプレビューウィジェット"""

    roi_selected = Signal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.base_image: Optional[np.ndarray] = None
        self.overlay_image: Optional[np.ndarray] = None
        self.scale = 1.0

        # ROI
        self.roi_mode = False
        self.roi_start = None
        self.roi_current = None
        self.roi_rect = None
        self.dragging = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("画像を選択してください")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: #2b2b2b; color: #888;")

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.label)
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)

    def set_base_image(self, image: Optional[np.ndarray]):
        self.base_image = image
        self._update_display()

    def set_overlay_image(self, image: Optional[np.ndarray]):
        self.overlay_image = image
        self._update_display()

    def set_roi_mode(self, enabled: bool):
        self.roi_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def set_roi(self, x: int, y: int, w: int, h: int):
        self.roi_rect = (x, y, w, h)
        self._update_display()

    def clear_roi(self):
        self.roi_rect = None
        self.roi_start = None
        self.roi_current = None
        self._update_display()

    def fit_to_window(self):
        if self.base_image is None:
            return

        viewport = self.scroll.viewport()
        img_h, img_w = self.base_image.shape[:2]

        scale_w = (viewport.width() - 20) / img_w
        scale_h = (viewport.height() - 20) / img_h
        self.scale = max(0.01, min(scale_w, scale_h, 1.0))
        self._update_display()

    def _update_display(self):
        if self.base_image is None:
            self.label.setText("画像を選択してください")
            return

        # 表示画像を作成
        display = self.base_image.copy()
        if len(display.shape) == 3 and display.shape[2] == 4:
            display = cv2.cvtColor(display, cv2.COLOR_BGRA2BGR)

        # オーバーレイ
        if self.overlay_image is not None:
            overlay = self.overlay_image
            if overlay.shape[:2] != display.shape[:2]:
                overlay = cv2.resize(overlay, (display.shape[1], display.shape[0]))
            if len(overlay.shape) == 3 and overlay.shape[2] == 4:
                alpha = overlay[:, :, 3:4].astype(float) / 255.0
                overlay_rgb = overlay[:, :, :3].astype(float)
                display = (display.astype(float) * (1 - alpha * 0.5) +
                          overlay_rgb * alpha * 0.5).astype(np.uint8)
            else:
                display = cv2.addWeighted(display, 0.5, overlay, 0.5, 0)

        # ROI描画
        display = self._draw_roi(display)

        # スケール
        if self.scale != 1.0:
            h, w = display.shape[:2]
            display = cv2.resize(display, (int(w * self.scale), int(h * self.scale)))

        # 表示
        qimage = convert_to_qimage(display)
        self.label.setPixmap(QPixmap.fromImage(qimage))

    def _draw_roi(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        color = (0, 162, 232)  # 水色

        # ドラッグ中
        if self.roi_mode and self.roi_start and self.roi_current:
            x1, y1 = self.roi_start
            x2, y2 = self.roi_current
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w > 0 and h > 0:
                overlay = result.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                result = cv2.addWeighted(overlay, 0.3, result, 0.7, 0)
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        # 確定済み
        elif self.roi_rect:
            x, y, w, h = self.roi_rect
            overlay = result.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            result = cv2.addWeighted(overlay, 0.2, result, 0.8, 0)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        return result

    def _screen_to_image(self, x: int, y: int) -> tuple:
        pixmap = self.label.pixmap()
        if pixmap and not pixmap.isNull():
            offset_x = (self.label.width() - pixmap.width()) // 2
            offset_y = (self.label.height() - pixmap.height()) // 2
            x = x - max(0, offset_x)
            y = y - max(0, offset_y)
        return int(x / self.scale), int(y / self.scale)

    def mousePressEvent(self, event: QMouseEvent):
        if self.base_image is None or not self.roi_mode:
            return

        pos = self.label.mapFrom(self, event.pos())
        img_x, img_y = self._screen_to_image(pos.x(), pos.y())

        self.dragging = True
        self.roi_start = (img_x, img_y)
        self.roi_current = (img_x, img_y)

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.dragging or not self.roi_mode:
            return

        pos = self.label.mapFrom(self, event.pos())
        img_x, img_y = self._screen_to_image(pos.x(), pos.y())
        self.roi_current = (img_x, img_y)
        self._update_display()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.dragging or not self.roi_mode:
            return

        pos = self.label.mapFrom(self, event.pos())
        img_x, img_y = self._screen_to_image(pos.x(), pos.y())

        self.dragging = False

        if self.roi_start:
            x1, y1 = self.roi_start
            x = min(x1, img_x)
            y = min(y1, img_y)
            w = abs(img_x - x1)
            h = abs(img_y - y1)

            self.roi_start = None
            self.roi_current = None

            if w > 10 and h > 10:
                self.roi_selected.emit(x, y, w, h)


class SimpleAlignerWindow(QMainWindow):
    """シンプル位置合わせウィンドウ"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simple Aligner - 位置合わせ専用")
        self.setMinimumSize(1000, 700)
        self.setAcceptDrops(True)

        # データ
        self.base_image: Optional[np.ndarray] = None
        self.base_path: str = ""
        self.images: List[ImageItem] = []
        self.current_index: int = -1
        self.roi: Optional[tuple] = None

        # アライナー
        self.aligner = Aligner(AlignConfig())

        # 設定保存
        self.settings = QSettings("EasyPNGTuber", "SimpleAligner")
        self._last_base_dir = ""
        self._last_add_dir = ""
        self._last_save_dir = ""

        self._setup_ui()
        self._load_settings()
        self._update_trim_info_label()
        self._update_status_summary()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = splitter

        # === 左パネル（画像リスト）===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # ベース画像
        base_group = QGroupBox("ベース画像")
        base_layout = QVBoxLayout(base_group)

        self.btn_load_base = QPushButton("ベース画像を選択...")
        self.btn_load_base.clicked.connect(self._load_base_image)
        base_layout.addWidget(self.btn_load_base)

        self.lbl_base = QLabel("未選択")
        self.lbl_base.setStyleSheet("color: #888;")
        base_layout.addWidget(self.lbl_base)

        left_layout.addWidget(base_group)

        # 差分画像リスト
        diff_group = QGroupBox("差分画像")
        diff_layout = QVBoxLayout(diff_group)

        self.btn_add_images = QPushButton("画像を追加...")
        self.btn_add_images.clicked.connect(self._add_images)
        diff_layout.addWidget(self.btn_add_images)

        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_selected)
        diff_layout.addWidget(self.image_list)

        self.btn_clear_images = QPushButton("リストをクリア")
        self.btn_clear_images.clicked.connect(self._clear_images)
        diff_layout.addWidget(self.btn_clear_images)

        left_layout.addWidget(diff_group)

        splitter.addWidget(left_panel)

        # === 中央（プレビュー）===
        self.preview = SimplePreviewWidget()
        self.preview.roi_selected.connect(self._on_roi_selected)
        splitter.addWidget(self.preview)

        # === 右パネル（操作）===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # ROI設定
        roi_group = QGroupBox("位置合わせ領域")
        roi_layout = QVBoxLayout(roi_group)

        self.btn_roi_select = QPushButton("領域を選択...")
        self.btn_roi_select.clicked.connect(self._start_roi_select)
        roi_layout.addWidget(self.btn_roi_select)

        self.lbl_roi = QLabel("未設定（全体）")
        self.lbl_roi.setStyleSheet("color: #888;")
        roi_layout.addWidget(self.lbl_roi)

        self.btn_roi_clear = QPushButton("クリア")
        self.btn_roi_clear.clicked.connect(self._clear_roi)
        self.btn_roi_clear.setEnabled(False)
        roi_layout.addWidget(self.btn_roi_clear)

        right_layout.addWidget(roi_group)

        # 位置合わせ
        align_group = QGroupBox("位置合わせ")
        align_layout = QVBoxLayout(align_group)

        self.btn_align_current = QPushButton("選択画像を位置合わせ")
        self.btn_align_current.clicked.connect(self._align_current)
        align_layout.addWidget(self.btn_align_current)

        self.btn_align_all = QPushButton("全画像を一括位置合わせ")
        self.btn_align_all.setStyleSheet("background-color: #2563eb; color: white;")
        self.btn_align_all.clicked.connect(self._align_all)
        align_layout.addWidget(self.btn_align_all)

        retry_threshold_layout = QHBoxLayout()
        retry_threshold_layout.addWidget(QLabel("再試行しきい値:"))
        self.spin_retry_threshold = QDoubleSpinBox()
        self.spin_retry_threshold.setRange(0.0, 1.0)
        self.spin_retry_threshold.setSingleStep(0.05)
        self.spin_retry_threshold.setDecimals(2)
        self.spin_retry_threshold.setValue(max(0.6, self.aligner.config.success_score_threshold))
        self.spin_retry_threshold.valueChanged.connect(self._on_retry_threshold_changed)
        retry_threshold_layout.addWidget(self.spin_retry_threshold)
        align_layout.addLayout(retry_threshold_layout)

        self.btn_align_low_score = QPushButton("低スコアのみ再試行")
        self.btn_align_low_score.clicked.connect(self._align_low_score_only)
        align_layout.addWidget(self.btn_align_low_score)

        self.btn_next_issue = QPushButton("次の要調整画像へ")
        self.btn_next_issue.clicked.connect(self._jump_to_next_issue)
        align_layout.addWidget(self.btn_next_issue)

        self.btn_set_center_roi = QPushButton("中央ROIを自動設定")
        self.btn_set_center_roi.clicked.connect(self._set_center_roi)
        align_layout.addWidget(self.btn_set_center_roi)

        self.lbl_align_status = QLabel("")
        self.lbl_align_status.setWordWrap(True)
        align_layout.addWidget(self.lbl_align_status)

        right_layout.addWidget(align_group)

        # ステータス
        status_group = QGroupBox("ステータス")
        status_layout = QVBoxLayout(status_group)
        self.lbl_status_summary = QLabel("未実行")
        self.lbl_status_summary.setStyleSheet("color: #888;")
        self.lbl_status_summary.setWordWrap(True)
        status_layout.addWidget(self.lbl_status_summary)
        self.lbl_status_detail = QLabel("詳細: 画像を選択してください")
        self.lbl_status_detail.setStyleSheet("color: #888;")
        self.lbl_status_detail.setWordWrap(True)
        status_layout.addWidget(self.lbl_status_detail)
        right_layout.addWidget(status_group)

        # 表示
        view_group = QGroupBox("表示")
        view_layout = QVBoxLayout(view_group)

        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        view_layout.addWidget(self.lbl_zoom)

        zoom_btn_layout = QHBoxLayout()
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(40)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        zoom_btn_layout.addWidget(self.btn_zoom_out)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(40)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        zoom_btn_layout.addWidget(self.btn_zoom_in)
        view_layout.addLayout(zoom_btn_layout)

        self.btn_zoom_fit = QPushButton("ウィンドウに合わせる")
        self.btn_zoom_fit.clicked.connect(self._zoom_fit)
        view_layout.addWidget(self.btn_zoom_fit)

        right_layout.addWidget(view_group)

        # 保存
        save_group = QGroupBox("保存")
        save_layout = QVBoxLayout(save_group)

        self.check_auto_trim = QCheckBox("位置合わせ余白を自動トリミング")
        self.check_auto_trim.setChecked(True)
        self.check_auto_trim.toggled.connect(self._update_trim_info_label)
        save_layout.addWidget(self.check_auto_trim)

        trim_margin_layout = QHBoxLayout()
        trim_margin_layout.addWidget(QLabel("マージン:"))
        self.spin_trim_margin = QSpinBox()
        self.spin_trim_margin.setRange(0, 100)
        self.spin_trim_margin.setValue(0)
        self.spin_trim_margin.setSuffix("px")
        self.spin_trim_margin.valueChanged.connect(self._update_trim_info_label)
        trim_margin_layout.addWidget(self.spin_trim_margin)
        trim_margin_layout.addStretch()
        save_layout.addLayout(trim_margin_layout)

        self.lbl_trim_info = QLabel("トリミング範囲: 未計算")
        self.lbl_trim_info.setStyleSheet("color: #888;")
        save_layout.addWidget(self.lbl_trim_info)

        self.btn_save_current = QPushButton("選択画像を保存...")
        self.btn_save_current.clicked.connect(self._save_current)
        save_layout.addWidget(self.btn_save_current)

        self.btn_save_all = QPushButton("全画像を一括保存...")
        self.btn_save_all.setStyleSheet("background-color: #16a34a; color: white;")
        self.btn_save_all.clicked.connect(self._save_all)
        save_layout.addWidget(self.btn_save_all)

        right_layout.addWidget(save_group)

        right_layout.addStretch()

        splitter.addWidget(right_panel)

        splitter.setSizes([200, 600, 200])
        layout.addWidget(splitter)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return

        paths = [url.toLocalFile() for url in urls
                 if url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]

        if not paths:
            return

        # ベース画像が未設定なら最初の画像をベースに
        if self.base_image is None:
            first_path = paths[0]
            self.base_image = load_image(first_path)
            if self.base_image is not None:
                self.base_path = first_path
                self._last_base_dir = str(Path(first_path).parent)
                self.lbl_base.setText(Path(first_path).name)
                self.lbl_base.setStyleSheet("color: #4ade80;")
                self.preview.set_base_image(self.base_image)
                self.preview.fit_to_window()
            paths = paths[1:]  # 残りを差分として追加

        # 残りを差分画像として追加
        for path in paths:
            item = ImageItem(path)
            item.image = load_image(path)
            if item.image is not None:
                self.images.append(item)
                list_item = QListWidgetItem(self._format_list_item_text(item))
                self.image_list.addItem(list_item)

        if paths:
            self._last_add_dir = str(Path(paths[0]).parent)

        self._update_trim_info_label()
        self._update_status_summary()

    def _load_base_image(self):
        start_dir = self._last_base_dir or self._last_add_dir or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "ベース画像を選択", start_dir, "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.base_image = load_image(path)
            if self.base_image is not None:
                self.base_path = path
                self._last_base_dir = str(Path(path).parent)
                self.lbl_base.setText(Path(path).name)
                self.lbl_base.setStyleSheet("color: #4ade80;")
                self.preview.set_base_image(self.base_image)
                self.preview.fit_to_window()
                self._update_trim_info_label()
                self._update_status_summary()

    def _add_images(self):
        start_dir = self._last_add_dir or self._last_base_dir or ""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "差分画像を追加", start_dir, "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        for path in paths:
            item = ImageItem(path)
            item.image = load_image(path)
            if item.image is not None:
                self.images.append(item)
                list_item = QListWidgetItem(self._format_list_item_text(item))
                self.image_list.addItem(list_item)
        if paths:
            self._last_add_dir = str(Path(paths[0]).parent)
        self._update_trim_info_label()
        self._update_status_summary()

    def _clear_images(self):
        self.images.clear()
        self.image_list.clear()
        self.current_index = -1
        self.preview.set_overlay_image(None)
        self.lbl_align_status.setText("")
        self._update_trim_info_label()
        self._update_status_summary()

    def _on_image_selected(self, index: int):
        self.current_index = index
        if 0 <= index < len(self.images):
            item = self.images[index]
            # 位置合わせ済みがあればそれを表示、なければ元画像
            if item.aligned_image is not None:
                self.preview.set_overlay_image(item.aligned_image)
            else:
                self.preview.set_overlay_image(item.image)
            self._update_current_detail(item)
        else:
            self.lbl_status_detail.setText("詳細: 画像を選択してください")
            self.lbl_status_detail.setStyleSheet("color: #888;")

    def _format_list_item_text(self, item: ImageItem) -> str:
        """画像リスト表示テキストを生成"""
        if item.aligned_image is None:
            return f"○ {item.name}"
        status = "✓" if item.alignment_success else "⚠"
        method = item.alignment_method if item.alignment_method else "-"
        return f"{status} [{item.alignment_score:.2f}/{method}] {item.name}"

    def _update_list_item(self, index: int):
        """指定行の表示を更新"""
        if not (0 <= index < len(self.images)):
            return
        list_item = self.image_list.item(index)
        if list_item is None:
            return
        list_item.setText(self._format_list_item_text(self.images[index]))

    def _update_current_detail(self, item: ImageItem):
        """選択中画像の詳細を更新"""
        if item.aligned_image is None:
            self.lbl_status_detail.setText("詳細: 未実行")
            self.lbl_status_detail.setStyleSheet("color: #888;")
            return

        method = item.alignment_method or "-"
        match_info = f"{item.alignment_inliers}/{item.alignment_total_matches}" if item.alignment_total_matches > 0 else "-"
        threshold = self.spin_retry_threshold.value() if hasattr(self, "spin_retry_threshold") else 0.6
        if item.alignment_success and item.alignment_score >= threshold:
            color = "#4ade80"
            head = "成功"
        elif item.alignment_success:
            color = "#fbbf24"
            head = "低スコア（再調整推奨）"
        else:
            color = "#f87171"
            head = "要調整"

        detail = f"詳細: {head} / score={item.alignment_score:.2f} / {method} / inlier={match_info}"
        if item.alignment_error:
            detail += f"\n理由: {item.alignment_error}"
        self.lbl_status_detail.setText(detail)
        self.lbl_status_detail.setStyleSheet(f"color: {color};")

    def _update_status_summary(self):
        """全体ステータス表示を更新"""
        total = len(self.images)
        if total == 0:
            self.lbl_status_summary.setText("未実行（差分画像を追加してください）")
            self.lbl_status_summary.setStyleSheet("color: #888;")
            return

        aligned_items = [item for item in self.images if item.aligned_image is not None]
        if not aligned_items:
            self.lbl_status_summary.setText(f"未実行: 0/{total}")
            self.lbl_status_summary.setStyleSheet("color: #fbbf24;")
            return

        success_count = sum(1 for item in aligned_items if item.alignment_success)
        retry_threshold = self.spin_retry_threshold.value() if hasattr(self, "spin_retry_threshold") else 0.6
        low_score_count = sum(1 for item in aligned_items if item.alignment_score < retry_threshold)
        avg_score = sum(item.alignment_score for item in aligned_items) / len(aligned_items)

        color = "#4ade80" if success_count == len(aligned_items) else "#fbbf24"
        if success_count == 0:
            color = "#f87171"

        self.lbl_status_summary.setText(
            f"実行済み: {len(aligned_items)}/{total} / 成功: {success_count} / 要調整: {low_score_count}\n"
            f"平均スコア: {avg_score:.2f}（再試行しきい値 {retry_threshold:.2f}）"
        )
        self.lbl_status_summary.setStyleSheet(f"color: {color};")

    def _on_retry_threshold_changed(self, value: float):
        self._update_status_summary()
        if 0 <= self.current_index < len(self.images):
            self._update_current_detail(self.images[self.current_index])

    def _start_roi_select(self):
        if self.base_image is None:
            QMessageBox.warning(self, "警告", "先にベース画像を選択してください")
            return
        self.preview.set_roi_mode(True)
        self.statusBar().showMessage("位置合わせ領域をドラッグして選択...")

    def _on_roi_selected(self, x: int, y: int, w: int, h: int):
        self.preview.set_roi_mode(False)

        # 最小サイズチェック
        if w < 50 or h < 50:
            QMessageBox.warning(self, "警告", f"領域が小さすぎます（最小50x50px）: {w}x{h}")
            return

        # 境界クランプ
        if self.base_image is not None:
            img_h, img_w = self.base_image.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)

        self.roi = (x, y, w, h)
        self.preview.set_roi(x, y, w, h)
        self.lbl_roi.setText(f"({x}, {y}) - {w}x{h}")
        self.lbl_roi.setStyleSheet("color: #00a2e8;")
        self.btn_roi_clear.setEnabled(True)
        self.statusBar().showMessage(f"位置合わせ領域: ({x}, {y}) - {w}x{h}")

    def _clear_roi(self):
        self.roi = None
        self.preview.clear_roi()
        self.lbl_roi.setText("未設定（全体）")
        self.lbl_roi.setStyleSheet("color: #888;")
        self.btn_roi_clear.setEnabled(False)

    def _set_center_roi(self):
        """顔領域を想定した中央ROIを自動設定"""
        if self.base_image is None:
            QMessageBox.warning(self, "警告", "先にベース画像を選択してください")
            return

        img_h, img_w = self.base_image.shape[:2]
        roi_w = max(50, int(img_w * 0.7))
        roi_h = max(50, int(img_h * 0.7))
        x = max(0, (img_w - roi_w) // 2)
        y = max(0, (img_h - roi_h) // 2)
        self.roi = (x, y, roi_w, roi_h)
        self.preview.set_roi(x, y, roi_w, roi_h)
        self.lbl_roi.setText(f"({x}, {y}) - {roi_w}x{roi_h} [中央自動]")
        self.lbl_roi.setStyleSheet("color: #00a2e8;")
        self.btn_roi_clear.setEnabled(True)
        self.statusBar().showMessage("中央ROIを設定しました。再試行してください。")

    def _collect_issue_indices(self, threshold: Optional[float] = None) -> List[int]:
        """再調整が必要な画像インデックスを収集"""
        if threshold is None:
            threshold = self.spin_retry_threshold.value()

        issue_indices: List[int] = []
        for i, item in enumerate(self.images):
            if item.aligned_image is None:
                issue_indices.append(i)
                continue
            if not item.alignment_success or item.alignment_score < threshold:
                issue_indices.append(i)
        return issue_indices

    def _jump_to_next_issue(self):
        """次の要調整画像へジャンプ"""
        if not self.images:
            QMessageBox.information(self, "情報", "差分画像がありません")
            return

        issue_indices = self._collect_issue_indices()
        if not issue_indices:
            QMessageBox.information(self, "情報", "要調整画像はありません")
            return

        start = self.current_index + 1 if self.current_index >= 0 else 0
        ordered = issue_indices + issue_indices
        target = None
        for idx in ordered:
            if idx >= start:
                target = idx
                break
        if target is None:
            target = issue_indices[0]

        self.image_list.setCurrentRow(target)
        self.statusBar().showMessage(f"要調整画像へ移動: {self.images[target].name}")

    def _align_low_score_only(self):
        """低スコア・失敗画像のみ再試行"""
        if self.base_image is None:
            QMessageBox.warning(self, "警告", "ベース画像を選択してください")
            return

        issue_indices = self._collect_issue_indices(self.spin_retry_threshold.value())
        if not issue_indices:
            QMessageBox.information(self, "情報", "再試行が必要な画像はありません")
            return

        progress = QProgressDialog("再試行中...", "キャンセル", 0, len(issue_indices), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        success_count = 0
        for pos, idx in enumerate(issue_indices):
            if progress.wasCanceled():
                break
            progress.setValue(pos)
            item = self.images[idx]
            progress.setLabelText(f"再試行: {item.name}")
            success = self._align_image(item)
            self._update_list_item(idx)
            if success:
                success_count += 1

        progress.setValue(len(issue_indices))
        self._update_trim_info_label()
        self._update_status_summary()
        if 0 <= self.current_index < len(self.images):
            self._update_current_detail(self.images[self.current_index])

        QMessageBox.information(
            self, "完了",
            f"低スコア再試行完了\n成功: {success_count}/{len(issue_indices)}\n"
            f"しきい値: {self.spin_retry_threshold.value():.2f}"
        )

    def _zoom_in(self):
        self.preview.scale = min(5.0, self.preview.scale * 1.25)
        self._update_zoom_label()
        self.preview._update_display()

    def _zoom_out(self):
        self.preview.scale = max(0.1, self.preview.scale / 1.25)
        self._update_zoom_label()
        self.preview._update_display()

    def _zoom_fit(self):
        self.preview.fit_to_window()
        self._update_zoom_label()

    def _update_zoom_label(self):
        self.lbl_zoom.setText(f"{int(self.preview.scale * 100)}%")

    def _get_trim_rect(self, target_items: List[ImageItem]) -> Optional[tuple]:
        """指定画像群に対する共通トリミング矩形を計算"""
        if not self.check_auto_trim.isChecked() or self.base_image is None:
            return None

        base_h, base_w = self.base_image.shape[:2]
        valid_masks = [np.full((base_h, base_w), 255, dtype=np.uint8)]

        for item in target_items:
            if item.aligned_image is None:
                continue
            if item.aligned_image.shape[:2] != (base_h, base_w):
                continue
            if item.aligned_valid_mask is not None:
                valid_masks.append(item.aligned_valid_mask)
            else:
                valid_masks.append(np.full((base_h, base_w), 255, dtype=np.uint8))

        return compute_common_valid_rect(valid_masks, margin=self.spin_trim_margin.value())

    def _update_trim_info_label(self):
        """トリミング設定ラベルを更新"""
        if not hasattr(self, "lbl_trim_info"):
            return

        if not self.check_auto_trim.isChecked():
            self.lbl_trim_info.setText("トリミング範囲: OFF")
            self.lbl_trim_info.setStyleSheet("color: #888;")
            return

        aligned_items = [item for item in self.images if item.aligned_image is not None]
        rect = self._get_trim_rect(aligned_items)
        if rect is None:
            self.lbl_trim_info.setText("トリミング範囲: 計算不可")
            self.lbl_trim_info.setStyleSheet("color: #fbbf24;")
            return

        if self.base_image is not None:
            base_h, base_w = self.base_image.shape[:2]
            if rect == (0, 0, base_w, base_h):
                self.lbl_trim_info.setText("トリミング範囲: 全体（余白なし）")
                self.lbl_trim_info.setStyleSheet("color: #4ade80;")
                return

        x, y, w, h = rect
        self.lbl_trim_info.setText(f"トリミング範囲: x={x}, y={y}, w={w}, h={h}")
        self.lbl_trim_info.setStyleSheet("color: #60a5fa;")

    def _create_roi_mask(self) -> Optional[np.ndarray]:
        if self.roi is None or self.base_image is None:
            return None

        x, y, w, h = self.roi
        img_h, img_w = self.base_image.shape[:2]
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255
        return mask

    def _align_image(self, item: ImageItem) -> bool:
        if self.base_image is None or item.image is None:
            return False

        mask = self._create_roi_mask()
        result = self.aligner.align(self.base_image, item.image, base_mask=mask)

        item.alignment_success = bool(result['success'])
        item.alignment_score = result['score']
        item.alignment_method = result.get('method', '')
        item.alignment_inliers = int(result.get('inliers', 0))
        item.alignment_total_matches = int(result.get('total_matches', 0))
        item.alignment_error = result.get('error_message', '')

        if result['matrix'] is not None:
            h, w = self.base_image.shape[:2]
            aligned, valid_mask = self.aligner.apply_transform_with_mask(
                item.image, result['matrix'], (w, h)
            )
            item.aligned_image = aligned
            item.aligned_valid_mask = valid_mask
        else:
            item.aligned_image = item.image.copy()
            item.aligned_valid_mask = np.full(item.image.shape[:2], 255, dtype=np.uint8)

        return item.alignment_success

    def _align_current(self):
        if self.base_image is None:
            QMessageBox.warning(self, "警告", "ベース画像を選択してください")
            return

        if self.current_index < 0 or self.current_index >= len(self.images):
            QMessageBox.warning(self, "警告", "画像を選択してください")
            return

        item = self.images[self.current_index]
        success = self._align_image(item)

        # リスト更新
        self._update_list_item(self.current_index)

        # プレビュー更新
        self.preview.set_overlay_image(item.aligned_image)

        method = item.alignment_method or "-"
        self.lbl_align_status.setText(
            f"{'成功' if success else '要調整'} (score={item.alignment_score:.2f}, method={method})"
        )
        if success:
            self.lbl_align_status.setStyleSheet("color: #4ade80;")
        elif item.alignment_score >= self.spin_retry_threshold.value():
            self.lbl_align_status.setStyleSheet("color: #fbbf24;")
        else:
            self.lbl_align_status.setStyleSheet("color: #f87171;")

        if not success:
            self.statusBar().showMessage("要調整: ROIを顔中心に絞って再試行、または「低スコアのみ再試行」を実行してください")
        else:
            self.statusBar().showMessage("位置合わせ成功")

        self._update_current_detail(item)
        self._update_trim_info_label()
        self._update_status_summary()

    def _align_all(self):
        if self.base_image is None:
            QMessageBox.warning(self, "警告", "ベース画像を選択してください")
            return

        if len(self.images) == 0:
            QMessageBox.warning(self, "警告", "画像を追加してください")
            return

        progress = QProgressDialog("位置合わせ中...", "キャンセル", 0, len(self.images), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        success_count = 0
        for i, item in enumerate(self.images):
            if progress.wasCanceled():
                break

            progress.setValue(i)
            progress.setLabelText(f"位置合わせ中: {item.name}")

            success = self._align_image(item)
            if success:
                success_count += 1

            # リスト更新
            self._update_list_item(i)

        progress.setValue(len(self.images))

        # 現在選択中の画像を更新
        if 0 <= self.current_index < len(self.images):
            item = self.images[self.current_index]
            self.preview.set_overlay_image(item.aligned_image)

        QMessageBox.information(
            self, "完了",
            f"一括位置合わせ完了\n{success_count}/{len(self.images)}件成功"
        )
        self.statusBar().showMessage(
            f"一括位置合わせ完了: 成功 {success_count}/{len(self.images)} / しきい値 {self.spin_retry_threshold.value():.2f}"
        )
        if 0 <= self.current_index < len(self.images):
            self._update_current_detail(self.images[self.current_index])
        self._update_trim_info_label()
        self._update_status_summary()

    def _save_current(self):
        if self.current_index < 0 or self.current_index >= len(self.images):
            QMessageBox.warning(self, "警告", "画像を選択してください")
            return

        item = self.images[self.current_index]
        if item.aligned_image is None:
            QMessageBox.warning(self, "警告", "先に位置合わせを実行してください")
            return

        # ベース画像名_連番形式: tomari_01.png, tomari_02.png...
        base_name = Path(self.base_path).stem if self.base_path else "output"
        num = self.current_index + 1
        default_name = f"{base_name}_{num:02d}.png"
        default_path = str(Path(self._last_save_dir) / default_name) if self._last_save_dir else default_name
        path, _ = QFileDialog.getSaveFileName(
            self, "保存先を選択", default_path, "PNG (*.png)"
        )
        if path:
            try:
                trim_rect = self._get_trim_rect([item])
                if self.check_auto_trim.isChecked() and trim_rect is None:
                    QMessageBox.warning(self, "警告", "トリミング範囲を計算できませんでした。")
                    return

                if trim_rect is not None and self.base_image is not None:
                    base_h, base_w = self.base_image.shape[:2]
                    if item.aligned_image.shape[:2] != (base_h, base_w):
                        QMessageBox.warning(self, "警告", "この画像はベースサイズと異なるためトリミングできません。")
                        return
                    output = crop_image(item.aligned_image, trim_rect)
                else:
                    output = item.aligned_image

                ok = save_image(path, output)
                if not ok:
                    raise RuntimeError("save_image returned False")
                self._last_save_dir = str(Path(path).parent)
                QMessageBox.information(self, "完了", f"保存しました:\n{path}")
            except Exception as e:
                QMessageBox.warning(self, "エラー", f"保存に失敗しました:\n{e}")

    def _save_all(self):
        if self.base_image is None:
            QMessageBox.warning(self, "警告", "ベース画像がありません")
            return

        aligned_count = sum(1 for item in self.images if item.aligned_image is not None)

        if aligned_count == 0:
            QMessageBox.warning(self, "警告", "位置合わせ済みの画像がありません")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "保存先フォルダを選択", self._last_save_dir or self._last_base_dir or ""
        )
        if not output_dir:
            return
        self._last_save_dir = output_dir

        output_path = Path(output_dir)
        saved = 0

        # ベース画像名_連番形式
        base_name = Path(self.base_path).stem if self.base_path else "output"

        trim_rect = self._get_trim_rect([item for item in self.images if item.aligned_image is not None])
        if self.check_auto_trim.isChecked() and trim_rect is None:
            QMessageBox.warning(self, "警告", "トリミング範囲を計算できませんでした。")
            return

        base_h, base_w = self.base_image.shape[:2]
        trim_skipped = []

        # ベース画像を00として保存
        base_filename = f"{base_name}_00.png"
        base_output = crop_image(self.base_image, trim_rect) if trim_rect is not None else self.base_image
        if not save_image(str(output_path / base_filename), base_output):
            QMessageBox.warning(self, "エラー", f"{base_filename} の保存に失敗しました")
            return
        saved += 1

        # 差分画像を01, 02, 03...として保存
        for i, item in enumerate(self.images):
            if item.aligned_image is not None:
                num = i + 1
                filename = f"{base_name}_{num:02d}.png"
                save_path = output_path / filename
                output = item.aligned_image
                if trim_rect is not None:
                    if item.aligned_image.shape[:2] == (base_h, base_w):
                        output = crop_image(item.aligned_image, trim_rect)
                    else:
                        trim_skipped.append(filename)
                if not save_image(str(save_path), output):
                    QMessageBox.warning(self, "エラー", f"{filename} の保存に失敗しました")
                    return
                saved += 1

        trim_note = ""
        if trim_rect is not None:
            x, y, w, h = trim_rect
            trim_note = f"\n・トリミング: x={x}, y={y}, w={w}, h={h}"
            if trim_skipped:
                trim_note += f"\n・トリミング未適用: {', '.join(trim_skipped)}"

        QMessageBox.information(
            self, "完了",
            f"{saved}件の画像を保存しました:\n{output_dir}\n\n"
            f"・{base_name}_00.png (ベース)\n"
            f"・{base_name}_01.png ～ (差分)"
            f"{trim_note}"
        )

    def _load_settings(self):
        """保存済み設定を読み込む"""
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        splitter_sizes = self.settings.value("window/splitter_sizes")
        if splitter_sizes:
            try:
                sizes = [int(v) for v in splitter_sizes]
                if hasattr(self, "main_splitter"):
                    self.main_splitter.setSizes(sizes)
            except Exception:
                pass

        self._last_base_dir = self.settings.value("paths/base_dir", "", type=str)
        self._last_add_dir = self.settings.value("paths/add_dir", "", type=str)
        self._last_save_dir = self.settings.value("paths/save_dir", "", type=str)

        self.check_auto_trim.setChecked(self.settings.value("align/auto_trim", True, type=bool))
        self.spin_trim_margin.setValue(self.settings.value("align/trim_margin", 0, type=int))
        self.spin_retry_threshold.setValue(self.settings.value("align/retry_threshold", 0.6, type=float))

    def _save_settings(self):
        """設定を保存"""
        self.settings.setValue("window/geometry", self.saveGeometry())
        if hasattr(self, "main_splitter"):
            self.settings.setValue("window/splitter_sizes", self.main_splitter.sizes())

        self.settings.setValue("paths/base_dir", self._last_base_dir)
        self.settings.setValue("paths/add_dir", self._last_add_dir)
        self.settings.setValue("paths/save_dir", self._last_save_dir)

        self.settings.setValue("align/auto_trim", self.check_auto_trim.isChecked())
        self.settings.setValue("align/trim_margin", self.spin_trim_margin.value())
        self.settings.setValue("align/retry_threshold", self.spin_retry_threshold.value())

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Simple Aligner")
    app.setStyle("Fusion")

    # ダークテーマ
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1e1e1e;
        }
        QWidget {
            background-color: #252526;
            color: #cccccc;
        }
        QPushButton {
            background-color: #0e639c;
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #1177bb;
        }
        QPushButton:pressed {
            background-color: #094771;
        }
        QPushButton:disabled {
            background-color: #3c3c3c;
            color: #666;
        }
        QGroupBox {
            border: 1px solid #3e3e42;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QListWidget {
            background-color: #1e1e1e;
            border: 1px solid #3e3e42;
        }
        QListWidget::item {
            padding: 5px;
        }
        QListWidget::item:selected {
            background-color: #094771;
        }
        QLabel {
            color: #cccccc;
        }
        QScrollArea {
            background-color: #1e1e1e;
            border: none;
        }
        QStatusBar {
            background-color: #007acc;
            color: white;
        }
        QProgressDialog {
            background-color: #252526;
        }
    """)

    window = SimpleAlignerWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
