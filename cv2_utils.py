"""
OpenCVユーティリティ
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


def load_image_as_bgra(path: str) -> np.ndarray:
    """画像をBGRAとして読み込み（アルファなしは255で補完）

    Args:
        path: 画像ファイルパス

    Returns:
        BGRA画像 (uint8)
    """
    # 日本語パス対応: np.fromfile + imdecode
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except (OSError, IOError) as e:
        raise ValueError(f"Failed to load image: {path}") from e
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    
    if img.ndim == 2:  # グレースケール
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:  # BGR（アルファなし）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img[:, :, 3] = 255  # 不透明アルファ
    elif img.shape[2] == 4:  # BGRA
        pass  # そのまま
    else:
        raise ValueError(f"Unsupported image format: {img.shape}")
    
    return img


def bgra_to_qimage(bgra: np.ndarray) -> 'QImage':
    """BGRA→QImage（Format_RGBA8888、ストレートアルファ）
    
    Args:
        bgra: BGRA画像 (uint8)
        
    Returns:
        QImage (RGBA8888形式)
    """
    from PySide6.QtGui import QImage
    
    rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
    h, w, ch = rgba.shape
    return QImage(rgba.data, w, h, w * ch, QImage.Format.Format_RGBA8888).copy()


def load_image(filepath: str, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """
    画像を読み込み

    Args:
        filepath: ファイルパス
        flags: OpenCV読み込みフラグ

    Returns:
        画像配列、失敗時はNone
    """
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}")
        return None

    # 日本語パス対応: np.fromfile + imdecode
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, IOError) as e:
        print(f"Failed to read file: {filepath} ({e})")
        return None
    image = cv2.imdecode(buf, flags)
    if image is None:
        print(f"Failed to load image: {filepath}")
        return None

    return image


def save_image(filepath: str, image: np.ndarray) -> bool:
    """
    画像を保存

    Args:
        filepath: ファイルパス
        image: 画像配列

    Returns:
        成功時True
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 日本語パス対応: imencode + tofile
        ext = path.suffix.lower()
        if ext in ('.jpg', '.jpeg'):
            ext = '.jpg'
        elif ext == '':
            # 拡張子なしの場合は.pngを付与
            ext = '.png'
            path = path.with_suffix('.png')
        success, buf = cv2.imencode(ext, image)
        if success:
            buf.tofile(str(path))
            return True
        return False
    except Exception as e:
        print(f"Save error: {e}")
        return False


def compute_common_valid_rect(
    valid_masks: List[np.ndarray],
    margin: int = 0
) -> Optional[Tuple[int, int, int, int]]:
    """有効領域マスク群の共通矩形を計算

    Args:
        valid_masks: 2Dマスクのリスト（255=有効）
        margin: 矩形に追加する外側マージン（px）

    Returns:
        (x, y, w, h)。共通領域がない場合はNone。
    """
    normalized_masks = [m for m in valid_masks if m is not None and m.ndim == 2]
    if not normalized_masks:
        return None

    ref_h, ref_w = normalized_masks[0].shape[:2]
    intersection = np.full((ref_h, ref_w), 255, dtype=np.uint8)

    for mask in normalized_masks:
        if mask.shape[:2] != (ref_h, ref_w):
            return None
        intersection = cv2.bitwise_and(
            intersection,
            np.where(mask > 0, 255, 0).astype(np.uint8)
        )

    ys, xs = np.where(intersection > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    margin = max(0, int(margin))
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(ref_w - 1, x_max + margin)
    y_max = min(ref_h - 1, y_max + margin)

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    if width <= 0 or height <= 0:
        return None

    return (x_min, y_min, width, height)


def crop_image(image: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    """画像を矩形で切り抜く"""
    x, y, w, h = rect
    img_h, img_w = image.shape[:2]

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, x1 + int(w))
    y2 = min(img_h, y1 + int(h))

    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid crop rect: {rect} for image size {img_w}x{img_h}")

    return image[y1:y2, x1:x2].copy()


def resize_image(image: np.ndarray, 
                 target_size: Optional[Tuple[int, int]] = None,
                 max_size: Optional[int] = None,
                 interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """
    画像をリサイズ
    
    Args:
        image: 入力画像
        target_size: 目標サイズ (w, h)
        max_size: 最大辺長（target_sizeがNoneの場合使用）
        interpolation: 補間方法
    
    Returns:
        リサイズ後画像
    """
    if target_size is not None:
        return cv2.resize(image, target_size, interpolation=interpolation)
    
    if max_size is not None:
        h, w = image.shape[:2]
        scale = min(max_size / max(h, w), 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    return image


def convert_to_qimage(image: np.ndarray) -> 'QImage':
    """
    OpenCV画像をPySide6 QImageに変換
    
    Args:
        image: BGRまたはBGRA画像
    
    Returns:
        QImage
    """
    from PySide6.QtGui import QImage
    
    if len(image.shape) == 2:
        # グレースケール
        height, width = image.shape
        bytes_per_line = width
        return QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8).copy()
    
    height, width, channels = image.shape
    
    if channels == 3:
        # BGR -> RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * width
        return QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
    
    elif channels == 4:
        # BGRA -> RGBA
        rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        bytes_per_line = 4 * width
        return QImage(rgba_image.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888).copy()
    
    else:
        raise ValueError(f"Unsupported channel count: {channels}")


def convert_from_qimage(qimage: 'QImage') -> np.ndarray:
    """
    PySide6 QImageをOpenCV画像に変換
    """
    import numpy as np
    
    width = qimage.width()
    height = qimage.height()
    
    ptr = qimage.bits()
    ptr.setsize(height * width * 4)
    
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
    
    # RGBA -> BGRA
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


def create_checkerboard(size: Tuple[int, int], 
                        checker_size: int = 20,
                        color1: Tuple[int, int, int] = (200, 200, 200),
                        color2: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    チェッカーボードパターンを作成（透明背景プレビュー用）
    """
    h, w = size
    result = np.zeros((h, w, 3), dtype=np.uint8)
    
    for y in range(0, h, checker_size):
        for x in range(0, w, checker_size):
            color = color1 if ((x // checker_size) + (y // checker_size)) % 2 == 0 else color2
            result[y:y+checker_size, x:x+checker_size] = color
    
    return result


def composite_images(background: np.ndarray, 
                    foreground: np.ndarray,
                    alpha: float = 1.0) -> np.ndarray:
    """
    画像を合成（前景はRGBA想定）
    """
    # 前景をリサイズ
    h, w = background.shape[:2]
    foreground_resized = cv2.resize(foreground, (w, h))
    
    # アルファチャンネル抽出
    if foreground_resized.shape[2] == 4:
        fg_alpha = foreground_resized[:, :, 3].astype(float) / 255.0 * alpha
        fg_rgb = foreground_resized[:, :, :3].astype(float)
    else:
        fg_alpha = np.full((h, w), alpha)
        fg_rgb = foreground_resized.astype(float)
    
    # 背景をfloatに
    if len(background.shape) == 3:
        bg_rgb = background[:, :, :3].astype(float)
    else:
        bg_rgb = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR).astype(float)
    
    # アルファ合成
    result = np.zeros_like(bg_rgb)
    for c in range(3):
        result[:, :, c] = bg_rgb[:, :, c] * (1 - fg_alpha) + fg_rgb[:, :, c] * fg_alpha
    
    return result.astype(np.uint8)
