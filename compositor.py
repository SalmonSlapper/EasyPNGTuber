"""
画像合成プロセッサー
マスク適用・フェザリング・合成処理を提供
"""
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class CompositeConfig:
    """合成設定"""
    feather_width: int = 10           # フェザリング幅（ピクセル）


class Compositor:
    """画像合成クラス（BGRA専用、ストレートアルファ）"""

    def __init__(self, config: CompositeConfig = None):
        self.config = config or CompositeConfig()

    def feather_mask(self, mask: np.ndarray, width: int) -> np.ndarray:
        """2Dマスクにフェザリング適用（GaussianBlur使用）

        Args:
            mask: 2D uint8 (0 or 255)
            width: フェザリング幅
            
        Returns:
            2D uint8 (0-255のグラデーション)
        """
        if width <= 0:
            return mask.copy()
        ksize = width * 2 + 1  # 奇数カーネル
        return cv2.GaussianBlur(mask, (ksize, ksize), 0)

    def apply_mask_to_diff(
        self,
        diff_image: np.ndarray,  # BGRA
        mask: np.ndarray,        # 2D uint8 (0 or 255)
        feather_width: int = 10
    ) -> np.ndarray:  # BGRA
        """差分画像にマスクを適用（マスク外を透明化）

        Args:
            diff_image: 差分画像 (BGRA, uint8)
            mask: マスク画像 (2D, uint8, 0 or 255)
            feather_width: フェザリング幅
            
        Returns:
            マスク適用済み差分画像 (BGRA, uint8)
        """
        assert mask.shape == diff_image.shape[:2], \
            f"Mask shape {mask.shape} doesn't match image shape {diff_image.shape[:2]}"
        
        # フェザリング適用
        feathered = self.feather_mask(mask, feather_width)
        
        # アルファチャンネルに乗算
        result = diff_image.copy()
        alpha_float = feathered.astype(np.float32) / 255.0
        result[:, :, 3] = (result[:, :, 3].astype(np.float32) * alpha_float).astype(np.uint8)
        
        return result

    def composite(
        self,
        base_image: np.ndarray,    # BGRA (ストレートアルファ)
        masked_diff: np.ndarray    # BGRA (ストレートアルファ)
    ) -> np.ndarray:  # BGRA
        """ベースに差分を合成（Porter-Duff Over演算）

        ストレートアルファのPorter-Duff Over:
        α_out = α_diff + α_base * (1 - α_diff)
        C_out = (C_diff * α_diff + C_base * α_base * (1 - α_diff)) / α_out

        ※α_out == 0 のピクセルは黒(0,0,0,0)

        Args:
            base_image: ベース画像 (BGRA, uint8)
            masked_diff: マスク適用済み差分画像 (BGRA, uint8)
            
        Returns:
            合成画像 (BGRA, uint8)
        """
        # float32に変換し0-1正規化
        base = base_image.astype(np.float32) / 255.0
        diff = masked_diff.astype(np.float32) / 255.0
        
        # アルファチャンネル抽出
        alpha_base = base[:, :, 3:4]
        alpha_diff = diff[:, :, 3:4]
        
        # RGB抽出
        rgb_base = base[:, :, :3]
        rgb_diff = diff[:, :, :3]
        
        # α_out計算
        alpha_out = alpha_diff + alpha_base * (1 - alpha_diff)
        
        # RGB計算（ゼロ除算回避）
        rgb_out = np.zeros_like(rgb_base)
        mask = alpha_out.squeeze() > 0
        
        # α_out > 0 のピクセルのみ計算
        rgb_out[mask] = (
            rgb_diff[mask] * alpha_diff[mask] +
            rgb_base[mask] * alpha_base[mask] * (1 - alpha_diff[mask])
        ) / alpha_out[mask]
        
        # 結果を結合
        result = np.concatenate([rgb_out, alpha_out], axis=2)
        
        # uint8に戻す
        return (result * 255.0).astype(np.uint8)

    def composite_batch(
        self,
        base_image: np.ndarray,
        masked_diffs: list
    ) -> list:
        """複数の差分画像を一括合成

        Args:
            base_image: ベース画像 (BGRA, uint8)
            masked_diffs: マスク適用済み差分画像のリスト
            
        Returns:
            合成画像のリスト
        """
        results = []
        for diff in masked_diffs:
            result = self.composite(base_image, diff)
            results.append(result)
        return results
