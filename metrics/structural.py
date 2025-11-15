"""
Structural Metrics

내부 조직 구조의 품질을 평가하는 특화 지표들:
- Gradient Similarity (조직 경계 평가)
- Edge Preservation (경계 보존 정도)
- Frequency Domain Analysis (세부 구조 평가)

⭐⭐ "내부 조직의 구조가 안 좋아 보이는" 문제를 정량화하는 핵심 지표들!
"""

import numpy as np
from scipy import ndimage, fftpack
from skimage import filters
from typing import Tuple, Optional


def calculate_gradient_similarity(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    method: str = 'sobel',
) -> float:
    """
    Gradient Similarity 계산

    ⭐⭐ 조직 경계 및 내부 구조의 gradient 유사도를 평가합니다.

    조직의 경계가 흐릿하거나 내부 구조가 뭉개진 경우,
    이 지표가 낮게 나옵니다.

    Args:
        real_images: 원본 이미지 (N, H, W) or (N, H, W, C)
        synthetic_images: 합성 이미지 (N, H, W) or (N, H, W, C)
        method: gradient 계산 방법 ('sobel', 'scharr', 'prewitt')

    Returns:
        평균 gradient similarity (0~1, 높을수록 좋음)
    """
    if real_images.shape != synthetic_images.shape:
        raise ValueError(f"Shape mismatch: {real_images.shape} vs {synthetic_images.shape}")

    gradient_sims = []

    for i in range(len(real_images)):
        real_img = real_images[i]
        synth_img = synthetic_images[i]

        # 다채널인 경우 채널별 평균
        if real_img.ndim == 3:
            channel_sims = []
            for c in range(real_img.shape[2]):
                sim = _gradient_similarity_single(
                    real_img[:, :, c], synth_img[:, :, c], method
                )
                channel_sims.append(sim)
            gradient_sim = np.mean(channel_sims)
        else:
            gradient_sim = _gradient_similarity_single(real_img, synth_img, method)

        gradient_sims.append(gradient_sim)

    return np.mean(gradient_sims)


def _gradient_similarity_single(real, synth, method):
    """단일 이미지의 gradient similarity 계산"""
    # Gradient 계산
    if method == 'sobel':
        real_gx = filters.sobel_h(real)
        real_gy = filters.sobel_v(real)
        synth_gx = filters.sobel_h(synth)
        synth_gy = filters.sobel_v(synth)
    elif method == 'scharr':
        real_gx = filters.scharr_h(real)
        real_gy = filters.scharr_v(real)
        synth_gx = filters.scharr_h(synth)
        synth_gy = filters.scharr_v(synth)
    elif method == 'prewitt':
        real_gx = filters.prewitt_h(real)
        real_gy = filters.prewitt_v(real)
        synth_gx = filters.prewitt_h(synth)
        synth_gy = filters.prewitt_v(synth)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Gradient magnitude
    real_mag = np.sqrt(real_gx**2 + real_gy**2)
    synth_mag = np.sqrt(synth_gx**2 + synth_gy**2)

    # Gradient direction
    real_dir = np.arctan2(real_gy, real_gx)
    synth_dir = np.arctan2(synth_gy, synth_gx)

    # Magnitude similarity (SSIM-like)
    c1 = (0.01 * (real_mag.max() - real_mag.min())) ** 2
    mag_sim = (2 * real_mag * synth_mag + c1) / (real_mag**2 + synth_mag**2 + c1)

    # Direction similarity
    dir_diff = np.abs(real_dir - synth_dir)
    dir_diff = np.minimum(dir_diff, 2 * np.pi - dir_diff)  # Circular distance
    dir_sim = 1 - dir_diff / np.pi

    # 가중 평균
    grad_sim = 0.7 * mag_sim + 0.3 * dir_sim

    return np.mean(grad_sim)


def calculate_edge_preservation(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    sigma: float = 1.0,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
) -> float:
    """
    Edge Preservation Index 계산

    경계 보존 정도를 평가합니다.
    조직 경계가 잘 보존되는지 측정합니다.

    Args:
        real_images: 원본 이미지 (N, H, W) or (N, H, W, C)
        synthetic_images: 합성 이미지 (N, H, W) or (N, H, W, C)
        sigma: Canny edge detection의 Gaussian sigma
        low_threshold: Canny low threshold
        high_threshold: Canny high threshold

    Returns:
        평균 edge preservation (0~1, 높을수록 좋음)
    """
    from skimage.feature import canny

    if real_images.shape != synthetic_images.shape:
        raise ValueError(f"Shape mismatch: {real_images.shape} vs {synthetic_images.shape}")

    edge_preservations = []

    for i in range(len(real_images)):
        real_img = real_images[i]
        synth_img = synthetic_images[i]

        # Grayscale 변환 (필요시)
        if real_img.ndim == 3:
            real_gray = np.mean(real_img, axis=2)
            synth_gray = np.mean(synth_img, axis=2)
        else:
            real_gray = real_img
            synth_gray = synth_img

        # Edge detection
        real_edges = canny(real_gray, sigma=sigma,
                          low_threshold=low_threshold,
                          high_threshold=high_threshold)
        synth_edges = canny(synth_gray, sigma=sigma,
                           low_threshold=low_threshold,
                           high_threshold=high_threshold)

        # Edge preservation 계산
        # True Positive: 실제 edge가 합성에서도 edge
        # False Negative: 실제 edge가 합성에서 edge 아님
        tp = np.sum(real_edges & synth_edges)
        fn = np.sum(real_edges & ~synth_edges)
        fp = np.sum(~real_edges & synth_edges)

        # F1-score 형태
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        edge_preservations.append(f1)

    return np.mean(edge_preservations)


def calculate_frequency_distance(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    freq_band: Optional[Tuple[float, float]] = None,
) -> dict:
    """
    Frequency Domain Analysis

    ⭐⭐ 주파수 도메인에서 이미지를 분석합니다.

    고주파 성분 = 세부 구조 (조직의 미세 패턴)
    저주파 성분 = 전체 형태

    "내부 조직이 안 좋다"는 문제는 종종 고주파 성분의 손실로 나타납니다.

    Args:
        real_images: 원본 이미지 (N, H, W) or (N, H, W, C)
        synthetic_images: 합성 이미지 (N, H, W) or (N, H, W, C)
        freq_band: 분석할 주파수 대역 (None이면 전체)

    Returns:
        dict: {
            'total_distance': 전체 주파수 거리,
            'low_freq_distance': 저주파 거리,
            'high_freq_distance': 고주파 거리 (⭐ 중요!),
            'high_freq_ratio': 고주파 파워 비율
        }
    """
    if real_images.shape != synthetic_images.shape:
        raise ValueError(f"Shape mismatch: {real_images.shape} vs {synthetic_images.shape}")

    total_dists = []
    low_freq_dists = []
    high_freq_dists = []
    high_freq_ratios = []

    for i in range(len(real_images)):
        real_img = real_images[i]
        synth_img = synthetic_images[i]

        # Grayscale 변환 (필요시)
        if real_img.ndim == 3:
            real_gray = np.mean(real_img, axis=2)
            synth_gray = np.mean(synth_img, axis=2)
        else:
            real_gray = real_img
            synth_gray = synth_img

        # FFT
        real_fft = fftpack.fft2(real_gray)
        synth_fft = fftpack.fft2(synth_gray)

        # Power spectrum
        real_power = np.abs(real_fft) ** 2
        synth_power = np.abs(synth_fft) ** 2

        # 주파수별 거리 계산
        h, w = real_gray.shape
        center_h, center_w = h // 2, w // 2

        # 거리 맵 생성 (중심으로부터의 거리 = 주파수)
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y - center_h)**2 + (x - center_w)**2)
        max_dist = np.sqrt(center_h**2 + center_w**2)

        # 저주파/고주파 구분 (중심으로부터 30% 이내 = 저주파)
        low_freq_mask = dist_from_center < max_dist * 0.3
        high_freq_mask = ~low_freq_mask

        # 전체 거리
        total_dist = np.mean(np.abs(real_power - synth_power))
        total_dists.append(total_dist)

        # 저주파 거리
        low_dist = np.mean(np.abs(real_power[low_freq_mask] - synth_power[low_freq_mask]))
        low_freq_dists.append(low_dist)

        # 고주파 거리 ⭐
        high_dist = np.mean(np.abs(real_power[high_freq_mask] - synth_power[high_freq_mask]))
        high_freq_dists.append(high_dist)

        # 고주파 파워 비율
        real_high_power = np.sum(real_power[high_freq_mask])
        synth_high_power = np.sum(synth_power[high_freq_mask])
        high_ratio = synth_high_power / (real_high_power + 1e-10)
        high_freq_ratios.append(high_ratio)

    results = {
        'total_distance': np.mean(total_dists),
        'low_freq_distance': np.mean(low_freq_dists),
        'high_freq_distance': np.mean(high_freq_dists),
        'high_freq_ratio': np.mean(high_freq_ratios),  # 1.0에 가까울수록 좋음
    }

    return results


def calculate_all_structural_metrics(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
) -> dict:
    """
    모든 구조 특화 지표를 한 번에 계산

    Returns:
        dict: {
            'gradient_similarity': float,
            'edge_preservation': float,
            'freq_total_distance': float,
            'freq_low_distance': float,
            'freq_high_distance': float,
            'freq_high_ratio': float,
        }
    """
    results = {}

    # Gradient similarity
    results['gradient_similarity'] = calculate_gradient_similarity(
        real_images, synthetic_images
    )

    # Edge preservation
    results['edge_preservation'] = calculate_edge_preservation(
        real_images, synthetic_images
    )

    # Frequency analysis
    freq_results = calculate_frequency_distance(real_images, synthetic_images)
    results.update({
        'freq_total_distance': freq_results['total_distance'],
        'freq_low_distance': freq_results['low_freq_distance'],
        'freq_high_distance': freq_results['high_freq_distance'],
        'freq_high_ratio': freq_results['high_freq_ratio'],
    })

    return results
