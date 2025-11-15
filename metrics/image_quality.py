"""
Image Quality Metrics

전통적인 이미지 품질 평가 지표들:
- SSIM (Structural Similarity Index Measure)
- MS-SSIM (Multi-Scale SSIM)
- PSNR (Peak Signal-to-Noise Ratio)
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Union, Tuple


def calculate_ssim(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    data_range: float = None,
    **kwargs
) -> Union[float, np.ndarray]:
    """
    SSIM (Structural Similarity Index Measure) 계산

    구조적, 텍스처 유사도를 평가합니다.
    84%의 MRI 합성 연구에서 사용되는 대표적 지표입니다.

    Args:
        real_images: 원본 이미지 (N, H, W) or (N, H, W, C)
        synthetic_images: 합성 이미지 (N, H, W) or (N, H, W, C)
        data_range: 데이터 범위 (None이면 자동 계산)
        **kwargs: ssim 함수의 추가 인자

    Returns:
        평균 SSIM 값 (0~1, 높을수록 좋음)
    """
    if real_images.shape != synthetic_images.shape:
        raise ValueError(f"Shape mismatch: {real_images.shape} vs {synthetic_images.shape}")

    if data_range is None:
        data_range = real_images.max() - real_images.min()

    ssim_values = []

    # 배치 처리
    for i in range(len(real_images)):
        real_img = real_images[i]
        synth_img = synthetic_images[i]

        # Grayscale인지 확인
        if real_img.ndim == 2:
            channel_axis = None
        elif real_img.ndim == 3:
            channel_axis = -1
        else:
            raise ValueError(f"Unsupported image dimension: {real_img.ndim}")

        ssim_val = ssim(
            real_img,
            synth_img,
            data_range=data_range,
            channel_axis=channel_axis,
            **kwargs
        )
        ssim_values.append(ssim_val)

    return np.mean(ssim_values)


def calculate_ms_ssim(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    data_range: float = None,
    weights: Tuple[float] = None,
) -> float:
    """
    MS-SSIM (Multi-Scale Structural Similarity Index Measure) 계산

    다중 해상도에서 구조적 유사도를 평가하여 민감도를 높입니다.
    SSIM보다 인간의 지각과 더 일치합니다.

    Args:
        real_images: 원본 이미지 (N, H, W) or (N, H, W, C)
        synthetic_images: 합성 이미지 (N, H, W) or (N, H, W, C)
        data_range: 데이터 범위
        weights: 각 스케일의 가중치

    Returns:
        평균 MS-SSIM 값 (0~1, 높을수록 좋음)
    """
    from skimage.transform import resize

    if real_images.shape != synthetic_images.shape:
        raise ValueError(f"Shape mismatch: {real_images.shape} vs {synthetic_images.shape}")

    if data_range is None:
        data_range = real_images.max() - real_images.min()

    # 기본 가중치 (5 scales)
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    ms_ssim_values = []

    for i in range(len(real_images)):
        real_img = real_images[i]
        synth_img = synthetic_images[i]

        scale_ssims = []
        current_real = real_img.copy()
        current_synth = synth_img.copy()

        for scale_idx in range(len(weights)):
            # SSIM 계산
            if current_real.ndim == 2:
                channel_axis = None
            else:
                channel_axis = -1

            ssim_val = ssim(
                current_real,
                current_synth,
                data_range=data_range,
                channel_axis=channel_axis,
            )
            scale_ssims.append(ssim_val)

            # 다음 스케일을 위해 다운샘플링
            if scale_idx < len(weights) - 1:
                new_shape = tuple(s // 2 for s in current_real.shape[:2])
                if current_real.ndim == 3:
                    new_shape = new_shape + (current_real.shape[2],)

                current_real = resize(current_real, new_shape, anti_aliasing=True)
                current_synth = resize(current_synth, new_shape, anti_aliasing=True)

        # 가중 평균
        ms_ssim = np.prod([s**w for s, w in zip(scale_ssims, weights)])
        ms_ssim_values.append(ms_ssim)

    return np.mean(ms_ssim_values)


def calculate_psnr(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    data_range: float = None,
) -> float:
    """
    PSNR (Peak Signal-to-Noise Ratio) 계산

    이미지의 신호 품질(잡음 대비)을 평가하는 전통적 지표입니다.
    61%의 MRI 합성 연구에서 사용됩니다.

    주의: PSNR은 높지만 조직 구조가 안 좋을 수 있습니다.
    따라서 SSIM, LPIPS 등과 함께 사용해야 합니다.

    Args:
        real_images: 원본 이미지 (N, H, W) or (N, H, W, C)
        synthetic_images: 합성 이미지 (N, H, W) or (N, H, W, C)
        data_range: 데이터 범위

    Returns:
        평균 PSNR 값 (dB, 높을수록 좋음, 일반적으로 20-50 범위)
    """
    if real_images.shape != synthetic_images.shape:
        raise ValueError(f"Shape mismatch: {real_images.shape} vs {synthetic_images.shape}")

    if data_range is None:
        data_range = real_images.max() - real_images.min()

    psnr_values = []

    for i in range(len(real_images)):
        psnr_val = psnr(
            real_images[i],
            synthetic_images[i],
            data_range=data_range
        )
        psnr_values.append(psnr_val)

    return np.mean(psnr_values)


def calculate_all_image_quality_metrics(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    data_range: float = None,
) -> dict:
    """
    모든 이미지 품질 지표를 한 번에 계산

    Returns:
        dict: {'ssim': float, 'ms_ssim': float, 'psnr': float}
    """
    results = {
        'ssim': calculate_ssim(real_images, synthetic_images, data_range),
        'ms_ssim': calculate_ms_ssim(real_images, synthetic_images, data_range),
        'psnr': calculate_psnr(real_images, synthetic_images, data_range),
    }

    return results
