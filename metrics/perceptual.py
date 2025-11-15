"""
Perceptual Metrics

인간의 지각과 일치하는 평가 지표들:
- LPIPS (Learned Perceptual Image Patch Similarity)
- VIF (Visual Information Fidelity)

⭐ LPIPS는 내부 조직 구조의 미세한 차이를 감지하는 데 SSIM보다 우수합니다!
"""

import numpy as np
import torch
from typing import Union, Optional
import warnings


def calculate_lpips(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    net: str = 'alex',
    device: Optional[str] = None,
    normalize: bool = True,
) -> float:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) 계산

    ⭐⭐ 강력 추천: 인간의 지각적 유사도와 가장 일치하는 지표

    SSIM보다 내부 조직 구조의 미세한 차이를 더 잘 감지합니다.
    논문에서 "생성물이 원본과 유사해 보이지만 내부 조직이 안 좋다"는
    문제를 정량화하는 데 매우 효과적입니다.

    Args:
        real_images: 원본 이미지 (N, H, W) or (N, H, W, C)
        synthetic_images: 합성 이미지 (N, H, W) or (N, H, W, C)
        net: 백본 네트워크 ('alex', 'vgg', 'squeeze')
        device: 계산 디바이스 (None이면 자동 선택)
        normalize: 입력을 [-1, 1]로 정규화할지 여부

    Returns:
        평균 LPIPS 거리 (0~1, 낮을수록 좋음, 일반적으로 0.0-0.5)
    """
    try:
        import lpips
    except ImportError:
        raise ImportError(
            "LPIPS를 사용하려면 'lpips' 패키지가 필요합니다.\n"
            "설치: pip install lpips"
        )

    if real_images.shape != synthetic_images.shape:
        raise ValueError(f"Shape mismatch: {real_images.shape} vs {synthetic_images.shape}")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # LPIPS 모델 초기화
    loss_fn = lpips.LPIPS(net=net).to(device)

    lpips_values = []

    with torch.no_grad():
        for i in range(len(real_images)):
            real_img = real_images[i]
            synth_img = synthetic_images[i]

            # Grayscale을 RGB로 변환 (LPIPS는 RGB 필요)
            if real_img.ndim == 2:
                real_img = np.stack([real_img] * 3, axis=-1)
                synth_img = np.stack([synth_img] * 3, axis=-1)

            # (H, W, C) -> (C, H, W)
            real_tensor = torch.from_numpy(real_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
            synth_tensor = torch.from_numpy(synth_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # 정규화 [-1, 1]
            if normalize:
                real_tensor = 2 * real_tensor / real_tensor.max() - 1
                synth_tensor = 2 * synth_tensor / synth_tensor.max() - 1

            # LPIPS 계산
            dist = loss_fn(real_tensor, synth_tensor)
            lpips_values.append(dist.item())

    return np.mean(lpips_values)


def calculate_vif(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    sigma_nsq: float = 2.0,
) -> float:
    """
    VIF (Visual Information Fidelity) 계산

    시각적 정보의 충실도를 평가합니다.
    이미지의 정보 내용이 얼마나 잘 보존되는지 측정합니다.

    Args:
        real_images: 원본 이미지 (N, H, W)
        synthetic_images: 합성 이미지 (N, H, W)
        sigma_nsq: 노이즈 분산

    Returns:
        평균 VIF 값 (0~1, 높을수록 좋음)
    """
    from scipy.ndimage import gaussian_filter

    if real_images.shape != synthetic_images.shape:
        raise ValueError(f"Shape mismatch: {real_images.shape} vs {synthetic_images.shape}")

    def vif_single_image(ref, dist, sigma_nsq=2.0):
        """단일 이미지에 대한 VIF 계산"""
        num_scales = 4
        sigma_nsq_vec = [sigma_nsq * (2 ** (i - 1)) for i in range(num_scales)]

        vif_val = 0.0

        for scale in range(num_scales):
            # 다운샘플링
            if scale > 0:
                ref = gaussian_filter(ref, sigma=1.0)[::2, ::2]
                dist = gaussian_filter(dist, sigma=1.0)[::2, ::2]

            # 지역 평균 및 분산 계산
            mu_ref = gaussian_filter(ref, sigma=1.5)
            mu_dist = gaussian_filter(dist, sigma=1.5)

            sigma_ref_sq = gaussian_filter(ref**2, sigma=1.5) - mu_ref**2
            sigma_dist_sq = gaussian_filter(dist**2, sigma=1.5) - mu_dist**2
            sigma_ref_dist = gaussian_filter(ref * dist, sigma=1.5) - mu_ref * mu_dist

            # 안정성을 위한 작은 값 추가
            sigma_ref_sq = np.maximum(sigma_ref_sq, 0)
            sigma_dist_sq = np.maximum(sigma_dist_sq, 0)

            # VIF 계산
            g = sigma_ref_dist / (sigma_ref_sq + 1e-10)
            sv_sq = sigma_dist_sq - g * sigma_ref_dist

            # 정보량 계산
            num = np.sum(np.log10(1 + g**2 * sigma_ref_sq / (sv_sq + sigma_nsq_vec[scale] + 1e-10)))
            den = np.sum(np.log10(1 + sigma_ref_sq / (sigma_nsq_vec[scale] + 1e-10)))

            if den > 0:
                vif_val += num / den

        return vif_val / num_scales

    vif_values = []

    for i in range(len(real_images)):
        real_img = real_images[i]
        synth_img = synthetic_images[i]

        # 다채널 이미지는 채널별 평균
        if real_img.ndim == 3:
            vif_per_channel = []
            for c in range(real_img.shape[2]):
                vif_c = vif_single_image(real_img[:, :, c], synth_img[:, :, c], sigma_nsq)
                vif_per_channel.append(vif_c)
            vif_val = np.mean(vif_per_channel)
        else:
            vif_val = vif_single_image(real_img, synth_img, sigma_nsq)

        vif_values.append(vif_val)

    return np.mean(vif_values)


def calculate_all_perceptual_metrics(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    lpips_net: str = 'alex',
    device: Optional[str] = None,
) -> dict:
    """
    모든 지각적 지표를 한 번에 계산

    Returns:
        dict: {'lpips': float, 'vif': float}
    """
    results = {}

    try:
        results['lpips'] = calculate_lpips(
            real_images, synthetic_images,
            net=lpips_net, device=device
        )
    except ImportError as e:
        warnings.warn(f"LPIPS 계산 건너뛰기: {e}")
        results['lpips'] = None

    try:
        results['vif'] = calculate_vif(real_images, synthetic_images)
    except Exception as e:
        warnings.warn(f"VIF 계산 실패: {e}")
        results['vif'] = None

    return results
