"""
Distribution Metrics

이미지 분포 간 유사도를 평가하는 지표들:
- FID (Fréchet Inception Distance)
- MMD (Maximum Mean Discrepancy)

이 지표들은 개별 이미지가 아닌, 전체 데이터셋의 분포 유사도를 측정합니다.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from scipy import linalg
import warnings


def calculate_fid(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    batch_size: int = 50,
    device: Optional[str] = None,
    use_inception: bool = True,
) -> float:
    """
    FID (Fréchet Inception Distance) 계산

    생성 이미지 분포와 원본 이미지 분포 간의 차이를 평가합니다.
    낮을수록 두 분포가 유사함을 의미합니다.

    FID는 전체 데이터셋의 품질을 평가하므로, 개별 이미지의
    조직 구조 품질은 SSIM, LPIPS 등으로 별도 평가해야 합니다.

    Args:
        real_images: 원본 이미지 (N, H, W) or (N, H, W, C)
        synthetic_images: 합성 이미지 (M, H, W) or (M, H, W, C)
        batch_size: 배치 크기
        device: 계산 디바이스
        use_inception: InceptionV3 사용 (False면 간단한 CNN 사용)

    Returns:
        FID 값 (낮을수록 좋음, 일반적으로 0-100 범위)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Feature extractor 준비
    if use_inception:
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
            model.fc = nn.Identity()  # 마지막 FC layer 제거
            model.eval().to(device)
            required_size = (299, 299)
        except Exception as e:
            warnings.warn(f"InceptionV3 로드 실패, 간단한 CNN 사용: {e}")
            use_inception = False

    if not use_inception:
        # 간단한 CNN feature extractor
        model = SimpleCNNFeatureExtractor().eval().to(device)
        required_size = (224, 224)

    # Feature 추출
    real_features = _extract_features(real_images, model, batch_size, device, required_size)
    synth_features = _extract_features(synthetic_images, model, batch_size, device, required_size)

    # 평균 및 공분산 계산
    mu_real = np.mean(real_features, axis=0)
    mu_synth = np.mean(synth_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_synth = np.cov(synth_features, rowvar=False)

    # FID 계산
    fid = _calculate_frechet_distance(mu_real, sigma_real, mu_synth, sigma_synth)

    return fid


def _extract_features(images, model, batch_size, device, required_size):
    """Feature 추출 헬퍼 함수"""
    from torchvision.transforms import functional as F

    features = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # 전처리
            batch_tensors = []
            for img in batch:
                # Grayscale -> RGB
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)

                # Numpy -> Tensor
                tensor = torch.from_numpy(img).permute(2, 0, 1).float()

                # 크기 조정
                tensor = F.resize(tensor.unsqueeze(0), required_size).squeeze(0)

                # 정규화 [0, 1]
                tensor = tensor / tensor.max() if tensor.max() > 0 else tensor

                batch_tensors.append(tensor)

            batch_tensor = torch.stack(batch_tensors).to(device)

            # Feature 추출
            feat = model(batch_tensor)
            features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet distance 계산"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # 공분산의 곱의 제곱근 계산
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # 수치적 안정성
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # 허수 부분 제거
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class SimpleCNNFeatureExtractor(nn.Module):
    """간단한 CNN feature extractor (InceptionV3 대안)"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.features(x)


def calculate_mmd(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    kernel: str = 'rbf',
    gamma: Optional[float] = None,
) -> float:
    """
    MMD (Maximum Mean Discrepancy) 계산

    합성 데이터 분포와 실제 데이터 분포 간의 차이를 측정하는 통계적 방법입니다.
    커널 기반으로 분포 간 거리를 계산합니다.

    Args:
        real_images: 원본 이미지 (N, H, W) or (N, H, W, C)
        synthetic_images: 합성 이미지 (M, H, W) or (M, H, W, C)
        kernel: 커널 종류 ('rbf', 'linear')
        gamma: RBF 커널의 gamma 파라미터

    Returns:
        MMD 값 (0 이상, 낮을수록 좋음)
    """
    # 이미지를 벡터로 flatten
    real_flat = real_images.reshape(len(real_images), -1)
    synth_flat = synthetic_images.reshape(len(synthetic_images), -1)

    # 샘플링 (계산 효율성을 위해)
    max_samples = 1000
    if len(real_flat) > max_samples:
        indices = np.random.choice(len(real_flat), max_samples, replace=False)
        real_flat = real_flat[indices]
    if len(synth_flat) > max_samples:
        indices = np.random.choice(len(synth_flat), max_samples, replace=False)
        synth_flat = synth_flat[indices]

    # 커널 함수
    if kernel == 'rbf':
        if gamma is None:
            # 중앙값 휴리스틱
            pairwise_dists = np.sum((real_flat[:, None, :] - real_flat[None, :, :]) ** 2, axis=2)
            gamma = 1.0 / np.median(pairwise_dists[pairwise_dists > 0])

        def kernel_func(x, y):
            pairwise_sq_dists = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
            return np.exp(-gamma * pairwise_sq_dists)
    elif kernel == 'linear':
        def kernel_func(x, y):
            return x @ y.T
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # MMD 계산
    k_xx = kernel_func(real_flat, real_flat)
    k_yy = kernel_func(synth_flat, synth_flat)
    k_xy = kernel_func(real_flat, synth_flat)

    n = len(real_flat)
    m = len(synth_flat)

    # Unbiased estimator
    mmd_sq = (np.sum(k_xx) - np.trace(k_xx)) / (n * (n - 1)) + \
             (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1)) - \
             2 * np.mean(k_xy)

    return np.sqrt(max(mmd_sq, 0))  # 수치 오차로 인한 음수 방지


def calculate_all_distribution_metrics(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    batch_size: int = 50,
    device: Optional[str] = None,
) -> dict:
    """
    모든 분포 지표를 한 번에 계산

    Returns:
        dict: {'fid': float, 'mmd': float}
    """
    results = {}

    try:
        results['fid'] = calculate_fid(
            real_images, synthetic_images,
            batch_size=batch_size, device=device
        )
    except Exception as e:
        warnings.warn(f"FID 계산 실패: {e}")
        results['fid'] = None

    try:
        results['mmd'] = calculate_mmd(real_images, synthetic_images)
    except Exception as e:
        warnings.warn(f"MMD 계산 실패: {e}")
        results['mmd'] = None

    return results
