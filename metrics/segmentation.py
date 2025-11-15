"""
Segmentation-Based Metrics

해부학적 구조의 정확도를 평가하는 지표들:
- Dice Coefficient (구조적 일치도)
- Hausdorff Distance (경계 간 최대 거리)

이 지표들은 세그멘테이션 맵이 필요합니다.
세그멘테이션 모델을 사용하거나, 수동 주석을 사용할 수 있습니다.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Optional, Union


def calculate_dice(
    real_segmentations: np.ndarray,
    synthetic_segmentations: np.ndarray,
    labels: Optional[list] = None,
) -> Union[float, dict]:
    """
    Dice Coefficient 계산

    특정 조직(회백질, 백질 등)의 구조적 일치도를 직접 측정합니다.

    세그멘테이션 맵을 입력으로 받아, 각 해부학적 구조가
    얼마나 정확하게 재현되는지 평가합니다.

    Args:
        real_segmentations: 원본 세그멘테이션 (N, H, W)
        synthetic_segmentations: 합성 세그멘테이션 (N, H, W)
        labels: 평가할 레이블 리스트 (None이면 모든 레이블)

    Returns:
        평균 Dice coefficient (0~1, 높을수록 좋음)
        또는 레이블별 Dice coefficient dict
    """
    if real_segmentations.shape != synthetic_segmentations.shape:
        raise ValueError(
            f"Shape mismatch: {real_segmentations.shape} vs {synthetic_segmentations.shape}"
        )

    if labels is None:
        # 모든 고유 레이블 찾기 (배경 제외)
        labels = np.unique(real_segmentations)
        labels = labels[labels != 0]  # 배경 제외

    dice_scores = {}

    for label in labels:
        label_dice_scores = []

        for i in range(len(real_segmentations)):
            real_mask = (real_segmentations[i] == label)
            synth_mask = (synthetic_segmentations[i] == label)

            # Dice coefficient 계산
            intersection = np.sum(real_mask & synth_mask)
            union = np.sum(real_mask) + np.sum(synth_mask)

            if union == 0:
                # 두 마스크 모두 비어있으면 perfect match
                dice = 1.0
            else:
                dice = 2.0 * intersection / union

            label_dice_scores.append(dice)

        dice_scores[f'label_{label}'] = np.mean(label_dice_scores)

    # 전체 평균
    dice_scores['mean'] = np.mean(list(dice_scores.values()))

    return dice_scores


def calculate_hausdorff(
    real_segmentations: np.ndarray,
    synthetic_segmentations: np.ndarray,
    labels: Optional[list] = None,
    percentile: float = 95,
) -> Union[float, dict]:
    """
    Hausdorff Distance 계산

    구조 경계 간의 최대 거리를 측정합니다.
    조직 경계가 얼마나 정확한지 평가합니다.

    Args:
        real_segmentations: 원본 세그멘테이션 (N, H, W)
        synthetic_segmentations: 합성 세그멘테이션 (N, H, W)
        labels: 평가할 레이블 리스트
        percentile: Hausdorff percentile (95 = 95th percentile Hausdorff)

    Returns:
        평균 Hausdorff distance (낮을수록 좋음, 픽셀 단위)
        또는 레이블별 Hausdorff distance dict
    """
    if real_segmentations.shape != synthetic_segmentations.shape:
        raise ValueError(
            f"Shape mismatch: {real_segmentations.shape} vs {synthetic_segmentations.shape}"
        )

    if labels is None:
        labels = np.unique(real_segmentations)
        labels = labels[labels != 0]

    hausdorff_distances = {}

    for label in labels:
        label_hausdorff = []

        for i in range(len(real_segmentations)):
            real_mask = (real_segmentations[i] == label)
            synth_mask = (synthetic_segmentations[i] == label)

            # 두 마스크 모두 비어있으면 거리 0
            if not real_mask.any() and not synth_mask.any():
                label_hausdorff.append(0.0)
                continue

            # 한쪽만 비어있으면 무한대 (실제로는 큰 값)
            if not real_mask.any() or not synth_mask.any():
                label_hausdorff.append(float('inf'))
                continue

            # Distance transform
            real_dist = distance_transform_edt(~real_mask)
            synth_dist = distance_transform_edt(~synth_mask)

            # Hausdorff distance (directed)
            real_to_synth = real_dist[synth_mask]
            synth_to_real = synth_dist[real_mask]

            if percentile == 100:
                # Maximum Hausdorff
                hd = max(np.max(real_to_synth), np.max(synth_to_real))
            else:
                # Percentile Hausdorff (더 robust)
                hd = max(
                    np.percentile(real_to_synth, percentile),
                    np.percentile(synth_to_real, percentile)
                )

            label_hausdorff.append(hd)

        # 무한대 제외하고 평균
        valid_distances = [d for d in label_hausdorff if d != float('inf')]
        if valid_distances:
            hausdorff_distances[f'label_{label}'] = np.mean(valid_distances)
        else:
            hausdorff_distances[f'label_{label}'] = float('inf')

    # 전체 평균 (무한대 제외)
    valid_values = [v for v in hausdorff_distances.values() if v != float('inf')]
    if valid_values:
        hausdorff_distances['mean'] = np.mean(valid_values)
    else:
        hausdorff_distances['mean'] = float('inf')

    return hausdorff_distances


def segment_images_with_model(
    images: np.ndarray,
    model_path: Optional[str] = None,
    device: str = 'cpu',
) -> np.ndarray:
    """
    세그멘테이션 모델을 사용하여 이미지를 세그멘테이션

    사용자가 직접 세그멘테이션 모델을 제공해야 합니다.
    또는 pretrained 모델을 사용할 수 있습니다.

    Args:
        images: 입력 이미지 (N, H, W) or (N, H, W, C)
        model_path: 세그멘테이션 모델 경로
        device: 계산 디바이스

    Returns:
        세그멘테이션 맵 (N, H, W)
    """
    # 이것은 예시 구조입니다
    # 실제로는 사용자의 세그멘테이션 모델을 로드해야 합니다

    raise NotImplementedError(
        "세그멘테이션 모델을 직접 구현하거나 제공해야 합니다.\n"
        "예시:\n"
        "  - nnU-Net, FastSurfer 등의 Brain MRI 세그멘테이션 모델 사용\n"
        "  - 자체 훈련한 세그멘테이션 모델 사용\n"
        "  - 수동 주석 사용"
    )


def calculate_all_segmentation_metrics(
    real_images: np.ndarray,
    synthetic_images: np.ndarray,
    segmentation_model_path: Optional[str] = None,
    real_segmentations: Optional[np.ndarray] = None,
    synthetic_segmentations: Optional[np.ndarray] = None,
    labels: Optional[list] = None,
) -> dict:
    """
    모든 세그멘테이션 기반 지표를 한 번에 계산

    두 가지 사용 방법:
    1. 세그멘테이션 맵을 직접 제공
    2. 세그멘테이션 모델을 제공 (자동 세그멘테이션)

    Args:
        real_images: 원본 이미지
        synthetic_images: 합성 이미지
        segmentation_model_path: 세그멘테이션 모델 경로
        real_segmentations: 원본 세그멘테이션 (이미 있는 경우)
        synthetic_segmentations: 합성 세그멘테이션 (이미 있는 경우)
        labels: 평가할 레이블

    Returns:
        dict: {
            'dice_<label>': float,
            'dice_mean': float,
            'hausdorff_<label>': float,
            'hausdorff_mean': float,
        }
    """
    # 세그멘테이션 맵 준비
    if real_segmentations is None or synthetic_segmentations is None:
        if segmentation_model_path is None:
            raise ValueError(
                "세그멘테이션 맵 또는 세그멘테이션 모델을 제공해야 합니다."
            )

        # 자동 세그멘테이션 (사용자가 구현해야 함)
        real_segmentations = segment_images_with_model(
            real_images, segmentation_model_path
        )
        synthetic_segmentations = segment_images_with_model(
            synthetic_images, segmentation_model_path
        )

    # Dice coefficient
    dice_results = calculate_dice(
        real_segmentations, synthetic_segmentations, labels
    )

    # Hausdorff distance
    hausdorff_results = calculate_hausdorff(
        real_segmentations, synthetic_segmentations, labels
    )

    # 결과 합치기
    results = {}
    for key, value in dice_results.items():
        results[f'dice_{key}'] = value
    for key, value in hausdorff_results.items():
        results[f'hausdorff_{key}'] = value

    return results
