"""
MRI Synthetic Data Evaluation - 사용 예시

이 파일은 다양한 사용 시나리오를 보여줍니다.
"""

import numpy as np
from evaluator import MRIEvaluator


def example_1_basic_usage():
    """
    예시 1: 기본 사용법

    diffusion model로 생성한 MRI 이미지와 원본 이미지를 비교합니다.
    """
    print("\n" + "="*70)
    print("예시 1: 기본 사용법")
    print("="*70)

    # 1. 이미지 로드 (실제로는 파일에서 로드)
    # 예: real_images = np.load('real_mri.npy')
    #     synthetic_images = np.load('synthetic_mri.npy')

    # 데모용 랜덤 데이터 (실제로는 실제 MRI 데이터 사용)
    np.random.seed(42)
    n_images = 10
    h, w = 256, 256

    real_images = np.random.rand(n_images, h, w).astype(np.float32)
    # 합성 이미지는 약간의 노이즈와 블러 추가
    synthetic_images = real_images + np.random.randn(n_images, h, w) * 0.05

    # 2. 평가기 초기화
    evaluator = MRIEvaluator(verbose=True)

    # 3. 모든 지표 평가
    results = evaluator.evaluate_all(
        real_images,
        synthetic_images,
        include_perceptual=True,
        include_distribution=True,
        include_structural=True,
        include_segmentation=False,  # 세그멘테이션 맵이 없으므로 False
    )

    # 4. 결과 출력
    evaluator.print_report(results)

    # 5. 결과 저장
    evaluator.export_json(results, 'results/evaluation_results.json')
    evaluator.export_latex_table(results, 'results/latex_table.tex')


def example_2_focus_on_tissue_structure():
    """
    예시 2: 조직 구조에 집중한 평가

    "내부 조직 구조가 안 좋다"는 문제를 진단합니다.
    """
    print("\n" + "="*70)
    print("예시 2: 조직 구조 평가")
    print("="*70)

    # 데모 데이터
    np.random.seed(42)
    n_images = 10
    h, w = 256, 256

    real_images = np.random.rand(n_images, h, w).astype(np.float32)

    # 시나리오: 고주파 성분(세부 구조)이 손실된 경우 시뮬레이션
    from scipy.ndimage import gaussian_filter
    synthetic_images = np.array([
        gaussian_filter(img, sigma=2.0)  # 블러 적용
        for img in real_images
    ])

    # 조직 구조 평가에 중요한 지표들만 계산
    from metrics.perceptual import calculate_lpips
    from metrics.structural import (
        calculate_gradient_similarity,
        calculate_frequency_distance
    )

    print("\n🔬 조직 구조 특화 평가...")

    # LPIPS (미세한 구조 차이)
    try:
        lpips = calculate_lpips(real_images, synthetic_images)
        print(f"  LPIPS: {lpips:.4f} {'⚠️  높음!' if lpips > 0.2 else '✓'}")
    except ImportError:
        print("  LPIPS: 건너뛰기 (lpips 패키지 필요)")

    # Gradient Similarity (경계 선명도)
    grad_sim = calculate_gradient_similarity(real_images, synthetic_images)
    print(f"  Gradient Similarity: {grad_sim:.4f} {'⚠️  낮음!' if grad_sim < 0.7 else '✓'}")

    # Frequency Analysis (세부 구조)
    freq_results = calculate_frequency_distance(real_images, synthetic_images)
    high_freq_ratio = freq_results['high_freq_ratio']
    print(f"  High-Freq Ratio: {high_freq_ratio:.4f} {'⚠️  세부 구조 손실!' if abs(high_freq_ratio - 1.0) > 0.2 else '✓'}")

    print("\n💡 해석:")
    if grad_sim < 0.7 or abs(high_freq_ratio - 1.0) > 0.2:
        print("  조직의 세부 구조가 원본보다 흐릿하거나 손실되었습니다.")
        print("  권장 사항:")
        print("    - Diffusion model의 step 수 증가")
        print("    - Loss function에 perceptual loss 추가")
        print("    - 고해상도 학습 데이터 사용")
    else:
        print("  조직 구조가 잘 보존되었습니다.")


def example_3_with_segmentation():
    """
    예시 3: 세그멘테이션 기반 평가

    특정 해부학적 구조(회백질, 백질 등)의 정확도를 평가합니다.
    """
    print("\n" + "="*70)
    print("예시 3: 세그멘테이션 기반 평가")
    print("="*70)

    # 데모 데이터
    np.random.seed(42)
    n_images = 10
    h, w = 256, 256

    # 세그멘테이션 맵 생성 (실제로는 세그멘테이션 모델 출력)
    # Label: 0=배경, 1=회백질, 2=백질, 3=CSF
    real_segmentations = np.random.randint(0, 4, size=(n_images, h, w))

    # 합성 이미지의 세그멘테이션 (약간의 오차 포함)
    synthetic_segmentations = real_segmentations.copy()
    # 10% 픽셀에 노이즈 추가
    noise_mask = np.random.rand(n_images, h, w) < 0.1
    synthetic_segmentations[noise_mask] = np.random.randint(0, 4, size=noise_mask.sum())

    # 세그멘테이션 기반 평가
    from metrics.segmentation import calculate_dice, calculate_hausdorff

    print("\n🧠 해부학적 구조 평가...")

    # Dice coefficient
    dice_results = calculate_dice(
        real_segmentations,
        synthetic_segmentations,
        labels=[1, 2, 3]  # 배경 제외
    )

    print("  Dice Coefficient:")
    for label_name, label_id in [('회백질', 1), ('백질', 2), ('CSF', 3)]:
        dice = dice_results.get(f'label_{label_id}', None)
        if dice is not None:
            status = '✓' if dice > 0.7 else '⚠️  낮음!'
            print(f"    {label_name}: {dice:.4f} {status}")

    # Hausdorff distance
    hausdorff_results = calculate_hausdorff(
        real_segmentations,
        synthetic_segmentations,
        labels=[1, 2, 3],
        percentile=95
    )

    print("\n  Hausdorff Distance (95th percentile):")
    for label_name, label_id in [('회백질', 1), ('백질', 2), ('CSF', 3)]:
        hd = hausdorff_results.get(f'label_{label_id}', None)
        if hd is not None and hd != float('inf'):
            status = '✓' if hd < 5.0 else '⚠️  높음!'
            print(f"    {label_name}: {hd:.2f} px {status}")


def example_4_batch_evaluation():
    """
    예시 4: 배치 평가

    여러 실험 조건의 결과를 한 번에 평가하고 비교합니다.
    """
    print("\n" + "="*70)
    print("예시 4: 배치 평가")
    print("="*70)

    # 여러 실험 조건
    experiments = {
        'DDPM_50steps': {'noise_level': 0.05, 'blur_sigma': 1.0},
        'DDPM_100steps': {'noise_level': 0.03, 'blur_sigma': 0.5},
        'DDPM_200steps': {'noise_level': 0.01, 'blur_sigma': 0.2},
    }

    # 데모 데이터
    np.random.seed(42)
    n_images = 10
    h, w = 256, 256
    real_images = np.random.rand(n_images, h, w).astype(np.float32)

    evaluator = MRIEvaluator(verbose=False)
    all_results = {}

    for exp_name, exp_config in experiments.items():
        print(f"\n실험: {exp_name}")

        # 합성 이미지 생성 (시뮬레이션)
        from scipy.ndimage import gaussian_filter
        synthetic_images = real_images + \
            np.random.randn(n_images, h, w) * exp_config['noise_level']
        synthetic_images = np.array([
            gaussian_filter(img, sigma=exp_config['blur_sigma'])
            for img in synthetic_images
        ])

        # 평가
        results = evaluator.evaluate_all(
            real_images, synthetic_images,
            include_perceptual=False,  # 빠른 평가를 위해 제외
            include_distribution=False,
        )

        all_results[exp_name] = results

        # 주요 지표만 출력
        print(f"  SSIM: {results.get('ssim', 0):.4f}")
        print(f"  Gradient Sim: {results.get('gradient_similarity', 0):.4f}")
        print(f"  High-Freq Ratio: {results.get('freq_high_ratio', 0):.4f}")

    # 최고 성능 실험 찾기
    print("\n🏆 최고 성능:")
    best_exp = max(all_results.items(), key=lambda x: x[1].get('ssim', 0))
    print(f"  {best_exp[0]} (SSIM: {best_exp[1].get('ssim', 0):.4f})")


def example_5_incremental_evaluation():
    """
    예시 5: 점진적 평가

    필요한 지표만 선택적으로 계산합니다.
    """
    print("\n" + "="*70)
    print("예시 5: 점진적 평가")
    print("="*70)

    # 데모 데이터
    np.random.seed(42)
    n_images = 10
    h, w = 256, 256
    real_images = np.random.rand(n_images, h, w).astype(np.float32)
    synthetic_images = real_images + np.random.randn(n_images, h, w) * 0.05

    # 1단계: 빠른 기본 지표
    print("\n1단계: 빠른 기본 평가...")
    from metrics.image_quality import calculate_ssim, calculate_psnr

    ssim = calculate_ssim(real_images, synthetic_images)
    psnr = calculate_psnr(real_images, synthetic_images)

    print(f"  SSIM: {ssim:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")

    # SSIM이 낮으면 추가 조사
    if ssim < 0.8:
        print("\n2단계: SSIM이 낮으므로 원인 조사...")

        from metrics.structural import (
            calculate_gradient_similarity,
            calculate_frequency_distance
        )

        grad_sim = calculate_gradient_similarity(real_images, synthetic_images)
        freq_results = calculate_frequency_distance(real_images, synthetic_images)

        print(f"  Gradient Similarity: {grad_sim:.4f}")
        print(f"  High-Freq Ratio: {freq_results['high_freq_ratio']:.4f}")

        if grad_sim < 0.7:
            print("\n  → 문제: 조직 경계가 흐릿함")
        if abs(freq_results['high_freq_ratio'] - 1.0) > 0.2:
            print("\n  → 문제: 세부 구조 손실")


if __name__ == '__main__':
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║          MRI Synthetic Data Evaluation - 사용 예시                 ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    # 예시 실행
    # 각 예시를 주석 해제하여 실행하세요

    # 예시 1: 기본 사용법
    # example_1_basic_usage()

    # 예시 2: 조직 구조 평가
    example_2_focus_on_tissue_structure()

    # 예시 3: 세그멘테이션 기반 평가
    # example_3_with_segmentation()

    # 예시 4: 배치 평가
    # example_4_batch_evaluation()

    # 예시 5: 점진적 평가
    # example_5_incremental_evaluation()

    print("\n" + "="*70)
    print("✅ 예시 실행 완료!")
    print("="*70)
    print("\n💡 팁:")
    print("  - 실제 사용 시 example_1_basic_usage()를 참고하세요")
    print("  - 조직 구조 문제 진단: example_2_focus_on_tissue_structure()")
    print("  - 해부학적 정확도: example_3_with_segmentation()")
    print("  - requirements.txt의 패키지들을 먼저 설치하세요:")
    print("    pip install -r requirements.txt")
