"""
NIfTI 파일을 사용한 MRI 평가 예시

NIfTI (.nii, .nii.gz) 포맷은 MRI 연구에서 가장 널리 사용되는 포맷입니다.
이 파일은 다양한 NIfTI 사용 시나리오를 보여줍니다.
"""

import numpy as np
from pathlib import Path
from evaluator import MRIEvaluator
from data_loader import (
    load_nifti,
    load_nifti_batch,
    load_nifti_directory,
    save_nifti,
    get_metadata_nifti,
    auto_load,
)


def example_1_single_nifti_file():
    """
    예시 1: 단일 NIfTI 파일 평가

    하나의 3D MRI 볼륨을 슬라이스별로 평가합니다.
    """
    print("\n" + "="*70)
    print("예시 1: 단일 NIfTI 파일 평가")
    print("="*70)

    # 방법 1: data_loader 직접 사용
    print("\n📂 방법 1: 수동으로 로드")

    # 원본 MRI 로드
    real_images = load_nifti(
        'path/to/real_mri.nii.gz',
        slice_axis=2,  # 0=sagittal, 1=coronal, 2=axial
        normalize=True
    )
    print(f"  원본: {real_images.shape}")  # (N_slices, H, W)

    # 합성 MRI 로드
    synthetic_images = load_nifti(
        'path/to/synthetic_mri.nii.gz',
        slice_axis=2,
        normalize=True
    )
    print(f"  합성: {synthetic_images.shape}")

    # 평가
    evaluator = MRIEvaluator()
    results = evaluator.evaluate_all(real_images, synthetic_images)
    evaluator.print_report(results)

    # 방법 2: evaluator에서 직접 로드 (더 간단!)
    print("\n📂 방법 2: 평가기에서 직접 로드 (권장)")

    evaluator = MRIEvaluator()
    results = evaluator.evaluate_from_files(
        'path/to/real_mri.nii.gz',
        'path/to/synthetic_mri.nii.gz',
        loader_kwargs={'slice_axis': 2}
    )
    evaluator.print_report(results)


def example_2_nifti_directory():
    """
    예시 2: 디렉토리의 모든 NIfTI 파일 평가

    여러 환자의 MRI를 한 번에 평가합니다.
    """
    print("\n" + "="*70)
    print("예시 2: 디렉토리 평가")
    print("="*70)

    # 디렉토리 구조:
    # real_mri/
    #   ├── patient001.nii.gz
    #   ├── patient002.nii.gz
    #   └── patient003.nii.gz
    # synthetic_mri/
    #   ├── patient001.nii.gz
    #   ├── patient002.nii.gz
    #   └── patient003.nii.gz

    # 방법 1: data_loader 사용
    real_images = load_nifti_directory(
        'path/to/real_mri/',
        pattern="*.nii.gz",
        slice_axis=2,
        slices_per_volume=10,  # 각 볼륨에서 10개 슬라이스만 샘플링
        verbose=True
    )

    synthetic_images = load_nifti_directory(
        'path/to/synthetic_mri/',
        pattern="*.nii.gz",
        slice_axis=2,
        slices_per_volume=10,
        verbose=True
    )

    evaluator = MRIEvaluator()
    results = evaluator.evaluate_all(real_images, synthetic_images)
    evaluator.print_report(results)

    # 방법 2: evaluator에서 직접 (더 간단!)
    evaluator = MRIEvaluator()
    results = evaluator.evaluate_from_files(
        'path/to/real_mri/',
        'path/to/synthetic_mri/',
        loader_kwargs={
            'slice_axis': 2,
            'slices_per_volume': 10
        }
    )


def example_3_specific_slices():
    """
    예시 3: 특정 슬라이스 범위만 평가

    전체 볼륨이 아닌 관심 있는 슬라이스만 평가합니다.
    """
    print("\n" + "="*70)
    print("예시 3: 특정 슬라이스 범위 평가")
    print("="*70)

    # 예: 100-150번 슬라이스만 평가 (뇌의 특정 부위)
    real_images = load_nifti(
        'path/to/real_mri.nii.gz',
        slice_axis=2,
        slice_range=(100, 150),  # 100~149번 슬라이스
        normalize=True
    )

    synthetic_images = load_nifti(
        'path/to/synthetic_mri.nii.gz',
        slice_axis=2,
        slice_range=(100, 150),
        normalize=True
    )

    print(f"선택된 슬라이스: {real_images.shape[0]}개")

    evaluator = MRIEvaluator()
    results = evaluator.evaluate_all(real_images, synthetic_images)
    evaluator.print_report(results)


def example_4_different_orientations():
    """
    예시 4: 다양한 방향(orientation)에서 평가

    Axial, Coronal, Sagittal 방향을 모두 평가합니다.
    """
    print("\n" + "="*70)
    print("예시 4: 다양한 방향에서 평가")
    print("="*70)

    orientations = {
        'Sagittal': 0,
        'Coronal': 1,
        'Axial': 2,
    }

    evaluator = MRIEvaluator(verbose=False)

    for orientation_name, slice_axis in orientations.items():
        print(f"\n{orientation_name} 평면 평가...")

        real_images = load_nifti(
            'path/to/real_mri.nii.gz',
            slice_axis=slice_axis,
            normalize=True
        )

        synthetic_images = load_nifti(
            'path/to/synthetic_mri.nii.gz',
            slice_axis=slice_axis,
            normalize=True
        )

        # 빠른 평가 (구조 지표만)
        results = evaluator.evaluate_all(
            real_images, synthetic_images,
            include_perceptual=False,
            include_distribution=False,
        )

        print(f"  SSIM: {results.get('ssim', 0):.4f}")
        print(f"  Gradient Sim: {results.get('gradient_similarity', 0):.4f}")


def example_5_nifti_metadata():
    """
    예시 5: NIfTI 메타데이터 활용

    Voxel size, affine 등 메타데이터를 활용합니다.
    """
    print("\n" + "="*70)
    print("예시 5: NIfTI 메타데이터 확인")
    print("="*70)

    # 메타데이터 추출
    metadata = get_metadata_nifti('path/to/real_mri.nii.gz')

    print("📋 메타데이터:")
    print(f"  Shape: {metadata['shape']}")
    print(f"  Voxel Size: {metadata['voxel_size']}")
    print(f"  Data Type: {metadata['data_type']}")
    print(f"  Affine:\n{metadata['affine']}")

    # Voxel size를 고려한 평가 예시
    voxel_size = metadata['voxel_size']
    print(f"\n💡 Voxel size가 이방성({voxel_size})이므로,")
    print(f"   가장 해상도가 높은 축으로 평가하는 것을 권장합니다.")

    # 가장 작은 voxel dimension 찾기
    min_voxel_axis = np.argmin(voxel_size[:3])
    print(f"   권장 slice_axis: {min_voxel_axis}")


def example_6_save_results_as_nifti():
    """
    예시 6: 결과를 NIfTI로 저장

    평가 후 특정 지표의 맵을 NIfTI로 저장합니다.
    """
    print("\n" + "="*70)
    print("예시 6: 결과를 NIfTI로 저장")
    print("="*70)

    # 이미지 로드
    real_images = load_nifti('path/to/real_mri.nii.gz')
    synthetic_images = load_nifti('path/to/synthetic_mri.nii.gz')

    # Slice-wise 차이 맵 계산
    from skimage.metrics import structural_similarity as ssim

    ssim_map = np.zeros(real_images.shape[0])
    for i in range(len(real_images)):
        ssim_map[i] = ssim(real_images[i], synthetic_images[i])

    print(f"SSIM per slice: {ssim_map.mean():.4f} ± {ssim_map.std():.4f}")

    # SSIM 맵을 3D 볼륨으로 저장
    # (N,) -> (N, 1, 1) -> (1, 1, N)로 변환하여 저장
    ssim_volume = ssim_map[:, None, None]

    save_nifti(
        ssim_volume,
        'ssim_map.nii.gz'
    )

    print("✓ SSIM 맵 저장: ssim_map.nii.gz")


def example_7_auto_format_detection():
    """
    예시 7: 자동 포맷 감지

    파일 확장자를 자동으로 감지하여 로드합니다.
    """
    print("\n" + "="*70)
    print("예시 7: 자동 포맷 감지")
    print("="*70)

    # NIfTI, NumPy, 이미지 등을 자동으로 감지
    paths = [
        'path/to/data.nii.gz',  # NIfTI
        'path/to/data.npy',      # NumPy
        'path/to/image_dir/',    # 이미지 디렉토리
    ]

    for path in paths:
        try:
            data = auto_load(path)
            print(f"✓ {path}: shape={data.shape}")
        except Exception as e:
            print(f"✗ {path}: {e}")

    # evaluator에서도 자동 감지
    evaluator = MRIEvaluator()
    results = evaluator.evaluate_from_files(
        'real_data.nii.gz',      # NIfTI 자동 감지
        'synthetic_data.nii.gz',
    )


def example_8_real_world_workflow():
    """
    예시 8: 실제 연구 워크플로우

    전체 평가 파이프라인을 보여줍니다.
    """
    print("\n" + "="*70)
    print("예시 8: 실제 연구 워크플로우")
    print("="*70)

    # 실제 사용 시나리오
    real_dir = 'data/real_mri/'
    synthetic_dir = 'data/synthetic_mri_ddpm_100steps/'
    output_dir = 'results/'

    # 1. 평가
    print("\n1️⃣ 평가 시작...")
    evaluator = MRIEvaluator(verbose=True)

    results = evaluator.evaluate_from_files(
        real_dir,
        synthetic_dir,
        loader_kwargs={
            'slice_axis': 2,  # Axial 슬라이스
            'slices_per_volume': 20,  # 각 볼륨에서 20개 샘플링
            'normalize': True,
        },
        include_perceptual=True,
        include_distribution=True,
        include_structural=True,
    )

    # 2. 결과 출력
    print("\n2️⃣ 결과 보고서...")
    evaluator.print_report(results, highlight_issues=True)

    # 3. 결과 저장
    print("\n3️⃣ 결과 저장...")
    Path(output_dir).mkdir(exist_ok=True)

    evaluator.export_json(
        results,
        f'{output_dir}/evaluation_results.json'
    )

    evaluator.export_latex_table(
        results,
        f'{output_dir}/results_table.tex'
    )

    # 4. 핵심 지표 확인
    print("\n4️⃣ 핵심 지표 요약:")
    print(f"  SSIM: {results.get('ssim', 0):.4f}")
    print(f"  LPIPS: {results.get('lpips', 0):.4f}")
    print(f"  Gradient Similarity: {results.get('gradient_similarity', 0):.4f}")
    print(f"  High-Freq Ratio: {results.get('freq_high_ratio', 0):.4f}")

    # 5. 문제 진단
    print("\n5️⃣ 자동 진단:")
    lpips = results.get('lpips', 0)
    grad_sim = results.get('gradient_similarity', 0)
    freq_ratio = results.get('freq_high_ratio', 0)

    if lpips > 0.2:
        print("  ⚠️  LPIPS가 높습니다 → 미세 구조 차이 존재")
        print("     권장: Perceptual loss 추가, 고해상도 학습")

    if grad_sim < 0.7:
        print("  ⚠️  Gradient Similarity가 낮습니다 → 경계가 흐릿함")
        print("     권장: Edge-preserving loss, Guidance scale 조정")

    if abs(freq_ratio - 1.0) > 0.2:
        print(f"  ⚠️  High-Freq Ratio({freq_ratio:.3f})가 1.0에서 멀어짐")
        if freq_ratio < 1.0:
            print("     → 세부 구조 손실 (과도한 평활화)")
            print("     권장: Denoising 강도 감소, Step 수 증가")
        else:
            print("     → 세부 구조 과장 (노이즈)")
            print("     권장: Denoising 강도 증가")

    print("\n✅ 평가 완료!")


def create_demo_nifti_files():
    """
    데모용 NIfTI 파일 생성

    실제 NIfTI 파일이 없는 경우 테스트용으로 사용합니다.
    """
    print("\n" + "="*70)
    print("데모용 NIfTI 파일 생성")
    print("="*70)

    # 데모 데이터 생성
    np.random.seed(42)

    # 3D MRI 볼륨 (176 x 256 x 256) - 일반적인 뇌 MRI 크기
    real_volume = np.random.rand(176, 256, 256).astype(np.float32)

    # 합성 볼륨 (약간의 블러 추가)
    from scipy.ndimage import gaussian_filter
    synthetic_volume = gaussian_filter(real_volume, sigma=1.0)

    # NIfTI 저장
    Path('demo_data').mkdir(exist_ok=True)

    save_nifti(real_volume, 'demo_data/real_mri.nii.gz')
    save_nifti(synthetic_volume, 'demo_data/synthetic_mri.nii.gz')

    print("✓ 데모 파일 생성 완료:")
    print("  - demo_data/real_mri.nii.gz")
    print("  - demo_data/synthetic_mri.nii.gz")

    # 메타데이터 확인
    metadata = get_metadata_nifti('demo_data/real_mri.nii.gz')
    print(f"\n메타데이터:")
    print(f"  Shape: {metadata['shape']}")


if __name__ == '__main__':
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║          NIfTI 파일을 사용한 MRI 평가 예시                         ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    # 데모 파일 생성 (실제 NIfTI 파일이 없는 경우)
    create_demo_nifti_files()

    # 예시 실행
    # 주석을 해제하여 원하는 예시를 실행하세요

    # 예시 1: 단일 파일
    # example_1_single_nifti_file()

    # 예시 2: 디렉토리
    # example_2_nifti_directory()

    # 예시 3: 특정 슬라이스
    # example_3_specific_slices()

    # 예시 4: 다양한 방향
    # example_4_different_orientations()

    # 예시 5: 메타데이터
    # example_5_nifti_metadata()

    # 예시 6: 결과 저장
    # example_6_save_results_as_nifti()

    # 예시 7: 자동 감지
    # example_7_auto_format_detection()

    # 예시 8: 실제 워크플로우 (권장!)
    # example_8_real_world_workflow()

    print("\n" + "="*70)
    print("✅ 예시 파일 준비 완료!")
    print("="*70)
    print("\n💡 사용법:")
    print("  1. 위 함수들의 주석을 해제하여 실행")
    print("  2. 파일 경로를 실제 데이터 경로로 변경")
    print("  3. example_8_real_world_workflow()를 참고하여 전체 파이프라인 구성")
