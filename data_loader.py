"""
Data Loader for MRI Evaluation

다양한 포맷의 MRI 이미지를 로드합니다:
- NumPy arrays (.npy)
- NIfTI files (.nii, .nii.gz)
- PNG/JPG images
"""

import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple
import warnings


def load_nifti(
    file_path: Union[str, Path],
    slice_axis: int = 2,
    slice_range: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    단일 NIfTI 파일 로드

    Args:
        file_path: NIfTI 파일 경로 (.nii 또는 .nii.gz)
        slice_axis: 슬라이스 축 (0, 1, 2) - 기본값 2 (axial)
        slice_range: 로드할 슬라이스 범위 (start, end). None이면 전체
        normalize: 0-1로 정규화할지 여부

    Returns:
        이미지 배열 (N, H, W) - N은 슬라이스 수
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "NIfTI 파일을 로드하려면 'nibabel' 패키지가 필요합니다.\n"
            "설치: pip install nibabel"
        )

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    # NIfTI 로드
    nii = nib.load(str(file_path))
    data = nii.get_fdata()

    # 3D 볼륨을 슬라이스로 분할
    if data.ndim == 3:
        # 슬라이스 축으로 이동
        data = np.moveaxis(data, slice_axis, 0)  # (N, H, W)
    elif data.ndim == 4:
        # 4D 볼륨 (e.g., 시계열 또는 멀티모달)
        warnings.warn(
            f"4D 볼륨 감지: shape={data.shape}. "
            f"첫 번째 볼륨만 사용합니다."
        )
        data = data[..., 0]  # 첫 번째 볼륨 사용
        data = np.moveaxis(data, slice_axis, 0)
    else:
        raise ValueError(f"지원하지 않는 차원: {data.ndim}D")

    # 슬라이스 범위 선택
    if slice_range is not None:
        start, end = slice_range
        data = data[start:end]

    # 정규화
    if normalize:
        data = _normalize_array(data)

    return data.astype(np.float32)


def load_nifti_batch(
    file_paths: List[Union[str, Path]],
    slice_axis: int = 2,
    slices_per_volume: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    여러 NIfTI 파일을 배치로 로드

    Args:
        file_paths: NIfTI 파일 경로 리스트
        slice_axis: 슬라이스 축
        slices_per_volume: 각 볼륨에서 추출할 슬라이스 수 (None이면 전체)
        normalize: 정규화 여부

    Returns:
        배치 이미지 배열 (N, H, W)
    """
    all_slices = []

    for file_path in file_paths:
        # 파일 로드
        slices = load_nifti(
            file_path,
            slice_axis=slice_axis,
            normalize=normalize
        )

        # 슬라이스 샘플링
        if slices_per_volume is not None and len(slices) > slices_per_volume:
            # 균등하게 샘플링
            indices = np.linspace(0, len(slices) - 1, slices_per_volume, dtype=int)
            slices = slices[indices]

        all_slices.append(slices)

    # 결합
    return np.concatenate(all_slices, axis=0)


def load_nifti_directory(
    directory: Union[str, Path],
    pattern: str = "*.nii.gz",
    slice_axis: int = 2,
    slices_per_volume: Optional[int] = None,
    normalize: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    디렉토리의 모든 NIfTI 파일을 로드

    Args:
        directory: 디렉토리 경로
        pattern: 파일 패턴 (기본값: "*.nii.gz")
        slice_axis: 슬라이스 축
        slices_per_volume: 각 볼륨에서 추출할 슬라이스 수
        normalize: 정규화 여부
        verbose: 진행 상황 출력

    Returns:
        배치 이미지 배열 (N, H, W)
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory}")

    # 파일 찾기
    file_paths = sorted(directory.glob(pattern))

    if len(file_paths) == 0:
        raise ValueError(f"'{pattern}' 패턴과 일치하는 파일이 없습니다: {directory}")

    if verbose:
        print(f"📂 {len(file_paths)}개의 NIfTI 파일 로드 중...")

    # 배치 로드
    images = load_nifti_batch(
        file_paths,
        slice_axis=slice_axis,
        slices_per_volume=slices_per_volume,
        normalize=normalize
    )

    if verbose:
        print(f"✓ 완료: {images.shape[0]}개 슬라이스, shape={images.shape}")

    return images


def load_numpy(
    file_path: Union[str, Path],
    normalize: bool = True,
) -> np.ndarray:
    """
    NumPy 파일 로드 (.npy)

    Args:
        file_path: .npy 파일 경로
        normalize: 정규화 여부

    Returns:
        이미지 배열
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    data = np.load(str(file_path))

    if normalize:
        data = _normalize_array(data)

    return data.astype(np.float32)


def load_images(
    file_paths: List[Union[str, Path]],
    grayscale: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    PNG/JPG 이미지 파일들을 로드

    Args:
        file_paths: 이미지 파일 경로 리스트
        grayscale: 그레이스케일로 변환할지 여부
        normalize: 정규화 여부

    Returns:
        이미지 배열 (N, H, W) 또는 (N, H, W, C)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "이미지 파일을 로드하려면 'Pillow' 패키지가 필요합니다.\n"
            "설치: pip install Pillow"
        )

    images = []

    for file_path in file_paths:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        img = Image.open(file_path)

        if grayscale:
            img = img.convert('L')

        img_array = np.array(img)
        images.append(img_array)

    data = np.stack(images, axis=0)

    if normalize:
        data = _normalize_array(data)

    return data.astype(np.float32)


def auto_load(
    path: Union[str, Path],
    **kwargs
) -> np.ndarray:
    """
    파일 확장자를 자동으로 감지하여 로드

    지원 포맷:
    - .nii, .nii.gz: NIfTI
    - .npy: NumPy
    - .png, .jpg, .jpeg: 이미지

    Args:
        path: 파일 또는 디렉토리 경로
        **kwargs: 각 로더의 추가 인자

    Returns:
        로드된 이미지 배열
    """
    path = Path(path)

    # 디렉토리인 경우
    if path.is_dir():
        # NIfTI 파일 먼저 찾기
        nifti_files = list(path.glob("*.nii.gz")) + list(path.glob("*.nii"))
        if nifti_files:
            return load_nifti_directory(path, **kwargs)

        # NumPy 파일
        npy_files = list(path.glob("*.npy"))
        if npy_files:
            # 첫 번째 파일만 로드 (또는 전체 로드 로직 추가)
            return load_numpy(npy_files[0], **kwargs)

        # 이미지 파일
        img_files = list(path.glob("*.png")) + list(path.glob("*.jpg"))
        if img_files:
            return load_images(img_files, **kwargs)

        raise ValueError(f"지원하는 파일이 없습니다: {path}")

    # 파일인 경우
    suffix = path.suffix.lower()

    if suffix == '.gz' and path.stem.endswith('.nii'):
        # .nii.gz
        return load_nifti(path, **kwargs)
    elif suffix == '.nii':
        return load_nifti(path, **kwargs)
    elif suffix == '.npy':
        return load_numpy(path, **kwargs)
    elif suffix in ['.png', '.jpg', '.jpeg']:
        return load_images([path], **kwargs)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix}")


def save_nifti(
    data: np.ndarray,
    output_path: Union[str, Path],
    affine: Optional[np.ndarray] = None,
    header: Optional[object] = None,
) -> None:
    """
    NumPy 배열을 NIfTI 파일로 저장

    Args:
        data: 저장할 데이터 (H, W, D) 또는 (N, H, W) -> (H, W, N)으로 변환
        output_path: 출력 파일 경로
        affine: Affine 변환 행렬 (None이면 identity)
        header: NIfTI 헤더 (None이면 기본값)
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "NIfTI 파일을 저장하려면 'nibabel' 패키지가 필요합니다."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # (N, H, W) -> (H, W, N)으로 변환
    if data.ndim == 3:
        data = np.moveaxis(data, 0, -1)

    # Affine 기본값
    if affine is None:
        affine = np.eye(4)

    # NIfTI 객체 생성
    nii = nib.Nifti1Image(data, affine, header=header)

    # 저장
    nib.save(nii, str(output_path))


def _normalize_array(data: np.ndarray) -> np.ndarray:
    """배열을 0-1로 정규화"""
    data_min = data.min()
    data_max = data.max()

    if data_max - data_min < 1e-10:
        # 모든 값이 같은 경우
        return np.zeros_like(data)

    return (data - data_min) / (data_max - data_min)


def get_metadata_nifti(file_path: Union[str, Path]) -> dict:
    """
    NIfTI 파일의 메타데이터 추출

    Args:
        file_path: NIfTI 파일 경로

    Returns:
        메타데이터 딕셔너리
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel 패키지가 필요합니다.")

    nii = nib.load(str(file_path))

    metadata = {
        'shape': nii.shape,
        'affine': nii.affine,
        'header': dict(nii.header),
        'voxel_size': nii.header.get_zooms(),
        'data_type': nii.get_data_dtype(),
    }

    return metadata


# 편의 함수
def load_real_and_synthetic(
    real_path: Union[str, Path],
    synthetic_path: Union[str, Path],
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    원본과 합성 이미지를 한 번에 로드

    Args:
        real_path: 원본 이미지 경로
        synthetic_path: 합성 이미지 경로
        **kwargs: 로더 인자

    Returns:
        (real_images, synthetic_images) 튜플
    """
    print("📂 원본 이미지 로드 중...")
    real_images = auto_load(real_path, **kwargs)

    print("📂 합성 이미지 로드 중...")
    synthetic_images = auto_load(synthetic_path, **kwargs)

    # Shape 확인
    if real_images.shape != synthetic_images.shape:
        warnings.warn(
            f"Shape 불일치: real={real_images.shape} vs synthetic={synthetic_images.shape}"
        )

    print(f"✓ 로드 완료:")
    print(f"  원본: {real_images.shape}")
    print(f"  합성: {synthetic_images.shape}")

    return real_images, synthetic_images
