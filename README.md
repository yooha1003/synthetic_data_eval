# MRI Synthetic Data Evaluation Framework

Diffusion model로 생성한 MRI 합성 데이터의 품질을 종합적으로 평가하기 위한 프레임워크입니다.

## 주요 특징

이 프레임워크는 MRI 합성 이미지의 **내부 조직 구조**까지 정밀하게 평가할 수 있도록 다양한 지표를 제공합니다.

### 📊 구현된 평가 지표

#### 1. 이미지 품질 지표 (Image Quality Metrics)
- **SSIM** (Structural Similarity Index Measure): 구조적 유사도
- **MS-SSIM** (Multi-Scale SSIM): 다중 해상도 구조적 유사도
- **PSNR** (Peak Signal-to-Noise Ratio): 신호 대 잡음비

#### 2. 지각적 유사도 지표 (Perceptual Metrics) ⭐
- **LPIPS** (Learned Perceptual Image Patch Similarity): 인간의 지각과 일치하는 평가
- **VIF** (Visual Information Fidelity): 시각적 정보 충실도

#### 3. 분포 유사도 지표 (Distribution Metrics)
- **FID** (Fréchet Inception Distance): 이미지 분포 간 거리
- **MMD** (Maximum Mean Discrepancy): 통계적 분포 차이

#### 4. 구조 특화 지표 (Structure-Specific Metrics) ⭐⭐
- **Gradient Similarity**: 조직 경계 및 구조의 gradient 유사도
- **Edge Preservation**: 경계 보존 정도
- **Frequency Domain Analysis**: 주파수 도메인 유사도 (고주파=세부 구조)

#### 5. 세그멘테이션 기반 지표 (Segmentation-Based Metrics)
- **Dice Coefficient**: 해부학적 구조 일치도
- **Hausdorff Distance**: 구조 경계 간 최대 거리

## 설치

```bash
pip install -r requirements.txt
```

## 빠른 시작

```python
from evaluator import MRIEvaluator
import numpy as np

# 평가기 초기화
evaluator = MRIEvaluator()

# 이미지 로드 (예시)
real_images = np.load('real_mri.npy')  # shape: (N, H, W) or (N, H, W, C)
synthetic_images = np.load('synthetic_mri.npy')

# 전체 지표 평가
results = evaluator.evaluate_all(real_images, synthetic_images)

# 결과 출력
evaluator.print_report(results)

# 논문용 LaTeX 표 생성
evaluator.export_latex_table(results, 'results_table.tex')
```

## 내부 조직 구조 평가에 중요한 지표

생성된 MRI가 **원본과 유사해 보이지만 내부 조직 구조가 안 좋은 경우**, 다음 지표들에 주목하세요:

1. **LPIPS** (↓낮을수록 좋음): SSIM보다 미세한 구조 차이를 더 잘 감지
2. **Gradient Similarity** (↑높을수록 좋음): 조직 경계의 선명도 평가
3. **High-Frequency Power Ratio**: 세부 구조의 보존 정도
4. **Dice Coefficient** (세그멘테이션 후): 특정 조직의 구조적 정확도

## 사용 예시

자세한 사용법은 `example_usage.py`를 참고하세요.

## 논문 보고 권장사항

### 필수 지표 (모든 논문에서 보고)
- SSIM, PSNR, FID

### 조직 구조 품질 강조 시
- LPIPS, Gradient Similarity, Frequency Analysis
- 방사선 전문의 평가 (Likert scale)

### 특정 해부학적 구조 평가 시
- Dice Coefficient, Hausdorff Distance (세그멘테이션 후)

## 인용

이 프레임워크를 사용하신 경우, 각 지표의 원 논문을 인용해주세요:
- SSIM: Wang et al., IEEE TIP 2004
- LPIPS: Zhang et al., CVPR 2018
- FID: Heusel et al., NeurIPS 2017
- MS-SSIM: Wang et al., IEEE Asilomar 2003

## 참고 문헌

조사하신 자료들을 바탕으로 구현되었습니다:
- [Medical Image Synthesis Survey](https://arxiv.org/abs/2308.00402)
- [Diffusion Models in Medical Imaging](https://github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging)
