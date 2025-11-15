# MRI 합성 데이터 평가 가이드

## 📌 개요

이 프레임워크는 Diffusion Model로 생성한 MRI 합성 데이터의 품질을 종합적으로 평가합니다.

특히 **"생성물이 원본과 유사해 보이지만 내부 조직의 구조가 안 좋은"** 문제를 정량화하는 데 특화되어 있습니다.

## 🎯 주요 특징

### 1. 전통적 품질 지표
- **SSIM, MS-SSIM, PSNR**: 전반적인 이미지 품질

### 2. 지각적 유사도 (⭐ 추천)
- **LPIPS**: SSIM보다 인간의 지각과 일치, 미세한 조직 차이 감지
- **VIF**: 시각적 정보 충실도

### 3. 분포 유사도
- **FID, MMD**: 전체 데이터셋의 분포 유사도

### 4. 구조 특화 지표 (⭐⭐ 강력 추천)
- **Gradient Similarity**: 조직 경계의 선명도
- **Edge Preservation**: 경계 보존 정도
- **Frequency Analysis**: 고주파 = 세부 구조, 저주파 = 전체 형태

### 5. 해부학적 정확도
- **Dice Coefficient**: 특정 조직의 구조적 일치도
- **Hausdorff Distance**: 경계의 정확도

## 🚀 빠른 시작

### 설치

```bash
pip install -r requirements.txt
```

### 기본 사용법

```python
from evaluator import MRIEvaluator
import numpy as np

# 이미지 로드
real_images = np.load('real_mri.npy')  # (N, H, W) or (N, H, W, C)
synthetic_images = np.load('synthetic_mri.npy')

# 평가
evaluator = MRIEvaluator()
results = evaluator.evaluate_all(real_images, synthetic_images)

# 결과 출력
evaluator.print_report(results)

# 저장
evaluator.export_json(results, 'results.json')
evaluator.export_latex_table(results, 'table.tex')
```

## 📊 지표 해석 가이드

### 내부 조직 구조 문제 진단

#### 증상: "원본과 유사해 보이지만 조직 구조가 안 좋음"

**확인할 지표:**

1. **LPIPS** (↓ 낮을수록 좋음)
   - **0.0-0.1**: 매우 우수
   - **0.1-0.2**: 양호
   - **0.2-0.3**: 보통 (⚠️ 미세한 구조 차이 존재)
   - **0.3+**: 나쁨 (⚠️⚠️ 구조적 차이 큼)

2. **Gradient Similarity** (↑ 높을수록 좋음)
   - **0.8-1.0**: 경계가 선명
   - **0.7-0.8**: 양호
   - **0.5-0.7**: 보통 (⚠️ 경계가 약간 흐릿)
   - **0.5 미만**: 나쁨 (⚠️⚠️ 경계 손실)

3. **High-Freq Ratio** (1.0에 가까울수록 좋음)
   - **0.9-1.1**: 세부 구조 잘 보존
   - **0.8-0.9 또는 1.1-1.2**: 양호
   - **0.7-0.8 또는 1.2-1.3**: 보통 (⚠️ 세부 구조 변화)
   - **0.7 미만 또는 1.3 초과**: 문제 (⚠️⚠️ 세부 구조 손실/과장)

### 문제별 해결 방안

#### 1. LPIPS가 높은 경우 (>0.2)
**문제**: 미세한 조직 패턴이 원본과 다름

**해결 방안**:
- Perceptual loss (LPIPS loss)를 훈련에 추가
- Diffusion timestep 증가
- 고해상도 학습

#### 2. Gradient Similarity가 낮은 경우 (<0.7)
**문제**: 조직 경계가 흐릿함

**해결 방안**:
- Edge-preserving loss 추가
- Diffusion noise schedule 조정
- Guidance scale 조정

#### 3. High-Freq Ratio가 1.0에서 멀 경우
**문제**:
- <1.0: 세부 구조 손실 (과도한 평활화)
- \>1.0: 세부 구조 과장 (노이즈)

**해결 방안**:
- Loss에 frequency domain constraint 추가
- Diffusion step 수 조정
- Denoising 강도 조정

## 📝 논문 보고 가이드

### 필수 포함 지표

모든 MRI 합성 논문에서 보고해야 할 지표:
- **SSIM**
- **PSNR**
- **FID**

### 조직 구조 품질 강조 시

내부 조직의 정확도를 강조하는 경우:
- **LPIPS** ⭐
- **Gradient Similarity** ⭐
- **High-Freq Ratio** ⭐
- 방사선 전문의 평가 (Likert scale)

### 특정 해부학적 구조 평가 시

특정 조직(회백질, 백질 등)의 정확도:
- **Dice Coefficient** (세그멘테이션 후)
- **Hausdorff Distance** (세그멘테이션 후)

### LaTeX 표 예시

```latex
\begin{table}[h]
\centering
\caption{Evaluation Results for MRI Synthesis}
\begin{tabular}{lcc}
\hline
Metric & Direction & Value \\
\hline
SSIM & $\uparrow$ & 0.8234 \\
PSNR (dB) & $\uparrow$ & 28.45 \\
FID & $\downarrow$ & 12.34 \\
LPIPS & $\downarrow$ & 0.156 \\
Gradient Similarity & $\uparrow$ & 0.782 \\
High-Freq Ratio & $\sim$1.0 & 0.945 \\
\hline
\end{tabular}
\end{table}
```

### 텍스트 설명 예시

```
Our synthetic MRI images achieved high structural similarity
(SSIM=0.823) and perceptual quality (LPIPS=0.156).

Importantly, the tissue microstructure was well preserved,
as evidenced by high gradient similarity (0.782) and
appropriate high-frequency content (ratio=0.945),
indicating that fine anatomical details were faithfully
reproduced.
```

## 🔬 고급 사용법

### 1. 특정 지표만 계산

```python
from metrics.perceptual import calculate_lpips
from metrics.structural import calculate_gradient_similarity

# LPIPS만
lpips = calculate_lpips(real_images, synthetic_images)

# Gradient similarity만
grad_sim = calculate_gradient_similarity(real_images, synthetic_images)
```

### 2. 세그멘테이션 기반 평가

```python
# 세그멘테이션 맵 준비 (N, H, W)
# Label: 0=배경, 1=회백질, 2=백질, 3=CSF
real_seg = segment_with_model(real_images)
synth_seg = segment_with_model(synthetic_images)

# 평가
results = evaluator.evaluate_all(
    real_images, synthetic_images,
    include_segmentation=True,
    segmentation_kwargs={
        'real_segmentations': real_seg,
        'synthetic_segmentations': synth_seg,
        'labels': [1, 2, 3]
    }
)
```

### 3. 배치 실험 비교

```python
experiments = ['DDPM_50', 'DDPM_100', 'DDPM_200']
results_all = {}

for exp in experiments:
    synth = load_synthetic(exp)
    results = evaluator.evaluate_all(real, synth)
    results_all[exp] = results

# 최고 성능 찾기
best = max(results_all.items(),
           key=lambda x: x[1]['gradient_similarity'])
```

## ⚠️ 주의사항

### 1. 데이터 형식
- 입력 이미지: `(N, H, W)` 또는 `(N, H, W, C)`
- 값 범위: 임의 (자동으로 정규화됨)
- dtype: `float32` 또는 `float64` 권장

### 2. 메모리
- FID, MMD는 메모리 집약적 (큰 데이터셋은 배치 처리)
- LPIPS는 GPU 사용 권장 (`device='cuda'`)

### 3. 계산 시간
- 빠름 (< 1초): SSIM, PSNR, Gradient, Edge
- 보통 (~ 10초): LPIPS, VIF, Frequency
- 느림 (~ 1분): FID, MMD (큰 데이터셋)

## 📚 참고 문헌

### 주요 지표 논문

- **SSIM**: Wang et al., "Image quality assessment: from error visibility to structural similarity", IEEE TIP 2004
- **MS-SSIM**: Wang et al., "Multi-scale structural similarity for image quality assessment", IEEE Asilomar 2003
- **LPIPS**: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018
- **FID**: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", NeurIPS 2017

### MRI 합성 관련

- [Awesome Diffusion Models in Medical Imaging](https://github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging)
- [Medical Image Synthesis Survey](https://arxiv.org/abs/2308.00402)

## 🤝 기여

이슈나 개선 제안은 GitHub Issues에 등록해주세요.

## 📧 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록하거나 담당자에게 연락주세요.
