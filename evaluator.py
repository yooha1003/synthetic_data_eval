"""
MRI Synthesis Evaluator

모든 평가 지표를 통합하여 사용하기 쉬운 인터페이스를 제공합니다.
"""

import numpy as np
from typing import Optional, Dict, List
import warnings
from pathlib import Path
import json

from metrics.image_quality import calculate_all_image_quality_metrics
from metrics.perceptual import calculate_all_perceptual_metrics
from metrics.distribution import calculate_all_distribution_metrics
from metrics.structural import calculate_all_structural_metrics
from metrics.segmentation import calculate_all_segmentation_metrics


class MRIEvaluator:
    """
    MRI 합성 데이터 평가기

    모든 평가 지표를 한 번에 계산하고, 결과를 보고서로 출력합니다.

    Example:
        >>> evaluator = MRIEvaluator()
        >>> results = evaluator.evaluate_all(real_images, synthetic_images)
        >>> evaluator.print_report(results)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        lpips_net: str = 'alex',
        verbose: bool = True,
    ):
        """
        Args:
            device: 계산 디바이스 ('cpu', 'cuda')
            lpips_net: LPIPS 백본 네트워크 ('alex', 'vgg', 'squeeze')
            verbose: 진행 상황 출력 여부
        """
        self.device = device
        self.lpips_net = lpips_net
        self.verbose = verbose

    def evaluate_all(
        self,
        real_images: np.ndarray,
        synthetic_images: np.ndarray,
        include_perceptual: bool = True,
        include_distribution: bool = True,
        include_structural: bool = True,
        include_segmentation: bool = False,
        segmentation_kwargs: Optional[dict] = None,
    ) -> Dict:
        """
        모든 평가 지표를 계산

        Args:
            real_images: 원본 이미지 (N, H, W) or (N, H, W, C)
            synthetic_images: 합성 이미지 (M, H, W) or (M, H, W, C)
            include_perceptual: LPIPS, VIF 계산 여부
            include_distribution: FID, MMD 계산 여부
            include_structural: Gradient, Edge, Frequency 계산 여부
            include_segmentation: Dice, Hausdorff 계산 여부
            segmentation_kwargs: 세그멘테이션 관련 인자

        Returns:
            모든 지표를 포함하는 딕셔너리
        """
        if real_images.shape != synthetic_images.shape:
            warnings.warn(
                f"Shape mismatch: {real_images.shape} vs {synthetic_images.shape}. "
                f"일부 지표는 계산되지 않을 수 있습니다."
            )

        results = {}

        # 1. Image Quality Metrics
        if self.verbose:
            print("📊 이미지 품질 지표 계산 중 (SSIM, MS-SSIM, PSNR)...")
        try:
            quality_results = calculate_all_image_quality_metrics(
                real_images, synthetic_images
            )
            results.update(quality_results)
            if self.verbose:
                print("  ✓ 완료")
        except Exception as e:
            warnings.warn(f"이미지 품질 지표 계산 실패: {e}")

        # 2. Perceptual Metrics
        if include_perceptual:
            if self.verbose:
                print("👁️  지각적 유사도 지표 계산 중 (LPIPS, VIF)...")
            try:
                perceptual_results = calculate_all_perceptual_metrics(
                    real_images, synthetic_images,
                    lpips_net=self.lpips_net,
                    device=self.device,
                )
                results.update(perceptual_results)
                if self.verbose:
                    print("  ✓ 완료")
            except Exception as e:
                warnings.warn(f"지각적 지표 계산 실패: {e}")

        # 3. Distribution Metrics
        if include_distribution:
            if self.verbose:
                print("📈 분포 유사도 지표 계산 중 (FID, MMD)...")
            try:
                distribution_results = calculate_all_distribution_metrics(
                    real_images, synthetic_images,
                    device=self.device,
                )
                results.update(distribution_results)
                if self.verbose:
                    print("  ✓ 완료")
            except Exception as e:
                warnings.warn(f"분포 지표 계산 실패: {e}")

        # 4. Structural Metrics
        if include_structural:
            if self.verbose:
                print("🔬 구조 특화 지표 계산 중 (Gradient, Edge, Frequency)...")
            try:
                structural_results = calculate_all_structural_metrics(
                    real_images, synthetic_images
                )
                results.update(structural_results)
                if self.verbose:
                    print("  ✓ 완료")
            except Exception as e:
                warnings.warn(f"구조 지표 계산 실패: {e}")

        # 5. Segmentation-based Metrics
        if include_segmentation:
            if self.verbose:
                print("🧠 세그멘테이션 기반 지표 계산 중 (Dice, Hausdorff)...")
            try:
                if segmentation_kwargs is None:
                    segmentation_kwargs = {}

                segmentation_results = calculate_all_segmentation_metrics(
                    real_images, synthetic_images,
                    **segmentation_kwargs
                )
                results.update(segmentation_results)
                if self.verbose:
                    print("  ✓ 완료")
            except Exception as e:
                warnings.warn(f"세그멘테이션 지표 계산 실패: {e}")

        return results

    def print_report(self, results: Dict, highlight_issues: bool = True):
        """
        평가 결과를 보기 좋게 출력

        Args:
            results: evaluate_all()의 결과
            highlight_issues: 문제가 있는 지표 강조 표시
        """
        print("\n" + "="*70)
        print("🏥 MRI 합성 데이터 평가 보고서")
        print("="*70)

        # 1. Image Quality
        print("\n📊 이미지 품질 지표 (Image Quality Metrics)")
        print("-" * 70)
        self._print_metric("SSIM", results.get('ssim'), "↑", 0.8, highlight_issues)
        self._print_metric("MS-SSIM", results.get('ms_ssim'), "↑", 0.8, highlight_issues)
        self._print_metric("PSNR", results.get('psnr'), "↑", 25.0, highlight_issues, unit="dB")

        # 2. Perceptual
        print("\n👁️  지각적 유사도 지표 (Perceptual Metrics) ⭐")
        print("-" * 70)
        self._print_metric("LPIPS", results.get('lpips'), "↓", 0.2, highlight_issues, lower_better=True)
        self._print_metric("VIF", results.get('vif'), "↑", 0.5, highlight_issues)

        # 3. Distribution
        print("\n📈 분포 유사도 지표 (Distribution Metrics)")
        print("-" * 70)
        self._print_metric("FID", results.get('fid'), "↓", 50.0, highlight_issues, lower_better=True)
        self._print_metric("MMD", results.get('mmd'), "↓", 0.1, highlight_issues, lower_better=True)

        # 4. Structural (⭐⭐ 중요!)
        print("\n🔬 구조 특화 지표 (Structure-Specific Metrics) ⭐⭐")
        print("-" * 70)
        print("   [내부 조직 구조 품질 평가에 핵심 지표들]")
        self._print_metric(
            "Gradient Similarity",
            results.get('gradient_similarity'),
            "↑", 0.7, highlight_issues
        )
        self._print_metric(
            "Edge Preservation",
            results.get('edge_preservation'),
            "↑", 0.7, highlight_issues
        )
        self._print_metric(
            "High-Freq Distance",
            results.get('freq_high_distance'),
            "↓", None, highlight_issues, lower_better=True
        )
        self._print_metric(
            "High-Freq Ratio",
            results.get('freq_high_ratio'),
            "~1.0", 1.0, highlight_issues, tolerance=0.2
        )

        # 5. Segmentation
        if any(k.startswith('dice_') for k in results.keys()):
            print("\n🧠 세그멘테이션 기반 지표 (Segmentation-Based Metrics)")
            print("-" * 70)
            for key, value in results.items():
                if key.startswith('dice_'):
                    label = key.replace('dice_', '')
                    self._print_metric(f"Dice ({label})", value, "↑", 0.7, highlight_issues)
            for key, value in results.items():
                if key.startswith('hausdorff_'):
                    label = key.replace('hausdorff_', '')
                    self._print_metric(
                        f"Hausdorff ({label})",
                        value, "↓", 5.0, highlight_issues,
                        lower_better=True, unit="px"
                    )

        # 종합 의견
        print("\n💡 종합 의견")
        print("-" * 70)
        self._print_summary(results)

        print("\n" + "="*70)

    def _print_metric(
        self,
        name: str,
        value: Optional[float],
        direction: str,
        threshold: Optional[float],
        highlight: bool,
        lower_better: bool = False,
        tolerance: float = 0.0,
        unit: str = "",
    ):
        """개별 지표 출력"""
        if value is None:
            print(f"  {name:25s}: N/A")
            return

        # 값 포맷팅
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)

        if unit:
            value_str += f" {unit}"

        # 상태 표시
        status = ""
        if highlight and threshold is not None:
            if lower_better:
                if value > threshold:
                    status = " ⚠️  (높음)"
            else:
                if tolerance > 0:
                    if abs(value - threshold) > tolerance:
                        status = " ⚠️  (편차 큼)"
                else:
                    if value < threshold:
                        status = " ⚠️  (낮음)"

        print(f"  {name:25s} {direction:4s}: {value_str:15s}{status}")

    def _print_summary(self, results: Dict):
        """종합 의견 출력"""
        issues = []

        # LPIPS 체크
        lpips = results.get('lpips')
        if lpips is not None and lpips > 0.2:
            issues.append(
                f"LPIPS({lpips:.3f})가 높습니다. "
                f"미세한 조직 구조가 원본과 다를 수 있습니다."
            )

        # Gradient similarity 체크
        grad_sim = results.get('gradient_similarity')
        if grad_sim is not None and grad_sim < 0.7:
            issues.append(
                f"Gradient Similarity({grad_sim:.3f})가 낮습니다. "
                f"조직 경계가 흐릿하거나 부정확할 수 있습니다."
            )

        # High-frequency 체크
        high_freq_ratio = results.get('freq_high_ratio')
        if high_freq_ratio is not None and abs(high_freq_ratio - 1.0) > 0.2:
            issues.append(
                f"고주파 비율({high_freq_ratio:.3f})이 1.0에서 벗어났습니다. "
                f"세부 조직 구조가 손실되거나 과장되었을 수 있습니다."
            )

        if issues:
            print("  🚨 발견된 문제점:")
            for i, issue in enumerate(issues, 1):
                print(f"     {i}. {issue}")
        else:
            print("  ✅ 모든 지표가 양호한 범위에 있습니다.")

        print("\n  📝 논문 보고 권장 지표:")
        print("     필수: SSIM, PSNR, FID")
        print("     조직 구조 강조: LPIPS, Gradient Similarity, High-Freq Ratio")
        print("     해부학적 정확도: Dice, Hausdorff (세그멘테이션 후)")

    def export_json(self, results: Dict, output_path: str):
        """결과를 JSON으로 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # None 값을 null로 변환
        cleaned_results = {
            k: (v if v is not None else None)
            for k, v in results.items()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"\n💾 결과 저장: {output_path}")

    def export_latex_table(self, results: Dict, output_path: str):
        """논문용 LaTeX 표 생성"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        latex_lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Evaluation Results for MRI Synthesis}",
            r"\label{tab:evaluation}",
            r"\begin{tabular}{lcc}",
            r"\hline",
            r"Metric & Direction & Value \\",
            r"\hline",
        ]

        # 주요 지표들
        metrics_to_include = [
            ('SSIM', 'ssim', r'$\uparrow$'),
            ('MS-SSIM', 'ms_ssim', r'$\uparrow$'),
            ('PSNR (dB)', 'psnr', r'$\uparrow$'),
            ('LPIPS', 'lpips', r'$\downarrow$'),
            ('FID', 'fid', r'$\downarrow$'),
            ('Gradient Similarity', 'gradient_similarity', r'$\uparrow$'),
            ('Edge Preservation', 'edge_preservation', r'$\uparrow$'),
            ('High-Freq Ratio', 'freq_high_ratio', r'$\sim$1.0'),
        ]

        for name, key, direction in metrics_to_include:
            value = results.get(key)
            if value is not None:
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                latex_lines.append(f"{name} & {direction} & {value_str} \\\\")

        latex_lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_lines))

        if self.verbose:
            print(f"\n📄 LaTeX 표 저장: {output_path}")
