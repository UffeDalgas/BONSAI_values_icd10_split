"""
Feature comparison analysis module.

Analyzes and compares ROC-AUC across ablation experiments to estimate
feature contributions to mortality prediction.

Usage:
    python -m corebehrt.analysis.feature_comparison --input outputs/ --output results/
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger("feature_comparison")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)


@dataclass
class AblationMetrics:
    """Metrics for a single ablation."""
    name: str
    features: List[str]
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    accuracy: float


class FeatureComparison:
    """Compare feature contributions across ablations."""

    def __init__(self, output_base: str = "./outputs"):
        self.output_base = Path(output_base)
        self.ablations: Dict[str, AblationMetrics] = {}
        self.comparison_df = None

    def load_ablation_results(self, ablation_name: str) -> bool:
        """Load metrics from a completed ablation."""
        ablation_dir = self.output_base / f"finetuning_ablation_{ablation_name}"

        if not ablation_dir.exists():
            logger.warning(f"Ablation directory not found: {ablation_dir}")
            return False

        # Look for cv_summary.json or fold-specific metrics
        cv_summary_path = ablation_dir / "cv_summary.json"

        if cv_summary_path.exists():
            try:
                with open(cv_summary_path) as f:
                    metrics_data = json.load(f)

                # Parse metrics (structure depends on BONSAI output format)
                self.ablations[ablation_name] = self._parse_metrics(
                    ablation_name, metrics_data
                )
                logger.info(f"Loaded metrics for ablation: {ablation_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to parse metrics for {ablation_name}: {e}")
                return False

        else:
            logger.warning(f"No cv_summary.json found in {ablation_dir}")
            return False

    def _parse_metrics(self, name: str, metrics_data: Dict) -> AblationMetrics:
        """Parse metrics from BONSAI output format."""
        # This depends on actual BONSAI cv_summary structure
        # Placeholder implementation - adjust based on actual format
        features = self._infer_features_from_name(name)

        roc_auc = metrics_data.get("roc_auc", {})
        roc_auc_mean = roc_auc.get("mean", 0.5)
        roc_auc_std = roc_auc.get("std", 0.0)

        pr_auc = metrics_data.get("pr_auc", {})
        pr_auc_mean = pr_auc.get("mean", 0.5)
        pr_auc_std = pr_auc.get("std", 0.0)

        accuracy = metrics_data.get("accuracy", {}).get("mean", 0.5)

        return AblationMetrics(
            name=name,
            features=features,
            roc_auc_mean=roc_auc_mean,
            roc_auc_std=roc_auc_std,
            pr_auc_mean=pr_auc_mean,
            pr_auc_std=pr_auc_std,
            accuracy=accuracy,
        )

    def _infer_features_from_name(self, name: str) -> List[str]:
        """Infer which features are included based on ablation name."""
        feature_map = {
            "baseline": [],
            "clocks": ["clocks"],
            "proteins": ["proteins"],
            "maple": ["maple"],
            "methylgpt": ["methylgpt"],
            "all": ["clocks", "proteins", "maple", "methylgpt"],
        }
        return feature_map.get(name, [])

    def compute_feature_contributions(self) -> pd.DataFrame:
        """Compute feature importance based on ROC-AUC improvements."""
        if not self.ablations:
            logger.error("No ablations loaded")
            return None

        # Create comparison dataframe
        ablation_list = []
        for name, metrics in self.ablations.items():
            ablation_list.append({
                "ablation": name,
                "features": ", ".join(metrics.features) if metrics.features else "none (EHR only)",
                "roc_auc_mean": metrics.roc_auc_mean,
                "roc_auc_std": metrics.roc_auc_std,
                "pr_auc_mean": metrics.pr_auc_mean,
                "accuracy": metrics.accuracy,
            })

        self.comparison_df = pd.DataFrame(ablation_list)

        # Sort by ROC-AUC
        self.comparison_df = self.comparison_df.sort_values("roc_auc_mean", ascending=False)

        # Compute deltas from baseline
        if "baseline" in self.comparison_df["ablation"].values:
            baseline_roc = self.comparison_df[
                self.comparison_df["ablation"] == "baseline"
            ]["roc_auc_mean"].values[0]

            self.comparison_df["delta_roc_auc"] = (
                self.comparison_df["roc_auc_mean"] - baseline_roc
            )

        return self.comparison_df

    def summarize_contributions(self) -> Dict:
        """Summarize individual feature contributions."""
        if self.comparison_df is None:
            self.compute_feature_contributions()

        if self.comparison_df is None:
            logger.error("Failed to compute contributions")
            return {}

        summary = {
            "results_by_ablation": self.comparison_df.to_dict("records"),
            "top_features": self._identify_top_features(),
            "recommendations": self._generate_recommendations(),
        }

        return summary

    def _identify_top_features(self) -> List[Dict]:
        """Identify which features provide most improvement."""
        if self.comparison_df is None or "delta_roc_auc" not in self.comparison_df.columns:
            return []

        # Filter single-feature ablations
        single_feature = self.comparison_df[
            self.comparison_df["features"] != "none (EHR only)"
        ].copy()

        # For now, just return comparison
        # In future: compute statistical significance via DeLong test
        feature_contributions = []

        for _, row in single_feature.iterrows():
            feature_contributions.append({
                "feature_set": row["features"],
                "roc_auc_improvement": row.get("delta_roc_auc", 0),
                "absolute_roc_auc": row["roc_auc_mean"],
            })

        return sorted(
            feature_contributions,
            key=lambda x: x["roc_auc_improvement"],
            reverse=True,
        )

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results."""
        if self.comparison_df is None:
            return []

        recommendations = []

        # Get best performing ablation
        best_ablation = self.comparison_df.iloc[0]["ablation"]
        best_roc = self.comparison_df.iloc[0]["roc_auc_mean"]

        if best_ablation != "baseline":
            recommendations.append(
                f"Best performance with {best_ablation} (ROC-AUC={best_roc:.3f})"
            )

        # Check if all features is best
        if "all" in self.comparison_df["ablation"].values:
            all_row = self.comparison_df[self.comparison_df["ablation"] == "all"].iloc[0]
            baseline_row = self.comparison_df[self.comparison_df["ablation"] == "baseline"].iloc[0]

            improvement = all_row["roc_auc_mean"] - baseline_row["roc_auc_mean"]
            if improvement > 0.01:
                recommendations.append(
                    f"Combining all features improves ROC-AUC by {improvement:.4f}"
                )
            else:
                recommendations.append(
                    "Combining all features provides minimal improvement - "
                    "consider using smaller feature set"
                )

        return recommendations

    def plot_comparison(self, output_path: str = None) -> None:
        """Plot ROC-AUC comparison across ablations."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
            return

        if self.comparison_df is None:
            self.compute_feature_contributions()

        fig, ax = plt.subplots(figsize=(10, 6))

        x_pos = np.arange(len(self.comparison_df))
        roc_auc = self.comparison_df["roc_auc_mean"].values
        roc_auc_std = self.comparison_df["roc_auc_std"].values
        ablations = self.comparison_df["ablation"].values

        # Plot bars with error bars
        ax.bar(x_pos, roc_auc, yerr=roc_auc_std, capsize=5, alpha=0.7)

        # Customize plot
        ax.set_xlabel("Ablation", fontsize=12)
        ax.set_ylabel("ROC-AUC", fontsize=12)
        ax.set_title("Feature Contribution to Mortality Prediction", fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ablations, rotation=45, ha="right")
        ax.set_ylim([0.4, 1.0])
        ax.grid(axis="y", alpha=0.3)

        # Add horizontal line for baseline
        if "baseline" in ablations:
            baseline_idx = np.where(ablations == "baseline")[0][0]
            baseline_roc = roc_auc[baseline_idx]
            ax.axhline(y=baseline_roc, color="r", linestyle="--", alpha=0.5, label="Baseline")
            ax.legend()

        plt.tight_layout()

        if output_path is None:
            output_path = "./outputs/ablation_comparison.png"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comparison plot: {output_path}")
        plt.close()

    def print_summary(self) -> None:
        """Print comparison summary."""
        if self.comparison_df is None:
            self.compute_feature_contributions()

        logger.info("\n")
        logger.info("╔" + "="*78 + "╗")
        logger.info("║" + " "*15 + "FEATURE ABLATION COMPARISON RESULTS" + " "*29 + "║")
        logger.info("╚" + "="*78 + "╝")

        logger.info("\nROC-AUC by Ablation:")
        logger.info("-" * 80)

        for _, row in self.comparison_df.iterrows():
            roc_str = f"{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}"
            features_str = row["features"]
            delta_str = f"({row.get('delta_roc_auc', 0):+.4f})" if "delta_roc_auc" in self.comparison_df.columns else ""

            logger.info(f"  {row['ablation']:15} {roc_str:20} {features_str:35} {delta_str}")

        # Summary
        summary = self.summarize_contributions()
        logger.info("\nTop Features by Contribution:")
        logger.info("-" * 80)

        for feature_info in summary["top_features"]:
            logger.info(
                f"  {feature_info['feature_set']:20} +{feature_info['roc_auc_improvement']:.4f} "
                f"(absolute: {feature_info['absolute_roc_auc']:.4f})"
            )

        logger.info("\nRecommendations:")
        logger.info("-" * 80)
        for rec in summary["recommendations"]:
            logger.info(f"  • {rec}")


def main():
    """Main entry point for feature comparison analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare feature ablation results")
    parser.add_argument(
        "--input",
        default="./outputs",
        help="Base output directory with ablation results",
    )
    parser.add_argument(
        "--output",
        default="./outputs",
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["baseline", "clocks", "proteins", "maple", "methylgpt", "all"],
        help="Which ablations to compare",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plot",
    )

    args = parser.parse_args()

    # Load results
    comparison = FeatureComparison(output_base=args.input)

    for ablation_name in args.ablations:
        comparison.load_ablation_results(ablation_name)

    # Analyze
    comparison.compute_feature_contributions()
    summary = comparison.summarize_contributions()

    # Print summary
    comparison.print_summary()

    # Plot if requested
    if args.plot:
        plot_path = Path(args.output) / "ablation_comparison.png"
        comparison.plot_comparison(str(plot_path))

    # Save detailed results
    results_path = Path(args.output) / "ablation_comparison.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nResults saved to {results_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
