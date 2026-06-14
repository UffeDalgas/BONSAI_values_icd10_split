"""
Batch ablation runner: Train and evaluate multiple fine-tuned models from single pretrain.

Design:
  1 pretrain model → N fine-tuned models with different feature subsets
  ↓
  Systematic comparison of model performance

Usage:
    from corebehrt.ablation.batch_ablation_runner import BatchAblationRunner

    runner = BatchAblationRunner(
        pretrain_checkpoint='./outputs/pretraining_dryrun/checkpoints/best.pt',
        output_dir='./outputs/ablation_results',
        n_workers=4
    )

    runner.add_ablation_config('ablation_ehr_only.yaml')
    runner.add_ablation_config('ablation_grim_age.yaml')
    runner.add_ablation_config('ablation_all_features.yaml')

    runner.train_all_models()
    runner.evaluate_all_models()
    runner.generate_comparison_report()
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger("batch_ablation_runner")


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    config_path: str
    features: List[str] = field(default_factory=list)
    description: str = ""

    def __hash__(self):
        return hash(self.name)


class BatchAblationRunner:
    """Orchestrate training and evaluation of multiple ablation models."""

    def __init__(
        self,
        pretrain_checkpoint: str,
        output_dir: str = "./outputs/ablation_results",
        n_workers: int = 1,
    ):
        """
        Args:
            pretrain_checkpoint: Path to pretrain model checkpoint
            output_dir: Where to save all ablation results
            n_workers: Number of parallel workers for training
        """
        self.pretrain_checkpoint = Path(pretrain_checkpoint)
        self.output_dir = Path(output_dir)
        self.n_workers = n_workers

        # Verify pretrain checkpoint exists
        if not self.pretrain_checkpoint.exists():
            raise FileNotFoundError(f"Pretrain checkpoint not found: {self.pretrain_checkpoint}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ablation_configs: List[AblationConfig] = []
        self.results: Dict[str, Dict] = {}

        logger.info(f"BatchAblationRunner initialized")
        logger.info(f"  Pretrain: {self.pretrain_checkpoint}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Workers: {n_workers}")

    def add_ablation_config(
        self,
        name: str,
        config_path: str,
        features: List[str],
        description: str = "",
    ):
        """Add an ablation configuration."""
        config = AblationConfig(
            name=name,
            config_path=config_path,
            features=features,
            description=description,
        )
        self.ablation_configs.append(config)
        logger.info(f"Added ablation: {name} ({len(features)} features)")

    def train_all_models(self) -> Dict[str, bool]:
        """Train all ablation models."""
        if not self.ablation_configs:
            logger.warning("No ablation configs to train")
            return {}

        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING {len(self.ablation_configs)} ABLATION MODELS")
        logger.info(f"{'='*80}\n")

        training_results = {}

        if self.n_workers == 1:
            # Sequential training
            for config in self.ablation_configs:
                success = self._train_single_model(config)
                training_results[config.name] = success
        else:
            # Parallel training
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(self._train_single_model, config): config.name
                    for config in self.ablation_configs
                }

                for future in as_completed(futures):
                    config_name = futures[future]
                    try:
                        success = future.result()
                        training_results[config_name] = success
                    except Exception as e:
                        logger.error(f"Training failed for {config_name}: {e}")
                        training_results[config_name] = False

        # Summary
        n_success = sum(training_results.values())
        logger.info(f"\nTraining complete: {n_success}/{len(training_results)} models trained successfully")

        return training_results

    def _train_single_model(self, config: AblationConfig) -> bool:
        """Train a single ablation model."""
        logger.info(f"\n{'─'*80}")
        logger.info(f"Training: {config.name}")
        logger.info(f"Features: {config.features}")
        logger.info(f"Description: {config.description}")
        logger.info(f"{'─'*80}\n")

        try:
            from corebehrt.main.finetune_cv import main_finetune

            # Train model
            main_finetune(config.config_path)

            logger.info(f"✓ {config.name} training completed")
            return True

        except Exception as e:
            logger.error(f"✗ {config.name} training failed: {e}")
            return False

    def evaluate_all_models(self) -> Dict[str, Dict]:
        """Evaluate all trained models."""
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATING {len(self.ablation_configs)} MODELS")
        logger.info(f"{'='*80}\n")

        for config in self.ablation_configs:
            logger.info(f"\nEvaluating: {config.name}")

            try:
                # Load predictions and compute metrics
                predictions_file = self.output_dir / f"{config.name}_predictions.csv"
                if not predictions_file.exists():
                    logger.warning(f"  Predictions file not found: {predictions_file}")
                    continue

                predictions_df = pd.read_csv(predictions_file)
                predictions = predictions_df["predicted_probability"].values
                labels = predictions_df["true_label"].values

                # Compute comprehensive metrics
                from corebehrt.functional.evaluation.comprehensive_metrics import ComprehensiveModelEvaluation

                evaluator = ComprehensiveModelEvaluation(predictions, labels)
                metrics = evaluator.evaluate_all()

                # Store results
                self.results[config.name] = {
                    "features": config.features,
                    "n_features": len(config.features),
                    "metrics": metrics,
                }

                # Print summary
                logger.info(evaluator.summary_report())

            except Exception as e:
                logger.error(f"  Evaluation failed: {e}")

        return self.results

    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comparison table across all models."""
        if not self.results:
            logger.warning("No results to compare")
            return pd.DataFrame()

        logger.info(f"\n{'='*80}")
        logger.info(f"ABLATION COMPARISON")
        logger.info(f"{'='*80}\n")

        # Extract key metrics for comparison
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result["metrics"]
            comparison_data.append({
                "Model": model_name,
                "Features": ", ".join(result["features"]) or "None",
                "N Features": result["n_features"],
                "ROC-AUC": metrics["roc_auc"],
                "ROC-AUC CI": f"[{metrics['roc_auc_ci_lower']:.3f}-{metrics['roc_auc_ci_upper']:.3f}]",
                "PR-AUC": metrics["pr_auc"],
                "ECE": metrics["ece"],
                "Calib Slope": metrics["calibration_slope"],
                "HL p-value": metrics["hl_pvalue"],
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("ROC-AUC", ascending=False)

        # Display
        logger.info("\n" + comparison_df.to_string(index=False))

        # Calculate deltas
        logger.info("\n" + "─"*80)
        logger.info("DELTA ROC-AUC (vs Baseline)")
        logger.info("─"*80)

        if len(comparison_df) > 0:
            baseline_auc = comparison_df.iloc[-1]["ROC-AUC"]  # Last row (lowest AUC)
            for idx, row in comparison_df.iterrows():
                delta = row["ROC-AUC"] - baseline_auc
                logger.info(f"{row['Model']:40s}: {delta:+.3f}")

        # Save to CSV
        output_csv = self.output_dir / "ablation_comparison.csv"
        comparison_df.to_csv(output_csv, index=False)
        logger.info(f"\n✓ Comparison table saved to: {output_csv}")

        # Save detailed results as JSON
        output_json = self.output_dir / "ablation_results.json"
        with open(output_json, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"✓ Detailed results saved to: {output_json}")

        return comparison_df

    def plot_comparison(self, output_file: Optional[str] = None):
        """Plot ROC curves for all models."""
        if not self.results:
            logger.warning("No results to plot")
            return

        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve

            logger.info(f"\nGenerating ROC comparison plot...")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # ROC-AUC by model
            model_names = list(self.results.keys())
            roc_aucs = [self.results[m]["metrics"]["roc_auc"] for m in model_names]
            n_features = [self.results[m]["n_features"] for m in model_names]

            axes[0].barh(model_names, roc_aucs)
            axes[0].set_xlabel("ROC-AUC")
            axes[0].set_title("Model Comparison by ROC-AUC")
            axes[0].set_xlim([0.5, 1.0])

            for i, (name, auc, nf) in enumerate(zip(model_names, roc_aucs, n_features)):
                axes[0].text(auc + 0.01, i, f"{auc:.3f} ({nf}F)", va="center")

            # ROC-AUC vs number of features (efficiency)
            axes[1].scatter(n_features, roc_aucs, s=100, alpha=0.6)
            for name, nf, auc in zip(model_names, n_features, roc_aucs):
                axes[1].annotate(name, (nf, auc), fontsize=8, alpha=0.7)

            axes[1].set_xlabel("Number of Features")
            axes[1].set_ylabel("ROC-AUC")
            axes[1].set_title("Model Efficiency: ROC-AUC vs Features")
            axes[1].grid(alpha=0.3)

            plt.tight_layout()

            output_path = output_file or self.output_dir / "ablation_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"✓ Plot saved to: {output_path}")

        except ImportError:
            logger.warning("matplotlib not available for plotting")

    def summary(self):
        """Print summary statistics."""
        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH ABLATION SUMMARY")
        logger.info(f"{'='*80}\n")

        logger.info(f"Pretrain checkpoint: {self.pretrain_checkpoint}")
        logger.info(f"Models trained: {len(self.ablation_configs)}")
        logger.info(f"Models evaluated: {len(self.results)}")

        if self.results:
            roc_aucs = [self.results[m]["metrics"]["roc_auc"] for m in self.results.keys()]
            logger.info(f"\nROC-AUC statistics:")
            logger.info(f"  Mean:   {np.mean(roc_aucs):.3f}")
            logger.info(f"  Median: {np.median(roc_aucs):.3f}")
            logger.info(f"  Min:    {np.min(roc_aucs):.3f}")
            logger.info(f"  Max:    {np.max(roc_aucs):.3f}")
            logger.info(f"  Range:  {np.max(roc_aucs) - np.min(roc_aucs):.3f}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of how to use BatchAblationRunner."""

    runner = BatchAblationRunner(
        pretrain_checkpoint="./outputs/pretraining_dryrun/checkpoints/best.pt",
        output_dir="./outputs/ablation_results",
        n_workers=4,
    )

    # Add ablation configurations
    runner.add_ablation_config(
        name="Model 1: EHR Only",
        config_path="./corebehrt/configs/ablation_ehr_only.yaml",
        features=[],
        description="Baseline: no biological features",
    )

    runner.add_ablation_config(
        name="Model 2: GrimAge v2",
        config_path="./corebehrt/configs/ablation_grim_age.yaml",
        features=["grim_age_v2"],
        description="GrimAge v2 only",
    )

    runner.add_ablation_config(
        name="Model 3: SystemsAge",
        config_path="./corebehrt/configs/ablation_systems_age.yaml",
        features=["systems_age"] + [f"systems_age_{i}" for i in range(1, 12)],
        description="SystemsAge (11 components)",
    )

    runner.add_ablation_config(
        name="Model 4: GrimAge + SystemsAge",
        config_path="./corebehrt/configs/ablation_combined.yaml",
        features=["grim_age_v2", "systems_age"],
        description="Both aging clocks",
    )

    # Train all models
    training_results = runner.train_all_models()
    print(f"Training results: {training_results}")

    # Evaluate all models
    eval_results = runner.evaluate_all_models()
    print(f"Evaluation complete: {len(eval_results)} models")

    # Generate comparison report
    comparison_df = runner.generate_comparison_report()
    print(comparison_df)

    # Plot
    runner.plot_comparison()

    # Summary
    runner.summary()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    )
    example_usage()
