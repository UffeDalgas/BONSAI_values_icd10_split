"""
Feature ablation experiment runner.

Runs progressive feature addition experiments to measure contribution of:
- Epigenetic clocks
- EpiScore proteins
- MAPLE embeddings
- MethylGPT embeddings

Usage:
    python -m corebehrt.main.feature_ablation_runner --ablations clocks proteins all
    python -m corebehrt.main.feature_ablation_runner --baseline-only  # Just baseline
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict
import json
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger("feature_ablation")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)

PRETRAIN_MODEL = "./outputs/pretraining_dryrun"
PREPARED_DATA = "./outputs/finetuning/processed_data_with_values/"


@dataclass
class AblationConfig:
    """Ablation experiment configuration."""
    name: str
    description: str
    features: List[str]  # ["clocks", "proteins", "maple", "methylgpt"]
    epochs: int
    batch_size: int
    early_stopping: int
    output_dir: str

    def to_dict(self) -> Dict:
        return asdict(self)


class FeatureAblationRunner:
    """Run feature ablation experiments."""

    def __init__(self, pretrain_model: str = PRETRAIN_MODEL, prepared_data: str = PREPARED_DATA):
        self.pretrain_model = pretrain_model
        self.prepared_data = prepared_data
        self.ablations = {}
        self.results = {}

    def register_ablation(self, config: AblationConfig) -> None:
        """Register an ablation experiment."""
        self.ablations[config.name] = config
        logger.info(f"Registered ablation: {config.name} ({config.description})")

    def setup_standard_ablations(self, epochs: int = 10, batch_size: int = 16, early_stopping: int = 5):
        """Setup standard feature ablation progression."""

        ablations = [
            AblationConfig(
                name="baseline",
                description="EHR only (no biological features)",
                features=[],
                epochs=epochs,
                batch_size=batch_size,
                early_stopping=early_stopping,
                output_dir="./outputs/finetuning_ablation_baseline",
            ),
            AblationConfig(
                name="clocks",
                description="+ Epigenetic clocks",
                features=["clocks"],
                epochs=epochs,
                batch_size=batch_size,
                early_stopping=early_stopping,
                output_dir="./outputs/finetuning_ablation_clocks",
            ),
            AblationConfig(
                name="proteins",
                description="+ EpiScore proteins",
                features=["proteins"],
                epochs=epochs,
                batch_size=batch_size,
                early_stopping=early_stopping,
                output_dir="./outputs/finetuning_ablation_proteins",
            ),
            AblationConfig(
                name="maple",
                description="+ MAPLE embeddings",
                features=["maple"],
                epochs=epochs,
                batch_size=batch_size,
                early_stopping=early_stopping,
                output_dir="./outputs/finetuning_ablation_maple",
            ),
            AblationConfig(
                name="methylgpt",
                description="+ MethylGPT embeddings",
                features=["methylgpt"],
                epochs=epochs,
                batch_size=batch_size,
                early_stopping=early_stopping,
                output_dir="./outputs/finetuning_ablation_methylgpt",
            ),
            AblationConfig(
                name="all",
                description="All features combined",
                features=["clocks", "proteins", "maple", "methylgpt"],
                epochs=epochs,
                batch_size=batch_size,
                early_stopping=early_stopping,
                output_dir="./outputs/finetuning_ablation_all",
            ),
        ]

        for ablation in ablations:
            self.register_ablation(ablation)

    def run_ablation(self, ablation_name: str, dry_run: bool = False) -> bool:
        """Run a single ablation experiment."""
        if ablation_name not in self.ablations:
            logger.error(f"Ablation '{ablation_name}' not registered")
            return False

        config = self.ablations[ablation_name]
        logger.info("=" * 80)
        logger.info(f"ABLATION: {config.name}")
        logger.info(f"Description: {config.description}")
        logger.info(f"Features: {config.features if config.features else '(none - EHR only)'}")
        logger.info("=" * 80)

        if dry_run:
            logger.info("[DRY-RUN] Would run finetuning with:")
            logger.info(f"  Output: {config.output_dir}")
            logger.info(f"  Epochs: {config.epochs}")
            logger.info(f"  Early stopping: {config.early_stopping}")
            self.results[ablation_name] = {"status": "dry-run"}
            return True

        # Run finetuning for this ablation
        from corebehrt.main.finetune_cv import main_finetune

        # Create config file for this ablation
        config_path = self._create_ablation_config(config)

        try:
            main_finetune(str(config_path))
            self.results[ablation_name] = {"status": "completed"}
            logger.info(f"✓ Completed ablation: {ablation_name}")
            return True

        except Exception as e:
            logger.error(f"✗ Failed ablation: {ablation_name}: {e}")
            self.results[ablation_name] = {"status": "failed", "error": str(e)}
            return False

    def run_all_ablations(self, dry_run: bool = False, selected: List[str] = None) -> Dict:
        """Run all registered ablations."""
        if selected:
            ablations_to_run = [a for a in self.ablations.keys() if a in selected]
        else:
            ablations_to_run = list(self.ablations.keys())

        logger.info("\n")
        logger.info("╔" + "="*78 + "╗")
        logger.info("║" + " "*15 + "FEATURE ABLATION EXPERIMENTS" + " "*37 + "║")
        logger.info(f"║ Running {len(ablations_to_run)} ablations" + " "*59 + "║")
        logger.info("╚" + "="*78 + "╝")
        logger.info("\n")

        for ablation_name in ablations_to_run:
            self.run_ablation(ablation_name, dry_run=dry_run)

        return self.results

    def _create_ablation_config(self, config: AblationConfig) -> Path:
        """Create a finetune config for this ablation."""
        base_config = {
            "logging": {
                "level": "INFO",
                "path": "./outputs/logs",
            },
            "paths": {
                "prepared_data": self.prepared_data,
                "pretrain_model": self.pretrain_model,
                "model": config.output_dir,
            },
            "evaluate": False,
            "model": {
                "cls": "default",
                "value_embedding_mode": "concat",
                "include_features": config.features,  # Custom parameter for feature selection
            },
            "trainer_args": {
                "sampler_function": {
                    "_target_": "corebehrt.modules.trainer.utils.Sampling.effective_n_samples"
                },
                "loss_weight_function": {
                    "_target_": "corebehrt.modules.trainer.utils.PositiveWeight.effective_n_samples"
                },
                "batch_size": config.batch_size,
                "val_batch_size": config.batch_size,
                "effective_batch_size": config.batch_size,
                "epochs": config.epochs,
                "info": True,
                "gradient_clip": {"clip_value": 1.0},
                "shuffle": True,
                "checkpoint_frequency": 1,
                "early_stopping": config.early_stopping,
                "stopping_criterion": "roc_auc",
                "n_layers_to_freeze": 1,
                "unfreeze_on_plateau": False,
                "unfreeze_at_epoch": None,
                "plateau_threshold": 0.01,
            },
            "optimizer": {
                "lr": 5e-4,
                "eps": 1e-6,
            },
            "scheduler": {
                "_target_": "transformers.get_linear_schedule_with_warmup",
                "num_warmup_steps": 100,
                "num_training_steps": 1000,
            },
            "metrics": {
                "accuracy": {
                    "_target_": "corebehrt.modules.monitoring.metrics.Accuracy",
                    "threshold": 0.5,
                },
                "roc_auc": {
                    "_target_": "corebehrt.modules.monitoring.metrics.ROC_AUC",
                },
                "pr_auc": {
                    "_target_": "corebehrt.modules.monitoring.metrics.PR_AUC",
                },
                "delong_roc_auc": {
                    "_target_": "corebehrt.modules.monitoring.metrics.Delong_ROC_AUC",
                },
            },
        }

        # Write as YAML
        import yaml

        config_path = Path(f"./corebehrt/configs/ablation_{config.name}.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(base_config, f, default_flow_style=False)

        logger.info(f"Created config: {config_path}")
        return config_path

    def summarize_results(self) -> Dict:
        """Summarize ablation results."""
        logger.info("\n")
        logger.info("╔" + "="*78 + "╗")
        logger.info("║" + " "*20 + "ABLATION RESULTS SUMMARY" + " "*34 + "║")
        logger.info("╚" + "="*78 + "╝")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_ablations": len(self.results),
            "completed": sum(1 for r in self.results.values() if r["status"] == "completed"),
            "failed": sum(1 for r in self.results.values() if r["status"] == "failed"),
            "dry_runs": sum(1 for r in self.results.values() if r["status"] == "dry-run"),
            "details": self.results,
        }

        logger.info(f"\nCompleted: {summary['completed']}/{summary['total_ablations']}")
        logger.info(f"Failed: {summary['failed']}/{summary['total_ablations']}")

        for ablation_name, result in self.results.items():
            status_icon = "✓" if result["status"] == "completed" else "✗"
            logger.info(f"  {status_icon} {ablation_name}: {result['status']}")

        return summary

    def compare_results(self) -> None:
        """Compare ROC-AUC across ablations (stub for future implementation)."""
        logger.info("\n")
        logger.info("Feature Contribution Analysis")
        logger.info("-" * 80)
        logger.info("To compare ROC-AUC improvements:")
        logger.info("1. Collect CV-aggregated metrics from each ablation output")
        logger.info("2. Run DeLong test to assess significance of differences")
        logger.info("3. Use bootstrap to estimate feature importance")
        logger.info("\nImplemented in: corebehrt/analysis/feature_comparison.py")


def main():
    """Run feature ablation experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run feature ablation experiments")
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=None,
        choices=["baseline", "clocks", "proteins", "maple", "methylgpt", "all"],
        help="Which ablations to run (default: all)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only baseline (EHR-only) model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (don't actually run, just show what would run)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )

    args = parser.parse_args()

    runner = FeatureAblationRunner()
    runner.setup_standard_ablations(
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping=max(2, args.epochs // 3),
    )

    # Select which ablations to run
    selected = None
    if args.baseline_only:
        selected = ["baseline"]
    elif args.ablations:
        selected = args.ablations

    # Run experiments
    runner.run_all_ablations(dry_run=args.dry_run, selected=selected)

    # Summarize
    summary = runner.summarize_results()
    runner.compare_results()

    # Save summary
    summary_path = Path("./outputs/ablation_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
