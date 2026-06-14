#!/usr/bin/env python3
"""
Run comprehensive ablation: 8 feature subsets × 5 injection methods = 40 models.

This script:
1. Trains all 40 model variants from single pretrain
2. Evaluates each with comprehensive metrics
3. Generates method × feature comparison tables
4. Identifies optimal (feature, method) combinations

Expected runtime:
  - Sequential: ~30 hours
  - Parallel (4 workers): ~7-8 hours
"""

import logging
import sys
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete ablation with injection methods."""

    from corebehrt.ablation.batch_ablation_runner import BatchAblationRunner

    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE ABLATION: FEATURES × INJECTION METHODS")
    logger.info("="*80 + "\n")

    # Initialize runner
    runner = BatchAblationRunner(
        pretrain_checkpoint="./outputs/pretraining_dryrun/checkpoints/best.pt",
        output_dir="./outputs/ablation_results_with_methods",
        n_workers=4  # Train 4 models in parallel
    )

    logger.info("Runner initialized")
    logger.info(f"  Pretrain: {runner.pretrain_checkpoint}")
    logger.info(f"  Output: {runner.output_dir}")
    logger.info(f"  Workers: {runner.n_workers}\n")

    # Feature subsets (9 total)
    feature_subsets = [
        ("ehr_only", "EHR Only", []),
        ("metadata", "Metadata + _preop", ["metadata_preop"]),
        ("grim_age", "GrimAge v2 + Intermediate", ["grim_age_v2"]),
        ("systems_age", "SystemsAge (11 components)", ["systems_age"]),
        ("cpgt_proteins", "CpGPTGrimAge v3 + Proteins", ["cpgt_proteins"]),
        ("maple", "MAPLE (32-dim)", ["maple"]),
        ("cpgt_embed", "CpGPT (64-dim)", ["cpgt_embed"]),
        ("methylgpt", "MethylGPT", ["methylgpt"]),
        ("all_features", "All Features", ["grim_age_v2", "systems_age", "maple", "cpgt_embed", "methylgpt"]),
    ]

    # Injection methods (5 total)
    injection_methods = [
        "concat",
        "film",
        "discrete",
        "comb",
        "comb_binning"
    ]

    # Add all 45 config combinations
    logger.info("="*80)
    logger.info("ADDING ABLATION CONFIGURATIONS")
    logger.info("="*80 + "\n")

    config_count = 0
    for feat_name, feat_display, feat_list in feature_subsets:
        for method in injection_methods:
            config_count += 1
            config_path = f"./corebehrt/configs/ablation_{feat_name}_{method}.yaml"
            config_name = f"{feat_name}_{method}"

            runner.add_ablation_config(
                name=config_name,
                config_path=config_path,
                features=feat_list,
                description=f"{feat_display} with {method} injection"
            )

            # Progress indicator
            if config_count % 5 == 0:
                logger.info(f"  Added {config_count} configs...")

    logger.info(f"  Total: {config_count} configs added\n")

    # ========================================================================
    # PHASE 1: TRAINING
    # ========================================================================

    logger.info("="*80)
    logger.info("PHASE 1: TRAINING ALL 40 MODELS")
    logger.info("="*80 + "\n")

    logger.info("Expected timeline:")
    logger.info(f"  Per model: ~45 minutes (3 epochs, early stopping)")
    logger.info(f"  Sequential: ~{config_count * 45 / 60:.1f} hours")
    logger.info(f"  Parallel (4 workers): ~{config_count * 45 / 60 / 4:.1f} hours\n")

    logger.info("Starting training...\n")

    training_results = runner.train_all_models()

    n_success = sum(training_results.values())
    logger.info(f"\n{'='*80}")
    logger.info(f"Training complete: {n_success}/{len(training_results)} models trained successfully")
    logger.info(f"{'='*80}\n")

    # ========================================================================
    # PHASE 2: EVALUATION
    # ========================================================================

    logger.info("="*80)
    logger.info("PHASE 2: EVALUATING ALL MODELS")
    logger.info("="*80 + "\n")

    logger.info("Computing comprehensive metrics:")
    logger.info("  - Discrimination: ROC-AUC, PR-AUC with bootstrap CI")
    logger.info("  - Calibration: ECE, Hosmer-Lemeshow, calibration slope")
    logger.info("  - Clinical utility: Net benefit, decision thresholds")
    logger.info("  - Risk stratification: Quintile analysis, ICI")
    logger.info("  - Statistical significance: 95% CI, p-values\n")

    eval_results = runner.evaluate_all_models()

    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluation complete: {len(eval_results)} models evaluated")
    logger.info(f"{'='*80}\n")

    # ========================================================================
    # PHASE 3: COMPARISON REPORT
    # ========================================================================

    logger.info("="*80)
    logger.info("PHASE 3: GENERATING COMPARISON REPORTS")
    logger.info("="*80 + "\n")

    comparison_df = runner.generate_comparison_report()

    # ========================================================================
    # PHASE 4: ADVANCED ANALYSIS
    # ========================================================================

    logger.info("\n" + "="*80)
    logger.info("PHASE 4: METHOD × FEATURE INTERACTION ANALYSIS")
    logger.info("="*80 + "\n")

    # Create method × feature comparison matrix
    try:
        import pandas as pd
        import numpy as np

        # Pivot table: methods (rows) × features (columns)
        logger.info("Creating Method × Feature ROC-AUC Matrix:\n")

        # Extract results into matrix form
        method_feature_auc = {}
        for config_name, result in runner.results.items():
            # Parse config_name: "feature_method"
            parts = config_name.rsplit("_", 1)
            if len(parts) == 2:
                feature = parts[0]
                method = parts[1]
                auc = result["metrics"]["roc_auc"]

                if method not in method_feature_auc:
                    method_feature_auc[method] = {}

                method_feature_auc[method][feature] = auc

        # Display matrix
        for method in injection_methods:
            logger.info(f"{method.upper()}:")
            if method in method_feature_auc:
                for feature, auc in sorted(method_feature_auc[method].items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"  {feature:<30s}: {auc:.3f}")
            logger.info("")

        # Find best (feature, method) combination
        logger.info("OPTIMAL COMBINATIONS:")
        logger.info("─" * 80)

        best_overall = None
        best_overall_auc = 0

        for method in injection_methods:
            if method in method_feature_auc:
                best_feat = max(method_feature_auc[method].items(), key=lambda x: x[1])
                logger.info(f"{method:<20s}: {best_feat[0]:<30s} → {best_feat[1]:.3f}")

                if best_feat[1] > best_overall_auc:
                    best_overall_auc = best_feat[1]
                    best_overall = (best_feat[0], method, best_feat[1])

        if best_overall:
            logger.info("\n" + "─"*80)
            logger.info(f"BEST OVERALL: {best_overall[0]} + {best_overall[1]} → {best_overall[2]:.3f}")
            logger.info("─"*80)

    except Exception as e:
        logger.warning(f"Could not generate method analysis: {e}")

    # ========================================================================
    # PHASE 5: SUMMARY
    # ========================================================================

    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80 + "\n")

    runner.summary()

    logger.info("\n" + "="*80)
    logger.info("ABLATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {runner.output_dir}")
    logger.info(f"  - ablation_comparison.csv (all metrics)")
    logger.info(f"  - ablation_results.json (detailed results)")
    logger.info(f"  - comparison_plot.png (visualization)")

    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("""
1. Review ablation_comparison.csv for overall results
2. Analyze method × feature ROC-AUC matrix above
3. Check which methods provide best calibration (lowest ECE)
4. Identify computational efficiency (ROC-AUC per parameter)
5. For deployment: use best (feature, method) combination
6. For publication: report both individual and synergistic effects
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
