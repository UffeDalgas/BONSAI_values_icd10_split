#!/usr/bin/env python3
"""
Quick script to run STEP 6 (finetune) directly, bypassing argument parsing issues.

This ensures the pretrain checkpoint loads correctly and finetune completes.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_finetune_step6")


def main():
    """Run finetune step directly."""

    logger.info("\n" + "="*80)
    logger.info("STEP 6: FINETUNE WITH BIOLOGICAL FEATURES")
    logger.info("="*80 + "\n")

    # Import and run finetune directly
    from corebehrt.main.finetune_cv import main_finetune

    finetune_config = "./corebehrt/configs/finetune_dryrun_values.yaml"

    # Verify config exists
    config_path = Path(finetune_config)
    if not config_path.exists():
        logger.error(f"Config file not found: {finetune_config}")
        return 1

    logger.info(f"Using config: {finetune_config}")
    logger.info("Starting finetune training with biological features...")
    logger.info("")

    try:
        main_finetune(finetune_config)
        logger.info("")
        logger.info("="*80)
        logger.info("✓ FINETUNE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info("")
        logger.info(f"Model checkpoint saved to: ./outputs/finetuning_dryrun_values")
        logger.info(f"Expected outputs:")
        logger.info(f"  - Trained model checkpoint")
        logger.info(f"  - Training metrics")
        logger.info(f"  - Evaluation results")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"Finetune failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
