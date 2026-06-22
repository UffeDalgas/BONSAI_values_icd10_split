#!/usr/bin/env python
"""Quick test to see if pretrain runs."""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting pretrain test...")

try:
    from corebehrt.main.pretrain import main_train
    logger.info("✓ Imported main_train")

    logger.info("Running pretrain with dryrun config...")
    result = main_train("corebehrt/configs/pretrain_dryrun.yaml")
    logger.info(f"✓ Pretrain returned: {result}")
    logger.info("✓ Pretrain completed successfully!")

except Exception as e:
    logger.error(f"✗ Pretrain failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
