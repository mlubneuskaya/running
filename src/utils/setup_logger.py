import datetime
import json
import logging
import os
from typing import Dict, Any, Tuple


def setup_run_logging(logger: logging.Logger, output_root: str, config: Dict[str, Any]) -> Tuple[str, str]:

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config['model']['name'].replace(".task", "").replace(".pt", "").split("/")[-1]
    run_id = f"run_{timestamp}_{model_name}"

    run_dir = os.path.join(output_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    log_file_path = os.path.join(run_dir, "execution.log")

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )

    separator = "=" * 50
    meta_info = {
        "Run ID": run_id,
        "Timestamp": timestamp,
        "Primary Model": {
            "Type": config['model']['type'],
            "Name": config['model']['name'],
            "Device": config['model'].get('device', 'cpu')
        }
    }

    if "roi_detector" in config:
        meta_info["ROI Detector"] = config["roi_detector"]
    else:
        meta_info["ROI Detector"] = "None (Full Frame)"

    logger.info(separator)
    logger.info("RUN CONFIGURATION START")
    logger.info(json.dumps(meta_info, indent=4))
    logger.info("RUN CONFIGURATION END")
    logger.info(separator)

    return run_dir, log_file_path