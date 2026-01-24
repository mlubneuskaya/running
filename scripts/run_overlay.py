import argparse
import os
import logging

from src.utils.overlay import create_overlay_video
from src.utils.file_discovery import get_video_files
from src.utils.load_config import load_config


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Generate Overlay Videos using YAML Config"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found at {args.config}")
        return

    video_dir = cfg["paths"]["video_input"]
    json_dir = cfg["paths"]["json_input"]
    output_dir = cfg["paths"]["output"]

    video_files = get_video_files(video_dir)

    for video_path in video_files:
        filename = os.path.basename(video_path)
        name_only = os.path.splitext(filename)[0]

        json_path = os.path.join(json_dir, f"{name_only}.json")
        output_path = os.path.join(output_dir, f"{name_only}_overlay.mp4")

        if os.path.exists(json_path):
            try:
                create_overlay_video(video_path, json_path, output_path)
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")
        else:
            logger.info(
                f"Skipping {filename}: No matching JSON found (looked for {json_path})"
            )


if __name__ == "__main__":
    main()
