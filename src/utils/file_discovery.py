import os
import glob
from typing import List


def get_video_files(input_path: str) -> List[str]:
    video_extensions = ["*.mp4", "*.MP4", "*.mov", "*.MOV"]

    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_path, ext)))
        return video_files
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
