import os
import glob
from typing import List, Union


def get_json_files(input_path: Union[str, List[str]]) -> List[str]:
    extensions = ["*.json"]

    return get_files(input_path, extensions)


def get_video_files(input_path: Union[str, List[str]]) -> List[str]:
    extensions = ["*.mp4", "*.MP4", "*.mov", "*.MOV"]

    return get_files(input_path, extensions)


def get_files(input_path: Union[str, List[str]], extensions: List[str]) -> List[str]:
    """
    Returns a list of files from:
    - A single file path
    - A directory (recursively searching subdirectories)
    - A list of directories or files
    """
    if isinstance(input_path, str):
        search_paths = [input_path]
    else:
        search_paths = input_path

    found_files = []

    for path in search_paths:
        if os.path.isfile(path):
            found_files.append(path)
        elif os.path.isdir(path):
            for ext in extensions:
                found_files.extend(
                    glob.glob(os.path.join(path, "**", ext), recursive=True)
                )
        else:
            raise FileNotFoundError(f"Input path not found: {path}")

    return sorted(list(set(found_files)))
