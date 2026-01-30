import os


def get_mirror_path(source_file: str, source_root: str, target_root: str, new_ext: str = None) -> str:
    """
    Generates a target file path that mirrors the subdirectory structure of the source.

    Args:
        source_file: Full path to the input video (e.g., /data/in/day1/run.mp4)
        source_root: The root folder of inputs (e.g., /data/in)
        target_root: The root folder for outputs (e.g., /data/out)
        new_ext: Optional new extension (e.g., '.json'). If None, keeps original.

    Returns:
        Full path to the target file (e.g., /data/out/day1/run.json)
    """
    if os.path.isfile(source_root):
        rel_path = os.path.basename(source_file)
    else:
        rel_path = os.path.relpath(source_file, source_root)

    if new_ext:
        base, _ = os.path.splitext(rel_path)
        rel_path = base + new_ext

    return os.path.join(target_root, rel_path)