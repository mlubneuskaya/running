import os
import shutil

from ultralytics import YOLO


def download_model(model_name, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    final_path = os.path.join(target_dir, model_name)

    if os.path.exists(final_path):
        return final_path

    YOLO(model_name)
    current_file_path = os.path.join(os.getcwd(), model_name)

    shutil.move(current_file_path, final_path)
    return final_path
