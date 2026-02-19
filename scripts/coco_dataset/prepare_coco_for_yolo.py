import json
import os
import random
from tqdm import tqdm


def convert_to_yolo(json_path, images_dir, output_dir, list_file_path):
    print(f"Loading {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    img_map = {img["id"]: img for img in data["images"]}

    img_annotations = {}
    for ann in data["annotations"]:
        if ann["iscrowd"]:
            continue
        img_id = ann["image_id"]
        if img_id not in img_annotations:  # one image can have multiple annotations
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    valid_image_paths = []
    background_image_candidates = []

    print(f"Processing and filtering annotations...")

    for img_id, anns in tqdm(img_annotations.items()):
        img_info = img_map.get(img_id)
        if not img_info:
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]
        file_name = img_info["file_name"]

        img_abs_path = os.path.abspath(os.path.join(images_dir, file_name))
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)

        yolo_lines = []

        has_visible_feet = False

        for ann in anns:
            box = ann["bbox"]
            x_c = (box[0] + box[2] / 2) / img_w
            y_c = (box[1] + box[3] / 2) / img_h
            w = box[2] / img_w
            h = box[3] / img_h

            body_kpts = ann.get("keypoints", [0] * 51)
            foot_kpts = ann.get("foot_kpts", [0] * 18)
            all_kpts = body_kpts + foot_kpts

            # Check feet visibility (indices 17-22 in the 23-point subset)
            # Feet range in flattened list: indices 51 to 69
            # Visibility is at index 2, 5, 8... relative to the start of the slice

            if ann.get("foot_valid", False):
                has_visible_feet = True

            norm_kpts = []
            for i in range(0, len(all_kpts), 3):
                px, py, pv = all_kpts[i], all_kpts[i + 1], all_kpts[i + 2]
                norm_kpts.append(f"{px / img_w:.6f} {py / img_h:.6f} {pv}")

            line = f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} " + " ".join(norm_kpts)
            yolo_lines.append(line)

        if has_visible_feet:  # at least one foot in image
            if yolo_lines:
                with open(txt_path, "w") as f:
                    f.write("\n".join(yolo_lines))
                valid_image_paths.append(img_abs_path)
        else:
            background_image_candidates.append((img_abs_path, txt_path))

    num_positives = len(valid_image_paths)
    num_background = int(num_positives * 0.10)

    if background_image_candidates:
        selected_background = random.sample(
            background_image_candidates,
            min(num_background, len(background_image_candidates)),
        )

        for bg_img_path, bg_txt_path in selected_background:
            with open(bg_txt_path, "w") as f:
                pass
            valid_image_paths.append(bg_img_path)

    print(
        f"Final dataset: {num_positives} positives, {len(valid_image_paths) - num_positives} backgrounds."
    )
    print(f"Saving manifest to {list_file_path}...")

    with open(list_file_path, "w") as f:
        f.write("\n".join(valid_image_paths))


if __name__ == "__main__":
    convert_to_yolo(
        json_path="./data/coco-wholebody/annotations/coco_wholebody_train_v1.0.json",
        images_dir="./data/coco-wholebody/images/train2017",
        output_dir="./data/coco-wholebody/labels/train2017",
        list_file_path="./data/coco-wholebody/train2017.txt",
    )

    convert_to_yolo(
        json_path="./data/coco-wholebody/annotations/coco_wholebody_val_v1.0.json",
        images_dir="./data/coco-wholebody/images/val2017",
        output_dir="./data/coco-wholebody/labels/val2017",
        list_file_path="./data/coco-wholebody/val2017.txt",
    )
