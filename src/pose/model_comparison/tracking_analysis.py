import pandas as pd


def count_detection_gaps(df_coords, keypoints):
    total_detected_frames = len(df_coords)

    lost_counts = {}
    for kp in keypoints:
        x_col = f"{kp}_x"
        if x_col in df_coords.columns:
            lost_counts[kp] = int(df_coords[x_col].isna().sum())
        else:
            lost_counts[kp] = total_detected_frames

    df = pd.DataFrame.from_dict(
        lost_counts, orient="index", columns=["gap_count"]
    ).reset_index()
    df = df.rename(columns={"index": "keypoint"})
    df["total_detected_frames"] = total_detected_frames
    return df
