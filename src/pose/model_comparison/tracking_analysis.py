import pandas as pd


def count_detection_gaps(df_coords, keypoints):
    is_row_complete = df_coords.notna().all(axis=1)

    lost_counts = {}

    if is_row_complete.sum() >= 2:
        first_full_idx = is_row_complete.idxmax()
        last_full_idx = is_row_complete[::-1].idxmax()
        total_detected_frames = last_full_idx - first_full_idx

        for kp in keypoints:
            x_col = f"{kp}_x"
            lost_counts[kp] = (
                df_coords[x_col].loc[first_full_idx:last_full_idx].isna().sum()
            )
    else:
        for kp in keypoints:
            lost_counts[kp] = 0
        total_detected_frames = 0

    df = pd.DataFrame.from_dict(
        lost_counts, orient="index", columns=["gap_count"]
    ).reset_index()
    df = df.rename(columns={"index": "keypoint"})
    df["total_detected_frames"] = total_detected_frames
    return df
