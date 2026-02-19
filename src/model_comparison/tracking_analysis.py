import pandas as pd

from src.utils.model_comparison.comparison_metrics import expand_lists_to_cols


def count_detection_gaps(df_coords, keypoints):
    is_row_complete = df_coords.notna().all(axis=1)

    lost_counts = {}

    if is_row_complete.sum() >= 2:
        first_full_idx = is_row_complete.idxmax()
        last_full_idx = is_row_complete[::-1].idxmax()
        total_detected_frames = last_full_idx - first_full_idx

        for kp in keypoints:
            x_col = f"{kp}_x"
            lost_counts[kp] = df_coords[x_col].loc[first_full_idx:last_full_idx].isna().sum()
    else:
        for kp in keypoints:
            lost_counts[kp] = 0
        total_detected_frames = 0

    df = pd.DataFrame.from_dict(lost_counts, orient='index', columns=['gap_count']).reset_index()
    df = df.rename(columns={'index': 'keypoint'})
    df['total_detected_frames'] = total_detected_frames
    return df



def aggregate_gaps_across_videos(video_dfs, model_name="Model_A"):
    total_internal_empty = 0

    combined_lost_counts = {}

    for df_video in video_dfs:
        df_coords = expand_lists_to_cols(df_video)

        video_stats = count_detection_gaps(df_coords)

        total_internal_empty += video_stats["internal_empty_rows"]

        for kp, count in video_stats["lost_per_kp"].items():
            combined_lost_counts[kp] = combined_lost_counts.get(kp, 0) + count

    return {
        "model_name": model_name,
        "total_internal_empty_rows": total_internal_empty,
        "total_lost_per_kp": combined_lost_counts,
        "video_count": len(video_dfs),
    }
