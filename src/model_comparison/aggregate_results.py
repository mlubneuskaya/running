import json
import os

import pandas as pd

from src.model_comparison.jitter_analysis import calculate_link_metrics, calculate_jitter_metrics
from src.model_comparison.tracking_analysis import count_detection_gaps
from src.utils.model_comparison.comparison_metrics import expand_lists_to_cols


def aggregate_experiment_results(path_to_fps_dict, detector, links_dict,):
    global_links = []
    global_jitter = []
    global_gaps = []

    for path, fps in path_to_fps_dict.items():
        video_key = os.path.basename(path).split('.')[0]

        with open(path, 'r') as f:
            raw_json = json.load(f)

        keypoints_list = list(set([x for xs in raw_json['connections'] for x in xs]))
        records = [row if row is not None else {} for row in raw_json['pose_data']]
        df_raw = pd.DataFrame(records)[keypoints_list]
        df_coords = expand_lists_to_cols(df_raw)

        df_links = calculate_link_metrics(df_coords, links=links_dict)
        scale = df_links['scale_factor'][0]
        df_jitter = calculate_jitter_metrics(df_coords, keypoints=keypoints_list, fps=fps, scale_factor=scale)

        gap_stats = count_detection_gaps(df_coords, keypoints=keypoints_list)

        for df in [df_links, df_jitter, gap_stats]:
            df['path'] = path
            df['video_id'] = video_key
            df['model'] = detector

        global_links.append(df_links)
        global_jitter.append(df_jitter)
        global_gaps.append(gap_stats)

    links_df = pd.concat(global_links, ignore_index=True)
    jitter_df = pd.concat(global_jitter, ignore_index=True)
    gaps_df = pd.concat(global_gaps, ignore_index=True)

    return links_df, jitter_df, gaps_df