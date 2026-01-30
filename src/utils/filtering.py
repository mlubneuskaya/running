import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Optional


def filter_main_runner(all_frames_data: List[List[Dict[str, Any]]]) -> List[Optional[Dict[str, Any]]]:
    """
    Analyzes the video to find the Main Runner.
    Strategy: Pick the person with the highest VARIANCE in 'Ankle Spread' (Distance between feet).
    """

    movement_data = defaultdict(list)

    for frame in all_frames_data:
        for p in frame:
            tid = p.get('track_id', -1)
            kpts = p.get('raw_keypoints')
            bbox = p.get('bbox')

            if tid == -1 or kpts is None or bbox is None:
                continue

            _, _, box_h, box_w = bbox

            l_ankle = kpts[15]
            r_ankle = kpts[16]

            dx = l_ankle[0] - r_ankle[0]
            dy = l_ankle[1] - r_ankle[1]
            ankle_distance = np.sqrt(dx ** 2 + dy ** 2)

            movement_data[tid].append(ankle_distance / box_h)

    if not movement_data:
        return [None] * len(all_frames_data)

    best_track_id = -1
    max_variance = -1.0

    for tid, values in movement_data.items():
        if len(values) < 15: continue

        variance = np.std(values)

        print(f"{tid:<5} {len(values):<8} {variance:.5f}")

        if variance > max_variance:
            max_variance = variance
            best_track_id = tid

    if best_track_id == -1:
        best_track_id = max(movement_data, key=lambda k: len(movement_data[k]))

    clean_timeline = [
        next((p for p in frame if p.get('track_id') == best_track_id), None)
        for frame in all_frames_data
    ]

    return clean_timeline