import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def calculate_gct(smoothed_df: pd.DataFrame, fps: float, side: str = "left"):
    timestamps = smoothed_df["timestamp_ms"].values

    heel_x = smoothed_df[f"{side}_heel_x"].values
    toe_x = smoothed_df[f"{side}_big_toe_x"].values
    sacrum = smoothed_df[f"{side}_hip_x"].values

    is_running_right = sacrum[-1] > sacrum[0]
    direction = 1 if is_running_right else -1

    heel_rel_x = (heel_x - sacrum) * direction
    toe_rel_x = (toe_x - sacrum) * direction

    min_stride_frames = int(fps * 0.5)

    landings, _ = find_peaks(heel_rel_x, distance=min_stride_frames, prominence=20)
    mid_swings, _ = find_peaks(
        -toe_rel_x, distance=min_stride_frames, prominence=10, height=0
    )

    gct_records = []

    for land_idx in landings:
        valid_swings = mid_swings[mid_swings > land_idx]

        if len(valid_swings) > 0:
            swing_idx = valid_swings[0]

            window_toe = toe_rel_x[land_idx:swing_idx]

            negative_frames = np.where(window_toe < 0)[0]

            if len(negative_frames) > 0:
                local_liftoff_idx = negative_frames[0]
                true_liftoff_idx = land_idx + local_liftoff_idx

                gct_ms = timestamps[true_liftoff_idx] - timestamps[land_idx]

                if 50 < gct_ms < 600:
                    gct_records.append(
                        {
                            "landing_time": timestamps[land_idx],
                            "liftoff_time": timestamps[true_liftoff_idx],
                            "gct_ms": gct_ms,
                        }
                    )

    return pd.DataFrame(gct_records)


def calculate_flight_times(
    left_gct: pd.DataFrame, right_gct: pd.DataFrame
) -> pd.DataFrame:
    events = []

    for _, row in left_gct.iterrows():
        events.append({"time": row["landing_time"], "type": "landing", "side": "left"})
        events.append({"time": row["liftoff_time"], "type": "liftoff", "side": "left"})

    for _, row in right_gct.iterrows():
        events.append({"time": row["landing_time"], "type": "landing", "side": "right"})
        events.append({"time": row["liftoff_time"], "type": "liftoff", "side": "right"})

    events_df = pd.DataFrame(events).sort_values(by="time").reset_index(drop=True)
    flight_records = []

    for i in range(len(events_df) - 1):
        current_event = events_df.iloc[i]
        next_event = events_df.iloc[i + 1]
        if current_event["type"] == "liftoff" and next_event["type"] == "landing":
            if current_event["side"] != next_event["side"]:
                flight_time_ms = next_event["time"] - current_event["time"]

                flight_records.append(
                    {
                        "takeoff_side": current_event["side"],
                        "landing_side": next_event["side"],
                        "liftoff_time": current_event["time"],
                        "landing_time": next_event["time"],
                        "flight_time_ms": flight_time_ms,
                    }
                )

    return pd.DataFrame(flight_records)
