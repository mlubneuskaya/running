import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def get_resultant_accel(df, prefix, dt):
    pos_x = df[f"{prefix}_x"].values
    pos_y = df[f"{prefix}_y"].values

    acc_x = np.gradient(np.gradient(pos_x, dt), dt)
    acc_y = np.gradient(np.gradient(pos_y, dt), dt)
    return np.sqrt(acc_x ** 2 + acc_y ** 2)


def get_vertical_velocity(df, prefix, dt):
    pos_y = df[f"{prefix}_y"].values
    return np.gradient(pos_y, dt)


def detect_initial_contact(res_accel, start_idx, window_size):
    search_win = res_accel[start_idx: start_idx + window_size]
    if len(search_win) == 0:
        return None
    return start_idx + np.argmin(search_win)


def detect_toe_off(toe_vel_y, ic_idx, fps):
    to_start = ic_idx + int(fps * 0.1)
    to_end = ic_idx + int(fps * 0.5)

    window = toe_vel_y[to_start:to_end]
    if len(window) == 0:
        return None

    neg_peaks, _ = find_peaks(-window, prominence=5)
    return (to_start + neg_peaks[0]) if len(neg_peaks) > 0 else None


def calculate_gct(df: pd.DataFrame, fps: float, side: str = "left"):
    dt = 1 / fps
    timestamps = df["timestamp_ms"].values

    heel_accel = get_resultant_accel(df, f"{side}_heel", dt)
    toe_vel_y = get_vertical_velocity(df, f"{side}_big_toe", dt)

    hip_x = df[f"{side}_hip_x"].values
    direction = 1 if hip_x[-1] > hip_x[0] else -1
    heel_rel_x = (df[f"{side}_heel_x"].values - hip_x) * direction

    candidate_landings, _ = find_peaks(heel_rel_x, distance=int(fps * 0.5), prominence=20)

    gct_records = []
    for start_idx in candidate_landings:
        ic_idx = detect_initial_contact(heel_accel, start_idx, int(fps * 0.1))
        if ic_idx is None: continue

        to_idx = detect_toe_off(toe_vel_y, ic_idx, fps)
        if to_idx is None: continue

        gct_records.append({
            "landing_time": timestamps[ic_idx],
            "liftoff_time": timestamps[to_idx],
            "gct_ms": timestamps[to_idx] - timestamps[ic_idx],
            "ic_accel": heel_accel[ic_idx],
            "to_vel_y": toe_vel_y[to_idx]
        })

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
