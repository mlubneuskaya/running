import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def get_resultant_accel(df, prefix, dt):
    hip_x = df[f"{prefix.split('_')[0]}_hip_x"].values
    pos_x = hip_x - df[f"{prefix}_x"].values
    pos_y = df[f"{prefix}_y"].values

    acc_x = np.gradient(np.gradient(pos_x, dt), dt)
    acc_y = np.gradient(np.gradient(pos_y, dt), dt)
    return np.sqrt(acc_x**2 + acc_y**2)


def get_vertical_velocity(df, prefix, dt):
    pos_y = df[f"{prefix}_y"].values
    return np.gradient(pos_y, dt)


# def detect_initial_contact(res_accel, start_idx, window_size):
#     search_win = res_accel[start_idx: start_idx + window_size]
#     if len(search_win) == 0:
#         return None
#     return start_idx + np.argmin(search_win)


def detect_initial_contact_base(toe_vel_y, heel_rel_x, ic_idx, fps):
    to_start = max(0, ic_idx - int(fps * 0.1))
    to_end = min(len(toe_vel_y), ic_idx + int(fps * 0.5))

    acceleration = np.gradient(toe_vel_y, 1 / fps)
    vel_window = toe_vel_y[to_start:to_end]
    heel_x_window = heel_rel_x[to_start:to_end]

    if len(vel_window) < 2:
        return None

    accel_window = acceleration[to_start:to_end]

    all_peaks, _ = find_peaks(accel_window)#, prominence=5)
    # 2. Filter peaks to only keep those where the heel relative position is positive
    valid_peaks = [p for p in all_peaks if heel_x_window[p] > -50]
    #valid_peaks = all_peaks
    peaks = np.array(valid_peaks)

    if len(peaks) == 0:
        return None

    global_peaks = to_start + peaks

    # This logic is actually solid! It correctly finds the closest peak using np.abs
    closest_peak_idx = global_peaks[np.argmin(np.abs(global_peaks - ic_idx))]

    return closest_peak_idx


def detect_toe_off_base(toe_vel_y, heel_rel_x, ic_idx, fps):
    to_start = ic_idx + int(fps * 0.1)
    to_end = ic_idx + int(fps * 0.8)

    vel_window = toe_vel_y[to_start:to_end]
    heel_rel_window = heel_rel_x[to_start:to_end]

    if len(vel_window) < 2:
        return None
    accel_window = np.gradient(vel_window, 1 / fps)
    masked_accel = np.where(heel_rel_window < 0, accel_window, np.inf)

    neg_peaks, _ = find_peaks(-masked_accel, prominence=5)

    if len(neg_peaks) == 0:
        return None

    global_to_peaks = to_start + neg_peaks
    closest_to_idx = global_to_peaks[np.argmin(np.abs(global_to_peaks - ic_idx))]

    return closest_to_idx


def detect_toe_off(accel_norm, heel_rel_x, ic_idx, fps):
    to_start = ic_idx + int(fps * 0.1)
    to_end = ic_idx + int(fps * 0.8)

    # Safety check to prevent index out of bounds
    if to_start >= len(accel_norm):
        return None
    to_end = min(len(accel_norm), to_end)

    accel_window = accel_norm[to_start:to_end]
    heel_rel_window = heel_rel_x[to_start:to_end]

    if len(accel_window) < 2:
        return None

    # 1. Find valleys by looking at negative normalized acceleration (10% threshold)
    all_valleys, _ = find_peaks(-accel_window, prominence=0.10)

    # 2. Filter: Heel must be behind the hip (< 0)
    #valid_valleys = [p for p in all_valleys if heel_rel_window[p] < 0]
    valid_valleys = all_valleys

    if len(valid_valleys) == 0:
        return None

    global_to_peaks = to_start + np.array(valid_valleys)

    # 3. Grab the first chronological valley that met our conditions
    closest_to_idx = global_to_peaks[0]

    return closest_to_idx


def detect_running_direction(df, side="left"):
    toe_x = df[f"{side}_big_toe_x"]
    heel_x = df[f"{side}_heel_x"]
    relative_dist = toe_x - heel_x

    average_orientation = np.nanmedian(relative_dist)

    if average_orientation > 0:
        return 1
    else:
        return -1


def detect_initial_contact(heel_y, start_idx, fps):
    to_start = max(0, start_idx - int(fps * 0.2))
    to_end = min(len(heel_y), start_idx + int(fps * 0.5))

    window = heel_y[to_start:to_end]

    if len(window) == 0:
        return None

    # Find the maximum Y value (lowest physical point).
    # (If your Y-axis is NOT inverted, simply change this to np.argmin)
    local_max_idx = np.argmax(window)

    return to_start + local_max_idx


def calculate_gct(df: pd.DataFrame, fps: float, side: str = "left"):
    dt = 1 / fps
    timestamps = df["timestamp_ms"].values

    # --- POSITIONAL DATA (For Landing) ---
    heel_y = df[f"{side}_heel_y"].values

    # --- ACCELERATION DATA (For Liftoff) ---
    toe_vel_y = get_vertical_velocity(df, f"{side}_big_toe", dt)
    toe_accel_y = np.gradient(toe_vel_y, dt)

    # Global normalization for toe acceleration [0, 1]
    toe_range = np.ptp(toe_accel_y)
    if toe_range > 1e-5:
        toe_accel_norm = (toe_accel_y - np.min(toe_accel_y)) / toe_range
    else:
        toe_accel_norm = toe_accel_y

    # Output records variables
    heel_accel = get_resultant_accel(df, f"{side}_big_toe", dt)

    # Relative position for segmenting strides
    hip_x = df[f"{side}_hip_x"].values
    direction = detect_running_direction(df, side=side)
    heel_rel_x = (df[f"{side}_heel_x"].values - hip_x) * direction

    candidate_landings, _ = find_peaks(
        heel_rel_x, distance=int(fps * 0.5), prominence=20
    )

    gct_records = []
    for start_idx in candidate_landings:
        # 1. Find landing using RAW POSITION (Lowest point)
        ic_idx = detect_initial_contact(heel_y, start_idx, fps)
        if ic_idx is None:
            continue

        # 2. Find liftoff using NORMALIZED ACCELERATION (Valleys)
        to_idx = detect_toe_off(toe_accel_norm, heel_rel_x, ic_idx, fps)
        if to_idx is None:
            continue

        gct_records.append({
            "landing_time": timestamps[ic_idx],
            "liftoff_time": timestamps[to_idx],
            "gct_ms": timestamps[to_idx] - timestamps[ic_idx],
            "ic_accel": heel_accel[ic_idx],
            "to_vel_y": toe_vel_y[to_idx],
        })

    return pd.DataFrame(gct_records)

def calculate_gct_base(df: pd.DataFrame, fps: float, side: str = "left"):
    dt = 1 / fps
    timestamps = df["timestamp_ms"].values

    heel_accel = get_resultant_accel(df, f"{side}_big_toe", dt)
    toe_vel_y = get_vertical_velocity(df, f"{side}_big_toe", dt)
    heel_vel_y = get_vertical_velocity(df, f"{side}_heel", dt)

    hip_x = df[f"{side}_hip_x"].values
    direction = detect_running_direction(
        df, side=side
    )
    heel_rel_x = (df[f"{side}_heel_x"].values - hip_x) * direction
    toe_rel_x = (df[f"{side}_big_toe_x"].values - hip_x) * direction

    candidate_landings, _ = find_peaks(heel_rel_x, distance=int(fps * 0.5))#, prominence=20)

    gct_records = []
    for start_idx in candidate_landings:
        ic_idx = detect_initial_contact(
            heel_vel_y, toe_rel_x, start_idx, fps
        )  # detect_initial_contact(heel_accel, start_idx, int(fps * 0.1))
        if ic_idx is None:
            continue

        to_idx = detect_toe_off(toe_vel_y, toe_rel_x, ic_idx, fps)
        if to_idx is None:
            continue

        gct_records.append(
            {
                "landing_time": timestamps[ic_idx],
                "liftoff_time": timestamps[to_idx],
                "gct_ms": timestamps[to_idx] - timestamps[ic_idx],
                "ic_accel": heel_accel[ic_idx],
                "to_vel_y": toe_vel_y[to_idx],
                "heel_rel_x": heel_rel_x[ic_idx],
            }
        )

    return pd.DataFrame(gct_records)


def calculate_flight_times(left_gct: pd.DataFrame, right_gct: pd.DataFrame) -> pd.DataFrame:
    flight_records = []

    # 1. Calculate Left-to-Right flight times
    for _, l_row in left_gct.iterrows():
        l_land = l_row["landing_time"]
        l_lift = l_row["liftoff_time"]

        # Find the first right step that lands AFTER this left step lands
        future_r_steps = right_gct[right_gct["landing_time"] > l_land]

        if not future_r_steps.empty:
            next_r_land = future_r_steps.iloc[0]["landing_time"]

            # Flight time = Right Landing - Left Liftoff
            # (If negative, it means they are walking/jogging and both feet were on the ground)
            flight_time_ms = next_r_land - l_lift

            # Optional: Filter out massive flight times caused by a completely missed step
            if flight_time_ms < 500:
                flight_records.append({
                    "takeoff_side": "left",
                    "landing_side": "right",
                    "liftoff_time": l_lift,
                    "landing_time": next_r_land,
                    "flight_time_ms": flight_time_ms,
                })

    # 2. Calculate Right-to-Left flight times
    for _, r_row in right_gct.iterrows():
        r_land = r_row["landing_time"]
        r_lift = r_row["liftoff_time"]

        # Find the first left step that lands AFTER this right step lands
        future_l_steps = left_gct[left_gct["landing_time"] > r_land]

        if not future_l_steps.empty:
            next_l_land = future_l_steps.iloc[0]["landing_time"]
            flight_time_ms = next_l_land - r_lift

            if flight_time_ms < 500:
                flight_records.append({
                    "takeoff_side": "right",
                    "landing_side": "left",
                    "liftoff_time": r_lift,
                    "landing_time": next_l_land,
                    "flight_time_ms": flight_time_ms,
                })

    if not flight_records:
        return pd.DataFrame()

    # Combine and sort them chronologically for the final output
    events_df = pd.DataFrame(flight_records).sort_values(by="liftoff_time").reset_index(drop=True)
    return events_df
