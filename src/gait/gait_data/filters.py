import numpy as np
import pandas as pd


# class GaitInertiaFilter:
#     def __init__(self, side_bases, anchors, window=21, swap_margin=25.0):
#         # Increased window bridges denser clusters of tracking errors without flattening the arc
#         self.side_bases = side_bases
#         self.anchors = anchors
#         self.window = window
#         self.swap_margin = swap_margin  # Pixel inertia required to change state
#
#     def filter_data(self, df):
#         cols = df.columns.tolist()
#         data = df.values.copy()

        # l_anchor_idx = [cols.index(f"left_{a}_{c}") for a in self.anchors for c in ["x", "y"]]
        # r_anchor_idx = [cols.index(f"right_{a}_{c}") for a in self.anchors for c in ["x", "y"]]
        #
        # swap_pairs = [
        #     (cols.index(f"left_{b}{s}"), cols.index(f"right_{b}{s}"))
        #     for b in self.side_bases for s in ["_x", "_y", "_conf"]
        # ]
        #
        # # 1. Macro Guide Rails
        # anchors_df = df.iloc[:, l_anchor_idx + r_anchor_idx].copy()
        # anchors_df = anchors_df.interpolate(method='linear', limit_direction='both')
        # guide_rails = anchors_df.rolling(
        #     window=self.window, center=True, min_periods=1
        # ).median().values
        #
        # l_refs = guide_rails[:, :len(l_anchor_idx)]
        # r_refs = guide_rails[:, len(l_anchor_idx):]
        #
        # # 2. State-Aware Evaluation (Hysteresis)
        # is_swapped = False
        #
        # for i in range(len(data)):
        #     l_raw = data[i, l_anchor_idx]
        #     r_raw = data[i, r_anchor_idx]
        #     l_ref = l_refs[i]
        #     r_ref = r_refs[i]
        #
        #     if np.isnan(l_raw).any() or np.isnan(r_raw).any() or np.isnan(l_ref).any() or np.isnan(r_ref).any():
        #         continue
        #
        #     # Calculate L1 distances
        #     d_normal = np.sum(np.abs(l_raw - l_ref)) + np.sum(np.abs(r_raw - r_ref))
        #     d_swapped = np.sum(np.abs(l_raw - r_ref)) + np.sum(np.abs(r_raw - l_ref))
        #
        #     # Hysteresis Logic: Require a significant margin to change the current state
        #     if not is_swapped:
        #         # We are normal. Need strong evidence to swap.
        #         if d_swapped < (d_normal - self.swap_margin):
        #             is_swapped = True
        #     else:
        #         # We are swapped. Need strong evidence to go back to normal.
        #         if d_normal < (d_swapped - self.swap_margin):
        #             is_swapped = False
        #
        #     # Apply swap if our current state dictates it
        #     if is_swapped:
        #         for l_idx, r_idx in swap_pairs:
        #             data[i, l_idx], data[i, r_idx] = data[i, r_idx], data[i, l_idx]
        #
        # return pd.DataFrame(data, columns=cols)


# import numpy as np
# import pandas as pd
#
#
class GaitSideFilter:
    def __init__(self, side_bases, anchors, sensitivity=2.0, window=5, lookback=1):
        self.side_bases = side_bases
        self.anchors = anchors
        self.sensitivity = sensitivity
        self.window = window
        self.lookback = lookback

    def filter_data(self, df):
        cols = df.columns.tolist()
        data = df.values.copy()

        l_anchor_idx = [cols.index(f"left_{a}_{c}") for a in self.anchors for c in ["x", "y"]]
        r_anchor_idx = [cols.index(f"right_{a}_{c}") for a in self.anchors for c in ["x", "y"]]

        swap_pairs = [
            (cols.index(f"left_{b}{s}"), cols.index(f"right_{b}{s}"))
            for b in self.side_bases for s in ["_x", "_y", "_conf"]
        ]

        recent_distances = []

        for i in range(1, len(data)):
            l_curr, r_curr = data[i, l_anchor_idx], data[i, r_anchor_idx]

            start = max(0, i - self.lookback)
            past = data[start:i]

            valid = (~np.isnan(past[:, l_anchor_idx]).any(axis=1) &
                     ~np.isnan(past[:, r_anchor_idx]).any(axis=1))

            if np.isnan(l_curr).any() or not valid.any():
                continue

            # IMPROVEMENT 1: Use median instead of mean to ignore wild outliers in the lookback
            l_prev = np.median(past[valid][:, l_anchor_idx], axis=0)
            r_prev = np.median(past[valid][:, r_anchor_idx], axis=0)

            # IMPROVEMENT 2: Use L1 Norm (absolute sum) instead of L2 (squared)
            # to stop a single rogue keypoint from blowing up the distance
            d_normal = np.sum(np.abs(l_curr - l_prev)) + np.sum(np.abs(r_curr - r_prev))
            d_swapped = np.sum(np.abs(l_curr - r_prev)) + np.sum(np.abs(r_curr - l_prev))
#
            # IMPROVEMENT 1b: Use median for the rolling history as well
            avg_move = np.median(recent_distances) if recent_distances else d_normal
            dynamic_threshold = avg_move * self.sensitivity

            if d_swapped < d_normal and d_swapped < dynamic_threshold:
                # Perform the swap
                for l_idx, r_idx in swap_pairs:
                    data[i, l_idx], data[i, r_idx] = data[i, r_idx], data[i, l_idx]
                d_chosen = d_swapped
            else:
                d_chosen = d_normal

            # IMPROVEMENT 3: Prevent "history poisoning".
            # If a massive tracking glitch occurs, don't let it inflate the threshold for the next N frames.
            # We cap the recorded distance to a reasonable maximum (e.g., the dynamic threshold itself).
            capped_distance = min(d_chosen, max(avg_move * 1.5, 1.0))  # 1.0 prevents trapping at 0
            recent_distances.append(capped_distance)

            if len(recent_distances) > self.window:
                recent_distances.pop(0)

        return pd.DataFrame(data, columns=cols)