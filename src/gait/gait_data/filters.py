import numpy as np
import pandas as pd


class GaitSideFilter:
    def __init__(self, side_bases, anchors, sensitivity=2.0, window=5):
        self.side_bases = side_bases
        self.anchors = anchors
        self.sensitivity = sensitivity
        self.window = window

    def filter_data(self, df):
        cols = df.columns.tolist()
        data = df.values.copy()

        l_anchor_idx = [
            cols.index(f"left_{a}_{c}") for a in self.anchors for c in ["x", "y"]
        ]
        r_anchor_idx = [
            cols.index(f"right_{a}_{c}") for a in self.anchors for c in ["x", "y"]
        ]

        swap_pairs = [
            (cols.index(f"left_{b}{s}"), cols.index(f"right_{b}{s}"))
            for b in self.side_bases
            for s in ["_x", "_y", "_conf"]
        ]

        recent_distances = []

        for i in range(1, len(data)):
            l_curr, r_curr = data[i, l_anchor_idx], data[i, r_anchor_idx]
            l_prev, r_prev = data[i - 1, l_anchor_idx], data[i - 1, r_anchor_idx]

            if np.isnan(l_curr).any() or np.isnan(l_prev).any():
                continue

            d_normal = np.sqrt(np.sum((l_curr - l_prev) ** 2 + (r_curr - r_prev) ** 2))
            d_swapped = np.sqrt(np.sum((l_curr - r_prev) ** 2 + (r_curr - l_prev) ** 2))

            avg_move = np.mean(recent_distances) if recent_distances else d_normal
            dynamic_threshold = avg_move * self.sensitivity

            if d_swapped < d_normal and d_swapped < dynamic_threshold:
                for l_idx, r_idx in swap_pairs:
                    data[i, l_idx], data[i, r_idx] = data[i, r_idx], data[i, l_idx]

                recent_distances.append(d_swapped)
            else:
                recent_distances.append(d_normal)

            if len(recent_distances) > self.window:
                recent_distances.pop(0)

        return pd.DataFrame(data, columns=cols)
