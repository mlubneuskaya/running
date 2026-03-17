from matplotlib import pyplot as plt
import numpy as np


def plot_kinematic_chain(df, annotations_df, side, part, axis, fps, show_annotations=True):
    column_name = f"{side}_{part}_{axis}"
    if column_name not in df.columns:
        return

    dt = 1/fps
    pos = df[column_name].values * -1
    vel = np.gradient(pos, dt)
    acc = np.gradient(vel, dt)

    def normalize(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    time_x = df["timestamp_ms"]

    plt.figure(figsize=(15, 7))

    plt.plot(time_x, normalize(pos), label="Position", color="#1f77b4", lw=2.5)
    plt.plot(time_x, normalize(vel), label="Velocity", color="#ff7f0e", ls="--", lw=1.5)
    plt.plot(time_x, normalize(acc), label="Acceleration", color="#2ca02c", ls=":", lw=1.5)

    if show_annotations and annotations_df is not None:
        current_side_events = annotations_df[annotations_df['side'] == side.lower()]

        for _, row in current_side_events.iterrows():
            frame_idx = int(row['frame'])
            if frame_idx < df.index.max():
                timestamp = df.loc[frame_idx]["timestamp_ms"]
                e_type = row['event_type']

                if "IC" in e_type:
                    color, marker = "red", "v"
                elif "TO" in e_type:
                    color, marker = "purple", "^"
                else:
                    color, marker = "black", "o"

                plt.axvline(x=timestamp, color=color, alpha=0.9, linestyle="-", lw=1)
                plt.scatter(timestamp, 1.05, color=color, marker=marker, s=120,
                            label=f"{side.upper()} {e_type}", zorder=5, clip_on=False)

                plt.text(timestamp, 1.08, e_type, color=color, fontweight='bold',
                         ha='center', fontsize=9)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    plt.title(f"{side.capitalize()} {part.capitalize()} Kinematic Profile", fontsize=14, pad=35)
    plt.ylabel("Normalized Magnitude", fontsize=11)
    plt.xlabel("Time (ms)", fontsize=11)
    plt.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.show()