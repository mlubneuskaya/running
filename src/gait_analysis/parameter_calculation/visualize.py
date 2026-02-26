from typing import Tuple

import cv2
import matplotlib.pyplot as plt


def visualize_gait_phase(
    video_path: str,
    landing_frame_idx: int,
    landing_time: float,
    window: int = 1,
    figsize: Tuple[int, int] = (20, 5),
    gait_phase: str = "LANDING",
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    start_frame = max(0, landing_frame_idx - window)
    end_frame = landing_frame_idx + window

    frames = []
    frame_indices = []

    for i in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_indices.append(i)

    cap.release()

    plt.figure(figsize=figsize)

    for i, frame in enumerate(frames):
        ax = plt.subplot(1, len(frames), i + 1)
        plt.imshow(frame)

        current_idx = frame_indices[i]
        if current_idx == landing_frame_idx:
            title_color = "red"
            title_text = (
                f"Frame {current_idx} - {landing_time: .2f} ms\n(DETECTED {gait_phase})"
            )
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
        else:
            title_color = "black"
            title_text = f"Frame {current_idx}"

        plt.title(title_text, color=title_color, fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
