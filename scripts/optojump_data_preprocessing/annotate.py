import atexit
import signal
import os
import glob

import cv2
import numpy as np
import pandas as pd


def annotate_video_ui(video_path: str, fps: int) -> list[dict] | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    WIN = "Annotate  —  T=touchdown  O=takeoff  Z=undo  ENTER=save  Q=skip  (side: L/R after each event)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.createTrackbar("Frame", WIN, 0, total_frames - 1, lambda _: None)

    marks: list[dict] = []
    paused = True

    EVENT_COLOR = {
        "touchdown": (0, 255, 80),  # green  (BGR)
        "takeoff": (0, 80, 255),  # red
    }

    def draw_hud(frame: np.ndarray, frame_idx: int, prompt: str = "") -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"Frame {frame_idx}/{total_frames - 1}   "
            f"{frame_idx / fps:.3f}s   "
            f"{'PAUSED' if paused else 'PLAYING'}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (200, 200, 200),
            2,
        )
        cv2.putText(
            overlay,
            "A/D: step   SPACE: play   T: touchdown   O: takeoff   "
            "Z: undo   ENTER: save   Q: skip   (L/R = side after each event)",
            (12, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 160, 160),
            1,
        )

        n_td = sum(1 for m in marks if m["event"] == "touchdown")
        n_to = sum(1 for m in marks if m["event"] == "takeoff")
        last_side = f" ({marks[-1].get('side', '?')})" if marks else ""
        last = f"   last: {marks[-1]['event']}{last_side} f{marks[-1]['frame']}" if marks else ""
        cv2.putText(
            overlay,
            f"marks:  {n_td} touchdown   {n_to} takeoff{last}",
            (12, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 100),
            1,
        )

        if prompt:
            cv2.putText(
                overlay,
                prompt,
                (12, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 220, 255),
                3,
                cv2.LINE_AA,
            )

        cv2.line(overlay, (0, h - 8), (w, h - 8), (60, 60, 60), 6)
        scrub_x = int(frame_idx / max(total_frames - 1, 1) * (w - 1))
        cv2.line(overlay, (0, h - 8), (scrub_x, h - 8), (180, 180, 180), 6)
        for m in marks:
            x = int(m["frame"] / max(total_frames - 1, 1) * (w - 1))
            cv2.line(overlay, (x, h - 22), (x, h - 2), EVENT_COLOR[m["event"]], 2)

        for m in marks:
            if m["frame"] == frame_idx:
                cv2.putText(
                    overlay,
                    m["event"].upper(),
                    (w // 2 - 160, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    EVENT_COLOR[m["event"]],
                    5,
                    cv2.LINE_AA,
                )

        return overlay

    result = None
    current_side: str | None = None  # tracks the side for automatic assignment

    def _ask_side() -> str | None:
        """Freeze the display and wait for L / R keypress. Returns side or None."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, cv2.getTrackbarPos("Frame", WIN))
        _, pf = cap.read()
        cv2.imshow(WIN, draw_hud(frame if pf is None else pf,
                                 cv2.getTrackbarPos("Frame", WIN),
                                 prompt="Side?  L = left   R = right   ESC = cancel"))
        while True:
            sk = cv2.waitKey(0) & 0xFF
            if sk == ord("l"):
                return "left"
            elif sk == ord("r"):
                return "right"
            elif sk == 27:
                return None

    def _flip(side: str) -> str:
        return "right" if side == "left" else "left"

    while True:
        frame_idx = cv2.getTrackbarPos("Frame", WIN)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(WIN, draw_hud(frame, frame_idx))
        key = cv2.waitKey(30 if not paused else 16) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == 13:  # ENTER
            result = marks
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("d"):
            paused = True
            cv2.setTrackbarPos("Frame", WIN, min(frame_idx + 1, total_frames - 1))
        elif key == ord("a"):
            paused = True
            cv2.setTrackbarPos("Frame", WIN, max(frame_idx - 1, 0))
        elif key == ord("t"):
            # Each touchdown starts a new contact bout → flip side.
            # Ask only if this is the very first event.
            if current_side is None:
                side = _ask_side()
                if side is None:
                    continue  # ESC: discard
                current_side = side
            else:
                current_side = _flip(current_side)
            marks.append({"event": "touchdown", "frame": frame_idx,
                           "time_sec": round(frame_idx / fps, 3), "side": current_side})
            print(f"  → touchdown ({current_side})  frame {frame_idx}  ({frame_idx / fps:.3f}s)")
        elif key == ord("o"):
            # Takeoff belongs to the same bout as its preceding touchdown → keep side.
            # Ask only if this is the very first event (leading takeoff).
            if current_side is None:
                side = _ask_side()
                if side is None:
                    continue  # ESC: discard
                current_side = side
            marks.append({"event": "takeoff", "frame": frame_idx,
                           "time_sec": round(frame_idx / fps, 3), "side": current_side})
            print(f"  → takeoff   ({current_side})  frame {frame_idx}  ({frame_idx / fps:.3f}s)")
        elif key == ord("z") and marks:
            removed = marks.pop()
            print(f"  ✗ undid {removed['event']} ({removed.get('side', '?')}) at frame {removed['frame']}")
            # Restore current_side to what it was before this mark was added.
            if not marks:
                current_side = None
            elif removed["event"] == "touchdown":
                # We flipped on this TD, so flip back.
                current_side = _flip(current_side)
            # For a removed takeoff the side doesn't change (it mirrored the TD).
        elif not paused:
            next_frame = min(frame_idx + 1, total_frames - 1)
            cv2.setTrackbarPos("Frame", WIN, next_frame)
            if next_frame == total_frames - 1:
                paused = True

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return result


def marks_to_labels(video_path: str, marks: list[dict]) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    sorted_marks = sorted(marks, key=lambda m: m["frame"])

    if not sorted_marks:
        return pd.DataFrame()

    labels = np.full(total_frames, "flight", dtype=object)
    sides  = np.full(total_frames, None,     dtype=object)

    # Walk marks sequentially so that partial contacts at the edges of the
    # recording are handled correctly:
    #   - A takeoff with no preceding touchdown means the landing happened
    #     before the recording started → contact from frame 0 to that takeoff.
    #   - A touchdown with no following takeoff means the takeoff is cut off
    #     at the end → contact from that touchdown to the last frame.
    i = 0
    while i < len(sorted_marks):
        m = sorted_marks[i]
        if m["event"] == "takeoff":
            # Unmatched leading takeoff: contact from recording start
            start, end = 0, min(total_frames - 1, int(m["frame"]))
            labels[start : end + 1] = "contact"
            sides[start : end + 1]  = m.get("side")
            i += 1
        else:  # touchdown
            if i + 1 < len(sorted_marks) and sorted_marks[i + 1]["event"] == "takeoff":
                nxt = sorted_marks[i + 1]
                start = max(0, int(m["frame"]))
                end   = min(total_frames - 1, int(nxt["frame"]))
                if start >= end:
                    print(f"  ⚠  skipping invalid pair: touchdown f{m['frame']} ≥ takeoff f{nxt['frame']}")
                else:
                    labels[start : end + 1] = "contact"
                    sides[start : end + 1]  = m.get("side")
                i += 2
            else:
                # Unmatched trailing touchdown: contact to recording end
                start = max(0, int(m["frame"]))
                labels[start:] = "contact"
                sides[start:]  = m.get("side")
                i += 1

    # Start the output DataFrame from the first contact frame (which may be
    # frame 0 when a leading takeoff is present).
    contact_indices = np.where(labels != "flight")[0]
    first_frame = int(contact_indices[0]) if len(contact_indices) else int(sorted_marks[0]["frame"])
    return pd.DataFrame(
        {
            "video_file": os.path.basename(video_path),
            "video_path": video_path,
            "frame_number": np.arange(first_frame, total_frames),
            "label": labels[first_frame:],
            "side":  sides[first_frame:],
        }
    )


def extract_metadata(video_path: str, df: pd.DataFrame) -> tuple[str, int, int]:
    study_id = int(os.path.basename(os.path.dirname(video_path)).split("_")[1])
    parts = os.path.splitext(os.path.basename(video_path))[0].split("_")
    test_id = int(parts[-1])
    raw_name = "_".join(parts[:-1])

    match = df[df["full_name"].str.lower().str.replace(" ", "_") == raw_name]
    athlete_name = (
        match.iloc[0]["full_name"]
        if not match.empty
        else raw_name.replace("_", " ").title()
    )
    return athlete_name, study_id, test_id


def process_video_directory(
    input_dir: str,
    clean_csv_path: str,
    output_ml_csv: str,
    fps: int,
    resume: bool = False,
) -> None:
    video_files = sorted(
        glob.glob(os.path.join(input_dir, "**", "*.mov"), recursive=True)
    )
    print(f"Found {len(video_files)} video(s) to process.")

    df = pd.read_csv(clean_csv_path)

    if resume and os.path.exists(output_ml_csv):
        existing = pd.read_csv(output_ml_csv)
        already_done = set(existing["video_file"].unique())
        all_labels = [existing]
        print(f"Resuming — {len(already_done)} video(s) already annotated, skipping.")
    else:
        already_done = set()
        all_labels: list[pd.DataFrame] = []

    pending = [v for v in video_files if os.path.basename(v) not in already_done]
    print(f"{len(pending)} video(s) remaining.\n")

    _saved = False

    def save_progress() -> None:
        nonlocal _saved
        if _saved:
            return
        _saved = True
        if all_labels:
            os.makedirs(os.path.dirname(output_ml_csv) or ".", exist_ok=True)
            pd.concat(all_labels, ignore_index=True).to_csv(output_ml_csv, index=False)
            print(
                f"\nSaved {sum(len(d) for d in all_labels)} labelled frames "
                f"to {output_ml_csv}."
            )
        else:
            print("\nNothing to save.")

    def handle_signal(sig, frame) -> None:
        print(f"\nInterrupted (signal {sig}).")
        save_progress()
        raise SystemExit(0)

    atexit.register(save_progress)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for i, video_path in enumerate(pending, 1):
        athlete_name, study_id, test_id = extract_metadata(video_path, df)
        video_filename = os.path.basename(video_path)
        print(
            f"[{i}/{len(pending)}] {video_filename}  |  {athlete_name}  "
            f"|  study={study_id}  test={test_id}"
        )
        print("  Scrub to each landing → press T, then scrub to liftoff → press O.")
        print("  Press ENTER when done with this video, Q to skip.\n")

        marks = annotate_video_ui(video_path, fps)

        if marks is None:
            print("  Skipped.\n")
            continue
        if not marks:
            print("  No marks placed — skipping.\n")
            continue

        n_td = sum(1 for m in marks if m["event"] == "touchdown")
        n_to = sum(1 for m in marks if m["event"] == "takeoff")
        print(f"  Confirmed: {n_td} touchdown(s), {n_to} takeoff(s).")
        if n_td != n_to:
            print(
                f"  ⚠  Unequal counts — {abs(n_td - n_to)} unmatched event(s) "
                f"will be treated as single-frame contacts."
            )

        ml_df = marks_to_labels(video_path, marks)
        if ml_df.empty:
            print("  No labels generated — skipping.\n")
            continue

        all_labels.append(ml_df)
        print(
            f"  Labelled {len(ml_df)} frames "
            f"(first frame: {ml_df['frame_number'].iloc[0]}).\n"
        )

    print("Batch processing complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual video annotation tool.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos already present in the output CSV.",
    )
    args = parser.parse_args()

    process_video_directory(
        input_dir="data/input/optojump/",
        clean_csv_path="data/input/optojump/optojump_output/parsed/optojump_basic.csv",
        output_ml_csv="data/output/annotations/optojump/ml_training_dataset.csv",
        fps=120,
        resume=args.resume,
    )
