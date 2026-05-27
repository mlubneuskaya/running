"""
Video annotation tool — per-frame contact/flight labels with extended visibility range.

Supported datasets:
  optojump  —  data/input/optojump/study_N/name_surname_M.mov
  tempos    —  data/input/tempos/{person_id}/{study_id}[.{video_id}].MP4

Output schema (one row per frame, from first_visible to last_visible):
    person_id | study_id | video_id | Name | Surname
    | video_path | frame_number | label | side

label ∈ {"contact", "flight"}
side  ∈ {"left", "right", None}

Keys in the annotation UI:
    F           mark current frame as first-visible  (green ▸ on scrubber)
    B           mark current frame as last-visible   (red   ◂ on scrubber)
    T           mark touchdown  — prompts for L/R side on first event;
                                  auto-flips side on subsequent touchdowns
    O           mark takeoff    — inherits side from last touchdown
    L / R       side response for the L/R prompt
    Z           undo the most-recently placed mark (any type)
    ENTER       accept and save
                  • requires F and B to be set, and F < B
                  • touchdown/takeoff marks are optional (no marks → all flight)
    Q / S / ESC skip this video (leaves existing rows in output untouched)
    A / D       step one frame backward / forward (pauses)
    SPACE       toggle play / pause

Usage:
    python annotate.py --dataset optojump
    python annotate.py --dataset tempos
    python annotate.py --dataset optojump --input-dir data/input/optojump/ --fps 120
"""

from __future__ import annotations

import atexit
import glob
import os
import signal
import argparse

import cv2
import numpy as np
import pandas as pd


# ── schema ────────────────────────────────────────────────────────────────────

OUTPUT_COLUMNS = [
    "person_id", "study_id", "video_id",
    "Name", "Surname",
    "video_path", "frame_number", "label", "side",
]

# ── annotation UI ─────────────────────────────────────────────────────────────

def annotate_video_ui(
    video_path: str,
    fps: int,
    ex_rows: pd.DataFrame | None = None,
    existing_first: int | None = None,
    existing_last:  int | None = None,
    existing_contacts: list[tuple[int, int]] | None = None,
) -> dict | None:
    """
    Combined UI: mark first/last visible frames AND touchdown/takeoff events.

    Parameters
    ----------
    ex_rows
        Annotations from the output CSV for this video
    existing_first, existing_last
        Frame numbers from the output CSV for this video (shown in cyan as
        reference; pre-fill the F/B marks so the user can accept with ENTER
        or adjust them).
    existing_contacts
        List of (start_frame, end_frame) contact ranges from existing data,
        drawn as a cyan band on the scrubber for reference.

    Returns
    -------
    dict with keys ``first`` (int), ``last`` (int), ``marks`` (list[dict])
    or ``None`` if the video was skipped.
    """
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video: {video_path}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert total_frames > 0, f"Video reports 0 frames: {video_path}"

    WIN = (
        "Annotate  —  "
        "F=first-vis  B=last-vis  "
        "T=touchdown  O=takeoff  (L/R=side)  "
        "Z=undo  ENTER=save  Q=skip"
    )
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.createTrackbar("Frame", WIN, 0, total_frames - 1, lambda _: None)

    # visibility marks – pre-filled from existing data
    first_frame: int | None = existing_first
    last_frame:  int | None = existing_last

    # Reconstruct touchdown/takeoff marks from existing per-frame rows so they
    # appear on the scrubber and are returned as-is when the user skips (Q/ESC).
    # action_stack is built alongside marks so every reconstructed event can be
    # undone with Z — each entry stores the prev_side exactly as the live TD/TO
    # handlers do.
    marks: list[dict] = []
    action_stack: list[dict] = []   # for undo; each entry records what changed
    _cur_side: str | None = None    # running current_side during reconstruction
    if ex_rows is not None and not ex_rows.empty and "label" in ex_rows.columns:
        c_rows = (ex_rows[ex_rows["label"] == "contact"]
                  .sort_values("frame_number"))
        if not c_rows.empty:
            frames = c_rows["frame_number"].values
            gaps   = np.where(np.diff(frames) > 1)[0]
            starts = [int(frames[0])]      + [int(frames[g + 1]) for g in gaps]
            ends   = [int(frames[g]) for g in gaps] + [int(frames[-1])]
            for _s, _e in zip(starts, ends):
                _side_rows = ex_rows[ex_rows["frame_number"] == _s]
                _raw_side  = _side_rows["side"].iloc[0] if not _side_rows.empty else None
                _side      = str(_raw_side) if pd.notna(_raw_side) else None
                # touchdown — record prev_side before the side changes
                action_stack.append({"type": "event", "prev_side": _cur_side})
                _cur_side = _side
                marks.append({"event": "touchdown", "frame": _s,
                              "time_sec": round(_s / fps, 3), "side": _side})
                # takeoff — current_side stays the same
                action_stack.append({"type": "event", "prev_side": _cur_side})
                marks.append({"event": "takeoff",   "frame": _e,
                              "time_sec": round(_e / fps, 3), "side": _side})

    current_side: str | None = _cur_side
    paused = True
    last_img: np.ndarray | None = None  # most recently decoded frame

    EVENT_COLOR = {
        "touchdown": (0, 255, 80),   # green
        "takeoff":   (0, 80, 255),   # red
    }

    def draw_hud(img: np.ndarray, fidx: int, prompt: str = "") -> np.ndarray:
        h, w = img.shape[:2]
        ov = img.copy()

        # ── top bar ──────────────────────────────────────────────────────────
        cv2.rectangle(ov, (0, 0), (w, 160), (0, 0, 0), -1)

        cv2.putText(ov,
            f"Frame {fidx}/{total_frames - 1} "
            f"{'PAUSED' if paused else 'PLAYING'}",
            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)

        cv2.putText(ov,
            "A/D:step  SPACE:play  "
            "F:first-vis  B:last-vis  "
            "T:touchdown  O:takeoff  (L/R:side)  "
            "Z:undo  ENTER:save  Q:skip",
            (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        # reference from existing file (cyan)
        ref_parts = []
        if existing_first is not None:
            ref_parts.append(f"first={existing_first}")
        if existing_last is not None:
            ref_parts.append(f"last={existing_last}")
        if existing_contacts:
            ref_parts.append(f"{len(existing_contacts)} contact range(s)")
        if ref_parts:
            cv2.putText(ov, "[REF from file]  " + "   ".join(ref_parts),
                (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # current visibility marks
        fstr = str(first_frame) if first_frame is not None else "—"
        lstr = str(last_frame)  if last_frame  is not None else "—"
        cv2.putText(ov, f"first_visible={fstr}    last_visible={lstr}",
            (12, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

        # touchdown / takeoff summary
        n_td = sum(1 for m in marks if m["event"] == "touchdown")
        n_to = sum(1 for m in marks if m["event"] == "takeoff")
        side_hint = f"  current_side={current_side}" if current_side else ""
        last_mark = (f"  last: {marks[-1]['event']} f{marks[-1]['frame']}"
                     if marks else "")
        cv2.putText(ov,
            f"TD={n_td}  TO={n_to}{side_hint}{last_mark}",
            (12, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 100), 1)

        # big prompt (e.g. "Side? L / R")
        if prompt:
            cv2.putText(ov, prompt,
                (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 220, 255), 3, cv2.LINE_AA)

        # ── scrubber ─────────────────────────────────────────────────────────
        sb_y = h - 8
        cv2.line(ov, (0, sb_y), (w, sb_y), (60, 60, 60), 6)
        scrub_x = int(fidx / max(total_frames - 1, 1) * (w - 1))
        cv2.line(ov, (0, sb_y), (scrub_x, sb_y), (180, 180, 180), 6)

        # existing contact ranges – cyan band
        if existing_contacts:
            for cs, ce in existing_contacts:
                cv2.line(ov,
                    (int(cs / max(total_frames - 1, 1) * (w - 1)), h - 20),
                    (int(ce / max(total_frames - 1, 1) * (w - 1)), h - 20),
                    (0, 200, 255), 5)

        # existing first/last – cyan thin verticals
        for ef in (existing_first, existing_last):
            if ef is not None:
                ex = int(ef / max(total_frames - 1, 1) * (w - 1))
                cv2.line(ov, (ex, h - 30), (ex, h - 2), (0, 200, 255), 2)

        # current first_visible – green
        if first_frame is not None:
            fx = int(first_frame / max(total_frames - 1, 1) * (w - 1))
            cv2.line(ov, (fx, h - 30), (fx, h - 2), (0, 255, 80), 3)
            cv2.putText(ov, "F", (max(0, fx - 5), h - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1)

        # current last_visible – red
        if last_frame is not None:
            lx = int(last_frame / max(total_frames - 1, 1) * (w - 1))
            cv2.line(ov, (lx, h - 30), (lx, h - 2), (0, 80, 255), 3)
            cv2.putText(ov, "B", (max(0, lx - 5), h - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)

        # touchdown / takeoff tick marks
        for m in marks:
            mx = int(m["frame"] / max(total_frames - 1, 1) * (w - 1))
            cv2.line(ov, (mx, h - 24), (mx, h - 2), EVENT_COLOR[m["event"]], 2)

        # event label overlay at current position
        for m in marks:
            if m["frame"] == fidx:
                cv2.putText(ov, m["event"].upper(),
                    (w // 2 - 160, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    EVENT_COLOR[m["event"]], 5, cv2.LINE_AA)

        return ov

    def _ask_side() -> str | None:
        """Freeze display and wait for L / R / ESC."""
        f_idx = cv2.getTrackbarPos("Frame", WIN)
        img   = last_img if last_img is not None else np.zeros((720, 1280, 3), np.uint8)
        cv2.imshow(WIN, draw_hud(img, f_idx,
                                 prompt="Side?  L=left   R=right   ESC=cancel"))
        while True:
            sk = cv2.waitKey(0) & 0xFF
            if sk == ord("l"):   return "left"
            elif sk == ord("r"): return "right"
            elif sk == 27:       return None

    def _flip(side: str) -> str:
        return "right" if side == "left" else "left"

    # Pre-populate result so that skipping (Q/ESC) returns existing data unchanged.
    # A copy of `marks` is taken here; ENTER will overwrite result with the live list.
    result: dict | None = (
        {"first": existing_first, "last": existing_last, "marks": list(marks)}
        if existing_first is not None and existing_last is not None
        else None
    )

    # --- Strictly Controlled State ---
    _cap_pos = 0
    paused = True
    last_set_tb = 0  # Tracks what WE told the trackbar to be, to ignore UI lag

    # Pre-read the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, last_img = cap.read()
    if not ret:
        last_img = np.zeros((720, 1280, 3), np.uint8)

    while True:

        # 1. Handle Trackbar Scrubbing (ONLY allowed when paused to prevent UI fighting)
        if paused:
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_set_tb)
            ret, frame_img = cap.read()
            if ret:
                _cap_pos = last_set_tb
                last_img = frame_img
            else:
                # Seek failed (common near EOF). Snap trackbar back to reality.
                cv2.setTrackbarPos("Frame", WIN, _cap_pos)
                last_set_tb = _cap_pos

        # 2. Handle Continuous Playback
        elif not paused:
            ret, frame_img = cap.read()
            if ret:
                # Trust our own +1 increment, ignore OpenCV's metadata
                _cap_pos += 1
                last_img = frame_img

                # Cap the UI update so it doesn't crash, but _cap_pos keeps going
                ui_pos = min(_cap_pos, total_frames - 1)
                cv2.setTrackbarPos("Frame", WIN, ui_pos)
                last_set_tb = ui_pos
            else:
                paused = True  # Auto-pause at true EOF

        # 3. Draw HUD & Wait for Input
        wait_ms = 30 if not paused else 16
        cv2.imshow(WIN, draw_hud(last_img, _cap_pos))
        key = cv2.waitKey(wait_ms) & 0xFF

        # --- Input Handling ---
        if key in (ord("q"), 27):  # skip
            break

        elif key == 13:  # ENTER → save
            missing = []
            if first_frame is None: missing.append("first-visible (F)")
            if last_frame is None: missing.append("last-visible (B)")
            if missing:
                print(f"  ⚠  Missing: {', '.join(missing)}")
                continue
            if first_frame >= last_frame:
                print(f"  ⚠  first ({first_frame}) must be < last ({last_frame})")
                continue
            result = {"first": first_frame, "last": last_frame, "marks": marks}
            break

        elif key == ord(" "):
            paused = not paused

        elif key == ord("d"):  # Step Forward 1 Frame
            paused = True
            ret, frame_img = cap.read()
            if ret:
                _cap_pos += 1
                last_img = frame_img
                ui_pos = min(_cap_pos, total_frames - 1)
                cv2.setTrackbarPos("Frame", WIN, ui_pos)
                last_set_tb = ui_pos

        elif key == ord("a"):  # Step Backward 1 Frame
            paused = True
            target = max(_cap_pos - 1, 0)
            if target != _cap_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame_img = cap.read()
                if ret:
                    _cap_pos = target
                    last_img = frame_img
                    ui_pos = min(_cap_pos, total_frames - 1)
                    cv2.setTrackbarPos("Frame", WIN, ui_pos)
                    last_set_tb = ui_pos

        elif key == ord("f"):
            action_stack.append({"type": "first", "prev": first_frame})
            first_frame = _cap_pos
            print(f"  → first_visible = frame {_cap_pos}  ({_cap_pos / fps:.3f}s)")

        elif key == ord("b"):
            action_stack.append({"type": "last", "prev": last_frame})
            last_frame = _cap_pos
            print(f"  → last_visible  = frame {_cap_pos}  ({_cap_pos / fps:.3f}s)")

        elif key == ord("t"):
            prev_side = current_side
            if current_side is None:
                side = _ask_side()
                if side is None: continue
                current_side = side
            else:
                current_side = _flip(current_side)
            marks.append({"event": "touchdown", "frame": _cap_pos,
                          "time_sec": round(_cap_pos / fps, 3), "side": current_side})
            action_stack.append({"type": "event", "prev_side": prev_side})
            print(f"  → touchdown ({current_side})  frame {_cap_pos}  ({_cap_pos / fps:.3f}s)")

        elif key == ord("o"):
            prev_side = current_side
            if current_side is None:
                side = _ask_side()
                if side is None: continue
                current_side = side
            marks.append({"event": "takeoff", "frame": _cap_pos,
                          "time_sec": round(_cap_pos / fps, 3), "side": current_side})
            action_stack.append({"type": "event", "prev_side": prev_side})
            print(f"  → takeoff   ({current_side})  frame {_cap_pos}  ({_cap_pos / fps:.3f}s)")

        elif key == ord("z") and action_stack:
            last_action = action_stack.pop()
            if last_action["type"] == "first":
                print(f"  ✗ undid first_visible (was frame {first_frame})")
                first_frame = last_action["prev"]
            elif last_action["type"] == "last":
                print(f"  ✗ undid last_visible (was frame {last_frame})")
                last_frame = last_action["prev"]
            elif last_action["type"] == "event":
                removed = marks.pop()
                current_side = last_action["prev_side"]
                print(f"  ✗ undid {removed['event']} ({removed.get('side', '?')}) "
                      f"at frame {removed['frame']}")

    cap.release()

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return result


# ── label generation ──────────────────────────────────────────────────────────

def marks_to_labels(
    video_path: str,
    marks: list[dict],
    first_visible: int,
    last_visible: int,
    meta: dict,
) -> pd.DataFrame:
    """
    Convert touchdown/takeoff marks to a per-frame DataFrame covering
    [first_visible, last_visible] (inclusive).

    All frames default to "flight".  Contact regions are filled in from the
    marks using the same paired-event logic as before:
      - Unmatched leading takeoff  → contact from first_visible to that frame.
      - Unmatched trailing touchdown → contact from that frame to last_visible.
      - Matched TD/TO pair → contact between those frames.
    """
    n = last_visible - first_visible + 1
    labels = np.full(n, "flight", dtype=object)
    sides  = np.full(n, None,     dtype=object)

    def to_idx(f: int) -> int:
        """Clip f to [first_visible, last_visible] and return array index."""
        return max(0, min(n - 1, int(f) - first_visible))

    sorted_marks = sorted(marks, key=lambda m: m["frame"])
    i = 0
    while i < len(sorted_marks):
        m = sorted_marks[i]
        if m["event"] == "takeoff":
            # leading takeoff — contact from first_visible
            e = to_idx(m["frame"])
            labels[0:e + 1] = "contact"
            sides[0:e + 1]  = m.get("side")
            i += 1
        else:  # touchdown
            if i + 1 < len(sorted_marks) and sorted_marks[i + 1]["event"] == "takeoff":
                nxt = sorted_marks[i + 1]
                s = to_idx(m["frame"])
                e = to_idx(nxt["frame"])
                if s > e:
                    print(f"  ⚠  skipping invalid pair: "
                          f"TD f{m['frame']} ≥ TO f{nxt['frame']}")
                else:
                    labels[s:e + 1] = "contact"
                    sides[s:e + 1]  = m.get("side")
                i += 2
            else:
                # trailing touchdown — contact to last_visible
                s = to_idx(m["frame"])
                labels[s:] = "contact"
                sides[s:]  = m.get("side")
                i += 1

    return pd.DataFrame({
        "person_id":    meta["person_id"],
        "study_id":     meta["study_id"],
        "video_id":     meta["video_id"],
        "Name":         meta["Name"],
        "Surname":      meta["Surname"],
        "video_path":   video_path,
        "frame_number": np.arange(first_visible, last_visible + 1),
        "label":        labels,
        "side":         sides,
    })


# ── metadata helpers ──────────────────────────────────────────────────────────

def _build_optojump_registry(clean_csv_path: str) -> tuple[dict[str, int], pd.DataFrame]:
    """
    Returns (name_key → person_id, raw dataframe).
    name_key = full_name.lower().strip()
    IDs are 1-indexed, sorted alphabetically by full_name.
    """
    df = pd.read_csv(clean_csv_path, encoding="utf-8-sig")
    assert "full_name" in df.columns, (
        f"'{clean_csv_path}' must have a 'full_name' column"
    )
    names    = sorted(df["full_name"].dropna().unique())
    registry = {n.lower().strip(): i + 1 for i, n in enumerate(names)}
    return registry, df


def _extract_optojump_meta(
    video_path: str,
    registry: dict[str, int],
    df_names: pd.DataFrame,
) -> tuple[int, int, int, str, str]:
    """(person_id, study_id, video_id, Name, Surname)"""
    folder = os.path.basename(os.path.dirname(video_path))
    assert folder.startswith("study_"), (
        f"Unexpected folder '{folder}' for optojump video '{video_path}'. "
        "Expected 'study_N'."
    )
    study_id = int(folder.split("_")[1])

    stem  = os.path.splitext(os.path.basename(video_path))[0]
    parts = stem.split("_")
    assert len(parts) >= 2, (
        f"Cannot parse video_id from filename '{stem}'. "
        "Expected '<name>_<video_id>'."
    )
    video_id = int(parts[-1])
    raw_name = "_".join(parts[:-1])
    name_key = raw_name.replace("_", " ").strip().lower()

    match = df_names[df_names["full_name"].str.lower().str.strip() == name_key]
    if not match.empty:
        row       = match.iloc[0]
        full_name = row["full_name"]
        Name      = str(row.get("name",    "")).strip()
        Surname   = str(row.get("surname", "")).strip()
    else:
        full_name = raw_name.replace("_", " ").title()
        tokens    = full_name.split()
        Name      = tokens[0] if tokens else ""
        Surname   = " ".join(tokens[1:]) if len(tokens) > 1 else ""

    key = full_name.lower().strip()
    if key not in registry:
        registry[key] = max(registry.values(), default=0) + 1
    person_id = registry[key]
    return person_id, study_id, video_id, Name, Surname


def _extract_tempos_meta(video_path: str) -> tuple[int, int, int, str, str]:
    """
    (person_id, study_id, video_id, "", "")

    Folder name == person_id (integer).
    File stem rules:
        "1"   → study_id=1, video_id=1
        "2"   → study_id=2, video_id=1
        "3.1" → study_id=3, video_id=1
        "3.2" → study_id=3, video_id=2
        ...
    """
    folder = os.path.basename(os.path.dirname(video_path))
    assert folder.isdigit(), (
        f"Tempos folder '{folder}' is not a numeric person ID "
        f"(video: '{video_path}')."
    )
    person_id = int(folder)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    if "." in stem:
        s, v     = stem.split(".", 1)
        study_id = int(s)
        video_id = int(v)
    else:
        study_id = int(stem)
        video_id = 1

    return person_id, study_id, video_id, "", ""


# ── video collection ──────────────────────────────────────────────────────────

def collect_optojump_videos(input_dir: str, clean_csv_path: str) -> list[dict]:
    registry, df_names = _build_optojump_registry(clean_csv_path)
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*.mov"), recursive=True))
    assert files, f"No .mov files found under '{input_dir}'"

    rows = []
    for vp in files:
        pid, sid, vid, name, sname = _extract_optojump_meta(vp, registry, df_names)
        rows.append(dict(video_path=vp,
                         person_id=pid, study_id=sid, video_id=vid,
                         Name=name, Surname=sname))
    return rows


def collect_tempos_videos(input_dir: str) -> list[dict]:
    files = sorted({
        f
        for pat in (os.path.join(input_dir, "*", "*.MP4"),
                    os.path.join(input_dir, "*", "*.mp4"))
        for f in glob.glob(pat)
    })
    assert files, f"No .MP4/.mp4 files found under '{input_dir}'"

    rows = []
    for vp in files:
        pid, sid, vid, name, sname = _extract_tempos_meta(vp)
        rows.append(dict(video_path=vp,
                         person_id=pid, study_id=sid, video_id=vid,
                         Name=name, Surname=sname))
    return rows


# ── helpers for reading existing per-frame output ─────────────────────────────

def _existing_info(
    ex_rows: pd.DataFrame,
) -> tuple[int | None, int | None, list[tuple[int, int]]]:
    """
    Given the existing rows for one video, return
    (first_visible, last_visible, contact_ranges).

    first_visible / last_visible are the min/max frame_number.
    contact_ranges is a list of (start, end) frame tuples derived from
    contiguous runs of label == "contact".
    """
    if ex_rows.empty:
        return None, None, []

    first_visible = int(ex_rows["frame_number"].min())
    last_visible  = int(ex_rows["frame_number"].max())

    contacts: list[tuple[int, int]] = []
    if "label" in ex_rows.columns:
        c_rows = (ex_rows[ex_rows["label"] == "contact"]
                  .sort_values("frame_number"))
        if not c_rows.empty:
            frames = c_rows["frame_number"].values
            gaps   = np.where(np.diff(frames) > 1)[0]
            starts = [int(frames[0])]      + [int(frames[g + 1]) for g in gaps]
            ends   = [int(frames[g]) for g in gaps] + [int(frames[-1])]
            contacts = list(zip(starts, ends))

    return first_visible, last_visible, contacts


# ── main processing loop ──────────────────────────────────────────────────────

def process_videos(
    video_list: list[dict],
    output_csv: str,
    fps: int,
) -> None:
    """
    Open every video in *video_list* for annotation.

    If *output_csv* already contains per-frame rows for a video they are shown
    in the UI as reference (cyan on the scrubber + terminal printout) and the
    video is still opened for re-annotation.  Saving (ENTER) replaces the old
    rows; skipping (Q) leaves them untouched.

    Data is persisted to disk after every accepted annotation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    # ── load existing output ─────────────────────────────────────────────────
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        current_df = pd.read_csv(output_csv)
        # Normalise stored paths to relative (from CWD) so they match what
        # collect_*_videos returns via glob.  Handles CSVs that stored absolute
        # paths (e.g. those produced by migrate_annotations.py).
        current_df["video_path"] = current_df["video_path"].apply(
            lambda p: os.path.relpath(p) if os.path.isabs(p) else p
        )
        n_existing = current_df["video_path"].nunique()
        print(f"Loaded {n_existing} already-annotated video(s) from '{output_csv}'.")
    else:
        current_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        n_existing = 0

    n_total = len(video_list)
    n_new   = sum(
        1 for v in video_list
        if v["video_path"] not in set(current_df["video_path"])
    )
    print(
        f"Total: {n_total} video(s).  "
        f"Already in output: {n_existing}.  "
        f"New (no annotation yet): {n_new}.\n"
    )

    # ── save helpers ─────────────────────────────────────────────────────────
    _notified = False

    def _persist() -> None:
        current_df.to_csv(output_csv, index=False)

    def save_progress() -> None:
        nonlocal _notified
        if _notified:
            return
        _notified = True
        _persist()
        n = len(current_df)
        print(f"\n✓ Saved {n} row(s) to '{output_csv}'." if n else "\nNothing to save.")

    def handle_signal(sig, _frame_arg) -> None:
        print(f"\nInterrupted (signal {sig}).")
        save_progress()
        raise SystemExit(0)

    atexit.register(save_progress)
    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── annotation loop ──────────────────────────────────────────────────────
    for i, meta in enumerate(video_list, 1):
        vp = meta["video_path"]
        print(
            f"[{i}/{n_total}]  {vp}\n"
            f"  person_id={meta['person_id']}"
            f"  study={meta['study_id']}"
            f"  video={meta['video_id']}"
            + (f"  {meta['Name']} {meta['Surname']}".rstrip() if meta["Name"] else "")
        )

        # existing rows for this video
        ex_rows = current_df[current_df["video_path"] == vp]
        ex_first, ex_last, ex_contacts = _existing_info(ex_rows)

        if ex_first is not None:
            print(f"  [existing] range  frames {ex_first} – {ex_last}")
        if ex_contacts:
            for cs, ce in ex_contacts:
                print(f"  [existing] contact  frames {cs} – {ce}")

        print("  F: first-visible   B: last-visible   "
              "T: touchdown   O: takeoff   ENTER: save   Q: skip\n")

        result = annotate_video_ui(
            vp, fps,
            ex_rows           = ex_rows,
            existing_first    = ex_first,
            existing_last     = ex_last,
            existing_contacts = ex_contacts,
        )

        if result is None:
            print("  Skipped (no existing annotation).\n")
            continue

        n_td = sum(1 for m in result["marks"] if m["event"] == "touchdown")
        n_to = sum(1 for m in result["marks"] if m["event"] == "takeoff")
        print(f"  Confirmed: {n_td} touchdown(s), {n_to} takeoff(s).")

        df_new = marks_to_labels(
            vp,
            result["marks"],
            result["first"],
            result["last"],
            meta,
        )

        # replace old rows for this video
        current_df = pd.concat(
            [current_df[current_df["video_path"] != vp], df_new],
            ignore_index=True,
        )
        _persist()
        _notified = False   # allow the final save message
        print(
            f"  ✓ {len(df_new)} frames labelled  "
            f"(first_visible={result['first']}, last_visible={result['last']})\n"
        )

    print("Batch processing complete.")
    save_progress()


# ── entry point ───────────────────────────────────────────────────────────────

DEFAULTS = {
    "optojump": dict(
        input_dir  = "data/input/optojump/",
        clean_csv  = "data/input/optojump/optojump_output/parsed/optojump_basic.csv",
        output_csv = "data/output/annotations/optojump/visibility_annotations.csv",
        fps        = 30,
    ),
    "tempos": dict(
        input_dir  = "data/input/tempos/",
        output_csv = "data/output/annotations/tempos/ml_labels.csv",
        fps        = 30,
    ),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate per-frame contact/flight labels with visibility range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=["optojump", "tempos"], required=True,
        help="Which dataset to process.",
    )
    parser.add_argument("--input-dir",  default=None, help="Override input directory.")
    parser.add_argument("--output-csv", default=None, help="Override output CSV path.")
    parser.add_argument("--fps",        type=int, default=None, help="Override FPS.")
    parser.add_argument(
        "--clean-csv",
        default=DEFAULTS["optojump"]["clean_csv"],
        help="Path to optojump_basic.csv (optojump only).",
    )
    args = parser.parse_args()

    d       = DEFAULTS[args.dataset]
    in_dir  = args.input_dir  or d["input_dir"]
    out_csv = args.output_csv or d["output_csv"]
    fps     = args.fps        or d["fps"]

    if args.dataset == "optojump":
        video_list = collect_optojump_videos(in_dir, args.clean_csv)
    else:
        video_list = collect_tempos_videos(in_dir)

    process_videos(video_list[140: ], out_csv, fps)
