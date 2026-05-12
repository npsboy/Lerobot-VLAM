"""Preprocess imitation learning recordings into X/Y samples.

Creates `preprocessed_data.json` with structure:
{
    "X": [ [ [...], ... ], [ [...], ... ], ... ],
    "Y": [ [...], [...], ... ],
    "X_mean": [...],
    "X_std": [...],
    "Y_mean": [...],
    "Y_std": [...]
}

All-joints setup:
- X has WINDOW_SIZE frames per sample; each frame has 18 values (for 6 joints):
    1) current angles for all 6 joints (6 values)
    2) target - current for all 6 joints (6 values)
    3) target angles for all 6 joints (6 values)
- Y has 6 values per sample:
    - per-joint delta from the last frame in the window to the next frame

Data is normalized using z-score normalization (standardization):
normalized = (value - mean) / std_dev
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


JOINT_INDICES = list(range(6))  # use all 6 joints (0..5)
INPUT_WIDTH = 6 + 6 + 6  # current(6) + target-current(6) + target(6) = 18
OUTPUT_WIDTH = 6  # predict per-joint delta for all 6 joints
WINDOW_SIZE = 10
WINDOW_STRIDE = 1


def _ensure_len(arr: List[float], n: int) -> List[float]:
    if arr is None:
        return [0.0] * n
    if len(arr) >= n:
        return arr[:n]
    return list(arr) + [0.0] * (n - len(arr))


def _build_frame_features(curr_pos: List[float], target_vals: List[float]) -> List[float]:
    sample: List[float] = []
    for i, tgt in zip(JOINT_INDICES, target_vals):
        cur = float(curr_pos[i])
        sample.append(cur)               # current angle
    for i, tgt in zip(JOINT_INDICES, target_vals):
        cur = float(curr_pos[i])
        sample.append(float(tgt - cur))  # target - current
    for i, tgt in zip(JOINT_INDICES, target_vals):
        sample.append(float(tgt))        # target angle
    if len(sample) != INPUT_WIDTH:
        raise ValueError(
            f"Built sample width {len(sample)} does not match expected input width {INPUT_WIDTH}"
        )
    return sample


def process(input_path: str = "imitation_learning_recordings.json",
            output_path: str = "preprocessed_data.json") -> dict:
    inp = Path(input_path)
    out = Path(output_path)
    # read with utf-8-sig to gracefully handle files that start with a BOM
    with inp.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    X_sessions = []
    Y_sessions = []

    for session in data:
        frames = session.get("frames", [])
        target = session.get("target", {})
        target_pos = target.get("positions")
        if not target_pos:
            # skip sessions without a known target
            continue

        target_pos_6 = _ensure_len(target_pos, 6)
        # Use all 6 joints
        target_vals = [float(target_pos_6[i]) for i in JOINT_INDICES]
        n = len(frames)
        session_X = []
        session_Y = []
        # Use sliding windows of WINDOW_SIZE frames, predict delta to the next frame.
        for t in range(0, n - WINDOW_SIZE, WINDOW_STRIDE):
            window: List[List[float]] = []
            for w in range(t, t + WINDOW_SIZE):
                frame = frames[w]
                curr_pos = _ensure_len(frame.get("positions"), 6)
                window.append(_build_frame_features(curr_pos, target_vals))

            last_pos = _ensure_len(frames[t + WINDOW_SIZE - 1].get("positions"), 6)
            next_pos = _ensure_len(frames[t + WINDOW_SIZE].get("positions"), 6)
            delta: List[float] = []
            for i in JOINT_INDICES:
                delta.append(float(next_pos[i] - last_pos[i]))
            if len(delta) != OUTPUT_WIDTH:
                raise ValueError(
                    f"Built label width {len(delta)} does not match expected output width {OUTPUT_WIDTH}"
                )

            session_X.append(window)
            session_Y.append(delta)

        if session_X:
            X_sessions.append(session_X)
            Y_sessions.append(session_Y)

    # Normalize X and Y using z-score normalization (standardization)
    X_flat = [window for session_windows in X_sessions for window in session_windows]
    Y_flat = [sample for session_samples in Y_sessions for sample in session_samples]

    if not X_flat:
        raise ValueError("No valid training samples were produced from the input recordings")

    X_arr = np.array(X_flat, dtype=np.float32)
    Y_arr = np.array(Y_flat, dtype=np.float32)

    X_arr_2d = X_arr.reshape(-1, INPUT_WIDTH)
    X_mean = X_arr_2d.mean(axis=0)
    X_std = X_arr_2d.std(axis=0)
    # Avoid division by zero: add a small epsilon
    X_std = np.where(X_std == 0, 1.0, X_std)
    X_normalized = (X_arr - X_mean) / X_std

    Y_mean = Y_arr.mean(axis=0)
    Y_std = Y_arr.std(axis=0)
    # Avoid division by zero: add a small epsilon
    Y_std = np.where(Y_std == 0, 1.0, Y_std)
    Y_normalized = (Y_arr - Y_mean) / Y_std

    out.write_text(json.dumps({
        "X": X_normalized.tolist(),
        "Y": Y_normalized.tolist(),
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
        "Y_mean": Y_mean.tolist(),
        "Y_std": Y_std.tolist()
    }, indent=2))
    return {"X_len": len(X_flat), "Y_len": len(Y_flat), "out": str(out)}


def _cli():
    p = argparse.ArgumentParser(description="Preprocess imitation learning recordings")
    p.add_argument("--input", "-i", default="imitation_learning_recordings.json")
    p.add_argument("--output", "-o", default="preprocessed_data.json")
    args = p.parse_args()
    res = process(args.input, args.output)
    print(f"Wrote {res['out']}: X={res['X_len']} Y={res['Y_len']}")


if __name__ == "__main__":
    _cli()
