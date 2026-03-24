import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create side-by-side Task 2 video")
    parser.add_argument("--left-video", required=True, help="Reference hand video")
    parser.add_argument("--right-video", required=True, help="Robot simulation video")
    parser.add_argument("--output-video", default="outputs/task2_side_by_side.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--length-mode",
        choices=["max", "min"],
        default="max",
        help="max: freeze last frame of shorter video until longer one ends",
    )
    return parser.parse_args()


def resize_keep_aspect(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = target_h / float(h)
    tw = max(1, int(round(w * scale)))
    return cv2.resize(frame, (tw, target_h), interpolation=cv2.INTER_AREA)


def put_label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (8, 8), (220, 42), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def read_frame(reader, idx: int, n_total: int, last_frame: np.ndarray) -> np.ndarray:
    if idx < n_total:
        return reader.get_data(idx)
    return last_frame


def main():
    args = parse_args()
    out_path = Path(args.output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    left_reader = imageio.get_reader(args.left_video)
    right_reader = imageio.get_reader(args.right_video)
    n_left = left_reader.count_frames()
    n_right = right_reader.count_frames()
    n = max(n_left, n_right) if args.length_mode == "max" else min(n_left, n_right)

    first_left = left_reader.get_data(0)
    first_right = right_reader.get_data(0)
    last_left = left_reader.get_data(n_left - 1)
    last_right = right_reader.get_data(n_right - 1)
    target_h = min(first_left.shape[0], first_right.shape[0], 480)

    writer = imageio.get_writer(str(out_path), fps=args.fps)
    for i in range(n):
        l = read_frame(left_reader, i, n_left, last_left)
        r = read_frame(right_reader, i, n_right, last_right)
        l = resize_keep_aspect(l, target_h)
        r = resize_keep_aspect(r, target_h)
        l = put_label(l, "Reference Hand")
        r = put_label(r, "Robot Sim")
        pad = np.zeros((target_h, 12, 3), dtype=np.uint8)
        combo = np.concatenate([l, pad, r], axis=1)
        writer.append_data(combo)

    writer.close()
    left_reader.close()
    right_reader.close()
    print(f"Done: side-by-side video saved -> {out_path}")


if __name__ == "__main__":
    main()

