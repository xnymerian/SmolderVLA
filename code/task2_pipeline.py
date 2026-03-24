import argparse
from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 - Hand to Robot pipeline starter")
    parser.add_argument("--input-video", required=True, help="Path to the reference hand video")
    parser.add_argument("--output-dir", default="outputs/task2", help="Output directory for Task 2 artifacts")
    parser.add_argument(
        "--model-path",
        default="models/hand_landmarker.task",
        help="Path to MediaPipe Hand Landmarker task file",
    )
    return parser.parse_args()


def _download_task_model_if_missing(model_path: Path) -> None:
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    print(f"Downloading model: {url}")
    urllib.request.urlretrieve(url, model_path)


def _extract_with_legacy_solutions(video_path: str):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    traj = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            wrist = lm[0]
            index_mcp = lm[5]
            middle_mcp = lm[9]
            pinky_mcp = lm[17]
            traj.append(
                [
                    frame_idx,
                    float(wrist.x),
                    float(wrist.y),
                    float(wrist.z),
                    float(index_mcp.x),
                    float(index_mcp.y),
                    float(index_mcp.z),
                    float(middle_mcp.x),
                    float(middle_mcp.y),
                    float(middle_mcp.z),
                    float(pinky_mcp.x),
                    float(pinky_mcp.y),
                    float(pinky_mcp.z),
                ]
            )
        frame_idx += 1

    cap.release()
    hands.close()
    return np.array(traj, dtype=np.float32)


def _extract_with_tasks_api(video_path: str, model_path: Path):
    _download_task_model_if_missing(model_path)

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    traj = []
    frame_idx = 0
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_idx / fps) * 1000.0)
            result = landmarker.detect_for_video(mp_image, ts_ms)
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                wrist = lm[0]
                index_mcp = lm[5]
                middle_mcp = lm[9]
                pinky_mcp = lm[17]
                traj.append(
                    [
                        frame_idx,
                        float(wrist.x),
                        float(wrist.y),
                        float(wrist.z),
                        float(index_mcp.x),
                        float(index_mcp.y),
                        float(index_mcp.z),
                        float(middle_mcp.x),
                        float(middle_mcp.y),
                        float(middle_mcp.z),
                        float(pinky_mcp.x),
                        float(pinky_mcp.y),
                        float(pinky_mcp.z),
                    ]
                )
            frame_idx += 1

    cap.release()
    return np.array(traj, dtype=np.float32)


def extract_wrist_trajectory(video_path: str, model_path: Path):
    """
    Extract wrist and key hand landmarks from a video using MediaPipe.
    Returns per-frame normalized coordinates.
    """
    if hasattr(mp, "solutions"):
        print("Using MediaPipe legacy solutions API.")
        return _extract_with_legacy_solutions(video_path)
    print("Using MediaPipe Tasks API (hand_landmarker.task).")
    return _extract_with_tasks_api(video_path, model_path)


def save_csv(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(
        [
            "frame",
            "wrist_x",
            "wrist_y",
            "wrist_z",
            "index_mcp_x",
            "index_mcp_y",
            "index_mcp_z",
            "middle_mcp_x",
            "middle_mcp_y",
            "middle_mcp_z",
            "pinky_mcp_x",
            "pinky_mcp_y",
            "pinky_mcp_z",
        ]
    )
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)

    traj = extract_wrist_trajectory(args.input_video, model_path)
    if traj.size == 0:
        print("Warning: no hand was detected in the video.")
        return

    csv_path = out_dir / "wrist_trajectory.csv"
    save_csv(csv_path, traj)
    print(f"Done: wrist trajectory saved -> {csv_path}")
    print("Next step: map this trajectory to VX300s end-effector targets and solve IK.")


if __name__ == "__main__":
    main()

