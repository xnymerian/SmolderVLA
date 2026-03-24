# Case Study

This repository contains my solution for the case study.

## Overview

The case study has two tasks:

1. **Task 1 - Embodied AI with SmolVLA + LeRobot**
   - Run a SmolVLA policy in MuJoCo.
   - Evaluate behavior and provide a short report.

2. **Task 2 - Vision-based Teleoperation (Hand to Robot)**
   - Extract hand landmarks from a reference video.
   - Map motion to **Trossen VX300s (Right Arm)** in MuJoCo using IK.
   - Produce a side-by-side comparison video.

## Final Deliverables

- `deliverables/task1_report.docx` - Task 1 report
- `deliverables/task1_Visualization.mp4` - Task 1 visualization
- `deliverables/task2_SidebySide.mp4` - Task 2 final side-by-side video

## Project Structure

- `task1_smolvla_inference.py` - Task 1 policy inference in MuJoCo
- `task2_pipeline.py` - Hand landmark extraction from video (MediaPipe)
- `task2_vx300s_replay.py` - VX300s IK replay from extracted trajectory
- `task2_make_side_by_side.py` - Side-by-side video creation utility
- 'wrist_trajectory.csv' - Generated hand trajectory used by replay

