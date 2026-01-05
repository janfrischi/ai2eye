User: @workspace Could you please create a super simple readme on how to use #file:detect_aruco.py and #file:final_pipeline.py. The detection script is used in the first step to get the transformation matrix from camera coordinate system to the robot coordinate system. The final pipeline then uses yolo11 posetracking to track the nose, uses the transformation matrix from the previous steps and essentially calculates the yaw and tilt angles for the 2DOF robot. 

Workspace: Collecting workspace information# AI2EYE: Camera-to-Robot Calibration and Pose Tracking

This project provides scripts for calibrating a RealSense camera to a robot coordinate system using ArUco markers and then tracking human poses (specifically the nose) to compute yaw and tilt angles for a 2DOF robot.

## Prerequisites

- RealSense camera (e.g., D435 or similar).
- ArUco markers (4x4_50 dictionary, 10cm size recommended).
- Python environment with dependencies installed (see requirements.txt).
- Known robot coordinates for calibration points.

## Step 1: Calibration with detect_aruco.py

This script detects ArUco markers and computes a transformation matrix from the camera coordinate system to the robot coordinate system.

### How to Run

1. Ensure your RealSense camera is connected.
2. Run the script:
   ```bash
   python detect_aruco.py
   ```
3. If a previous calibration exists, choose to use it or recalibrate.
4. **Phase 1: Stabilized Capture**
   - Place an ArUco marker in view.
   - Press 'c' to capture each of the 4 points (stabilizes over 10 frames).
   - Repeat for 4 points.
5. **Phase 2: Robot Coordinates**
   - For each captured point, enter the corresponding robot (x,y,z) coordinates when prompted (comma-separated, e.g., `1.0,2.0,3.0`).
6. **Phase 3: Computation**
   - The script computes and saves the transformation matrix to calibration.json.
7. Press 'q' to exit.

## Step 2: Pose Tracking with final_pipeline.py

This script uses YOLO11 pose estimation to track the nose of a person, transforms coordinates using the calibration matrix, and calculates yaw and tilt angles for the robot.

### How to Run

1. Ensure calibration.json exists from Step 1.
2. Run the script:
   ```bash
   python final_pipeline.py
   ```
3. The script will display real-time tracking:
   - Nose position in robot coordinates (X, Y, Z).
   - Yaw and tilt angles.
   - FPS in the video feed.
4. Press 'q' to exit.

## Notes

- Ensure the camera is properly aligned and calibrated for accurate results.
- The robot is assumed to be 2DOF (pan and tilt).
- If calibration fails, rerun detect_aruco.py.
- For issues, check camera connection and marker visibility.