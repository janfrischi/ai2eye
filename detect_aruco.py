import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import numpy as np
import json
import os
from collections import deque

CALIB_FILE = "calibration.json"

# Save calibration matrix to JSON
def save_calibration(matrix):
    # Convert numpy array to list for JSON serialization
    data = {"transformation_matrix": matrix.tolist()}
    with open(CALIB_FILE, 'w') as f:
        json.dump(data, f)
    print(f"\nCalibration saved to {CALIB_FILE}")

# Load calibration matrix from JSON
def load_calibration():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, 'r') as f:
            data = json.load(f)
            return np.array(data["transformation_matrix"])
    return None

# --- 1. SETUP REALSENSE CAMERA ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Get Intrinsics
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
# Assemble camera matrix fx, fy, ppx, ppy (focal lengths and principal point)
camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
dist_coeffs = np.array(intr.coeffs)

# --- 2. ARUCO SETUP ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
# Detector parameters, includes settings for how to treat corners, thresholds, etc.
parameters = aruco.DetectorParameters()
# Use interpolation refinement for better corner accuracy
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
parameters.adaptiveThreshWinSizeStep = 3
# Create the detector object
detector = aruco.ArucoDetector(aruco_dict, parameters)
# Physical constant of marker used for solvePnP math
marker_length = 0.10  

# --- 3. CALIBRATION CHECK ---
# Load calibration from .json file if exists
M_transform = load_calibration()

if M_transform is not None:
    choice = input("Existing calibration found. Use it? (y/n): ").lower()
    if choice != 'y':
        M_transform = None

if M_transform is None:
    # --- PHASE 1: DATA COLLECTION CAMERA WORLD, record 4 points in the camera coordinate system ---
    # List to hold point pairs
    pts_camera, pts_robot = [], []
    print("\n--- PHASE 1: STABILIZED CAPTURE ---")
    smooth_buffer = deque(maxlen=10)

    while len(pts_camera) < 4:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        frame = np.asanyarray(color_frame.get_data())
        # Detect ArUco markers
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if ids is not None:
            obj_points = np.array([[-marker_length/2,  marker_length/2, 0],
                                   [ marker_length/2,  marker_length/2, 0],
                                   [ marker_length/2, -marker_length/2, 0],
                                   [-marker_length/2, -marker_length/2, 0]], dtype=np.float32)
            
            # Solve PnP to get rotation and translation vectors
            _, rvec, tvec = cv2.solvePnP(obj_points, corners[0], camera_matrix, dist_coeffs)
            # Use smoothing to stabilize readings
            smooth_buffer.append(tvec.flatten())
            # Average over last 10 readings -> Creates mor stable coordinate for calibration
            avg_xyz = np.mean(smooth_buffer, axis=0)

            aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
            cv2.putText(frame, f"Captured: {len(pts_camera)}/4", (10, 30), 1, 1.5, (0, 255, 0), 2)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        # When 'c' is pressed, capture the averaged point and save into pts_camera list
        if key == ord('c') and ids is not None:
            pts_camera.append(avg_xyz)
            print(f"Captured Point {len(pts_camera)}: {avg_xyz}")

    # --- PHASE 2: USER INPUT / CALCULATION ROBOT WORLD ---
    for i in range(4):
        val = input(f"Enter Robot (x,y,z) for Point {i+1}: ")
        pts_robot.append([float(x.strip()) for x in val.split(',')])

    # --- PHASE 3: COMPUTE TRANSFORMATION, cv2.estimateAffine3D uses Least Squares estimation ---
    print("\n--- PHASE 3: CALIBRATION COMPUTATION ---")
    ret, M_transform, _ = cv2.estimateAffine3D(np.array(pts_camera), np.array(pts_robot))
    print("Computed Transformation Matrix:\n", M_transform)
    # Save transformation matrix to .json
    if ret:
        save_calibration(M_transform)
    else:
        print("Calibration error."); exit()

# --- PHASE 4: LIVE TRACKING ---
print("\n--- PHASE 4: TRACKING ACTIVE ---")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        frame = np.asanyarray(color_frame.get_data())
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if ids is not None:
            # Marker logic
            obj_points = np.array([[-marker_length/2,  marker_length/2, 0],
                                   [ marker_length/2,  marker_length/2, 0],
                                   [ marker_length/2, -marker_length/2, 0],
                                   [-marker_length/2, -marker_length/2, 0]], dtype=np.float32)
            _, rvec, tvec = cv2.solvePnP(obj_points, corners[0], camera_matrix, dist_coeffs)
            
            # 1. Transform Point
            p_robot = M_transform @ np.append(tvec.flatten(), 1.0)
            
            # 2. Derive Angles
            xr, yr, zr = p_robot
            pan = np.degrees(np.arctan2(yr, xr))
            tilt = np.degrees(np.arctan2(zr, np.sqrt(xr**2 + yr**2)))
            
            # 3. Visuals
            cv2.putText(frame, f"PAN: {pan:.1f} TILT: {tilt:.1f}", (10, 30), 1, 1.5, (0, 255, 0), 2)
            print(f"Robot Frame: {p_robot}", end='\r')

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()