import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import serial
from ultralytics import YOLO

CALIB_FILE = "calibration.json"

# --- SERIAL SETUP ---
try:
    ser = serial.Serial('COM6', 115200, timeout=0.1)
    print("Serial port opened successfully.")
except Exception as e:
    print(f"Could not open serial port: {e}")
    ser = None

def load_calibration():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, 'r') as f:
            data = json.load(f)
            return np.array(data["transformation_matrix"])
    return None

def get_target_pixel(person_kpts, person_conf):
    """
    Determines the best 2D pixel target from keypoints using the
    nose-priority fallback strategy.

    YOLO COCO keypoint indices:
        0: Nose
        3: Left Ear
        4: Right Ear

    Priority:
        1. Nose visible           -> track nose
        2. Both ears visible      -> average ears (person faces away)
        3. One ear visible        -> track that ear (profile view)

    Returns:
        (tx, ty, label) or None if no valid keypoint is found.
    """
    CONF_THRESHOLD = 0.5

    def is_valid(idx):
        if idx >= len(person_kpts) or idx >= len(person_conf):
            return False
        kpt  = person_kpts[idx]
        conf = person_conf[idx]
        return float(conf) > CONF_THRESHOLD and float(kpt[0]) > 0 and float(kpt[1]) > 0

    nose_valid  = is_valid(0)
    left_valid  = is_valid(3)
    right_valid = is_valid(4)

    if nose_valid:
        # Priority 1: nose visible — most accurate for frontal tracking
        return int(person_kpts[0][0]), int(person_kpts[0][1]), "Nose"

    elif left_valid and right_valid:
        # Priority 2: person faces away — average ears as nose estimate
        lx, ly = float(person_kpts[3][0]), float(person_kpts[3][1])
        rx, ry = float(person_kpts[4][0]), float(person_kpts[4][1])
        tx = int((lx + rx) / 2)
        ty = int((ly + ry) / 2)
        return tx, ty, "Ears(avg)"

    elif left_valid:
        # Priority 3: profile view, only left ear visible
        return int(person_kpts[3][0]), int(person_kpts[3][1]), "Left Ear"

    elif right_valid:
        # Priority 3: profile view, only right ear visible
        return int(person_kpts[4][0]), int(person_kpts[4][1]), "Right Ear"

    return None


def main():
    # 1. Initialization
    model = YOLO("yolo11n-pose.pt")

    M_transform = load_calibration()
    if M_transform is None:
        print("Error: calibration.json not found.")
        return

    pipeline = rs.pipeline()
    config = rs.config()
    width, height = 640, 480
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)

    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color)

    # Filters for depth stability
    spatial      = rs.spatial_filter()
    hole_filling = rs.hole_filling_filter()

    locked_id = None  # Track ID of the person we are following

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Filter depth
            depth_frame = spatial.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame).as_depth_frame()

            color_image = np.asanyarray(color_frame.get_data())

            # YOLO inference
            results = model.track(color_image, persist=True, classes=[0], verbose=False)

            for result in results:
                if result.keypoints is None or result.boxes is None:
                    continue

                kpts_xy   = result.keypoints.xy
                kpts_conf = result.keypoints.conf
                track_ids = result.boxes.id  # None if tracking lost this frame

                if track_ids is None:
                    continue

                track_ids = track_ids.int().tolist()

                # --- Lock onto the first valid track ID seen ---
                if locked_id is None:
                    locked_id = min(track_ids)
                    print(f"\n[Locked onto track ID: {locked_id}]")

                # If our locked person has left the frame, re-acquire
                if locked_id not in track_ids:
                    locked_id = min(track_ids)
                    print(f"\n[Re-acquired — new track ID: {locked_id}]")

                person_idx  = track_ids.index(locked_id)
                person_kpts = kpts_xy[person_idx]

                if len(person_kpts) < 5:
                    continue

                person_conf = kpts_conf[person_idx] if kpts_conf is not None else np.ones(len(person_kpts))

                target = get_target_pixel(person_kpts, person_conf)
                if target is None:
                    continue

                tx, ty, target_label = target
                if not (0 < tx < width and 0 < ty < height):
                    continue

                depth_z = depth_frame.get_distance(tx, ty)
                if depth_z <= 0:
                    continue

                # Deproject pixel to 3D camera frame
                p_cam = rs.rs2_deproject_pixel_to_point(intrinsics, [tx, ty], depth_z)

                # Transform to robot frame: M * [x, y, z, 1]^T
                p_robot = M_transform @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
                rx, ry, rz = p_robot

                # Calculate yaw and tilt angles
                yaw             = np.degrees(np.arctan2(ry, rx))
                horizontal_dist = np.sqrt(rx**2 + ry**2)
                tilt            = np.degrees(np.arctan2(rz, horizontal_dist))

                # --- SEND TO ESP32 ---
                if ser and ser.is_open:
                    data_packet = f"{yaw:.2f},{tilt:.2f}\n"
                    print(f"Sending Yaw and Tilt Angles: {data_packet.strip()}")
                    ser.write(data_packet.encode('utf-8'))

                # Visuals
                cv2.circle(color_image, (tx, ty), 5, (0, 0, 255), -1)
                cv2.putText(
                    color_image,
                    f"Y:{yaw:.1f} T:{tilt:.1f}",
                    (tx + 10, ty - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                )
                cv2.putText(
                    color_image,
                    target_label,
                    (tx + 10, ty - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1
                )

            cv2.imshow("Robot Frame Tracking", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if ser:
            ser.close()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()