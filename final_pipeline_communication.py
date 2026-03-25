import time
import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import serial.tools.list_ports
from ultralytics import YOLO
from ESP_serial import EspCli

CALIB_FILE = "calibration.json"

# --- CONFIG ---
TARGET_ID     = 255      # Robot ID to address; use EspCli.BROADCAST_ID (255) for all
ESP_PORT      = None     # Set to e.g. 'COM6' to skip auto-detection, or leave None
SEND_INTERVAL = 0.01 # seconds between ESP packets → 10 Hz
DEBUG         = False    # Set True to enable verbose per-packet logging


# ---------------------------------------------------------------------------
# COM port auto-detection
# ---------------------------------------------------------------------------

def detect_esp_port():
    """
    Scan available serial ports and return the best candidate for an ESP32.
    Falls back to manual selection if auto-detection is ambiguous.
    """
    ESP_KEYWORDS = [
        "cp210", "ch340", "ch341", "ftdi", "esp32", "esp8266",
        "uart", "usb serial", "usb-serial",
    ]

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("  No serial ports detected.")
        return None

    candidates = []
    for p in ports:
        combined = f"{p.description} {p.hwid}".lower()
        if any(kw in combined for kw in ESP_KEYWORDS):
            candidates.append(p)

    if len(candidates) == 1:
        print(f"  Auto-detected ESP port: {candidates[0].device}  ({candidates[0].description})")
        return candidates[0].device

    if len(candidates) > 1:
        print("  Multiple ESP-like ports found:")
        for i, p in enumerate(candidates):
            print(f"    [{i}] {p.device}  —  {p.description}")
        idx = input("  Select port index: ").strip()
        try:
            return candidates[int(idx)].device
        except (ValueError, IndexError):
            print("  Invalid selection.")
            return None

    print("  No ESP32 port auto-detected. Available ports:")
    for i, p in enumerate(ports):
        print(f"    [{i}] {p.device}  —  {p.description}")
    if len(ports) == 1:
        answer = input(f"  Only one port available ({ports[0].device}). Use it? [y/n]: ").strip().lower()
        return ports[0].device if answer == 'y' else None

    idx = input("  Select port index (or press Enter to cancel): ").strip()
    if not idx:
        return None
    try:
        return ports[int(idx)].device
    except (ValueError, IndexError):
        print("  Invalid selection.")
        return None


# ---------------------------------------------------------------------------
# Calibration loader — supports both legacy and keyed JSON formats
# ---------------------------------------------------------------------------

def load_calibration():
    """
    Load the transformation matrix from calibration.json.

    Supports:
      - Legacy format: {"transformation_matrix": [...]}
      - Keyed format:  {"robot_01": {"transformation_matrix": [...]}, ...}

    Returns a (3, 4) numpy array, or None on failure.
    """
    if not os.path.exists(CALIB_FILE):
        print(f"Error: {CALIB_FILE} not found.")
        return None

    with open(CALIB_FILE, 'r') as f:
        data = json.load(f)

    if "transformation_matrix" in data:
        return np.array(data["transformation_matrix"]).reshape(3, 4)

    valid = {k: v for k, v in data.items()
             if isinstance(v, dict) and "transformation_matrix" in v}

    if not valid:
        print("Error: No valid calibration entries found in calibration.json.")
        return None

    if len(valid) == 1:
        calib_id = next(iter(valid))
        print(f"Using calibration: '{calib_id}'")
        return np.array(valid[calib_id]["transformation_matrix"]).reshape(3, 4)

    print("Multiple calibrations found:")
    ids = list(valid.keys())
    for i, cid in enumerate(ids):
        desc = valid[cid].get("description", "")
        print(f"  [{i}] {cid:<20} {desc}")

    idx = input("Select calibration index: ").strip()
    try:
        calib_id = ids[int(idx)]
        return np.array(valid[calib_id]["transformation_matrix"]).reshape(3, 4)
    except (ValueError, IndexError):
        print("Invalid selection.")
        return None


# ---------------------------------------------------------------------------
# Keypoint target selector
# ---------------------------------------------------------------------------

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
        return int(person_kpts[0][0]), int(person_kpts[0][1]), "Nose"

    elif left_valid and right_valid:
        lx, ly = float(person_kpts[3][0]), float(person_kpts[3][1])
        rx, ry = float(person_kpts[4][0]), float(person_kpts[4][1])
        return int((lx + rx) / 2), int((ly + ry) / 2), "Ears(avg)"

    elif left_valid:
        return int(person_kpts[3][0]), int(person_kpts[3][1]), "Left Ear"

    elif right_valid:
        return int(person_kpts[4][0]), int(person_kpts[4][1]), "Right Ear"

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- ESP32 connection ---
    port = ESP_PORT or detect_esp_port()
    if not port:
        port = input("Enter COM port manually (e.g. COM6 or /dev/ttyUSB0): ").strip()
    if not port:
        print("No serial port specified. Exiting.")
        return

    try:
        esp = EspCli(port)
    except Exception as e:
        print(f"Failed to connect to ESP: {e}")
        return

    # --- Calibration ---
    M_transform = load_calibration()
    if M_transform is None:
        esp.close()
        return

    # --- YOLO & RealSense setup ---
    model = YOLO("yolo11n-pose.pt")

    pipeline = rs.pipeline()
    rs_config = rs.config()
    width, height = 640, 480
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)

    profile    = pipeline.start(rs_config)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    align      = rs.align(rs.stream.color)

    spatial      = rs.spatial_filter()
    hole_filling = rs.hole_filling_filter()

    locked_id   = None
    last_send   = 0.0
    frame_count = 0
    skip_reason = "none"

    try:
        while True:
            frame_count += 1

            # --- Periodic status line replaces per-frame prints ---
            if frame_count % 150 == 0:
                print(f"[frame {frame_count}] last_skip='{skip_reason}'  "
                      f"time_since_send={time.time() - last_send:.2f}s")

            frames         = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame    = aligned_frames.get_depth_frame()
            color_frame    = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                skip_reason = "no frames"
                continue

            depth_frame  = spatial.process(depth_frame)
            depth_frame  = hole_filling.process(depth_frame).as_depth_frame()
            color_image  = np.asanyarray(color_frame.get_data())

            results = model.track(color_image, persist=True, classes=[0], verbose=False)

            for result in results:
                if result.keypoints is None or result.boxes is None:
                    skip_reason = "no keypoints/boxes"
                    continue

                kpts_xy   = result.keypoints.xy
                kpts_conf = result.keypoints.conf
                track_ids = result.boxes.id

                if track_ids is None:
                    skip_reason = "no track IDs"
                    continue

                track_ids = track_ids.int().tolist()

                if locked_id is None:
                    locked_id = min(track_ids)
                    print(f"\n[Locked onto track ID: {locked_id}]")

                if locked_id not in track_ids:
                    locked_id = min(track_ids)
                    print(f"\n[Re-acquired — new track ID: {locked_id}]")

                person_idx  = track_ids.index(locked_id)
                person_kpts = kpts_xy[person_idx]

                if len(person_kpts) < 5:
                    skip_reason = "too few keypoints"
                    continue

                person_conf = kpts_conf[person_idx] if kpts_conf is not None else np.ones(len(person_kpts))

                target = get_target_pixel(person_kpts, person_conf)
                if target is None:
                    skip_reason = "no valid target keypoint"
                    continue

                tx, ty, target_label = target
                if not (0 < tx < width and 0 < ty < height):
                    skip_reason = "target out of bounds"
                    continue

                depth_z = depth_frame.get_distance(tx, ty)
                if depth_z <= 0:
                    skip_reason = "bad depth"
                    continue

                # Deproject pixel → 3D camera frame
                p_cam = rs.rs2_deproject_pixel_to_point(intrinsics, [tx, ty], depth_z)

                # Transform to robot frame
                p_robot    = M_transform @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
                rx, ry, rz = p_robot
                
                # --- Throttle to SEND_INTERVAL before sending ---
                now = time.time()
                if now - last_send < SEND_INTERVAL:
                    skip_reason = "throttled"
                    continue
                last_send = now
                
                # Pass non transformed
                esp.update_pos(TARGET_ID, p_cam[0], p_cam[1], p_cam[2] )
                skip_reason = "none"
                
                yaw  = np.degrees(np.arctan2(ry, rx))
                tilt = np.degrees(np.arctan2(rz, np.sqrt(rx**2 + ry**2)))
                
                # Print points
                print(
                    f"{target_label:<10} | {rx:>7.2f} | {ry:>7.2f} | {rz:>7.2f} "
                    f"| {yaw:>10.2f} | {tilt:>10.2f}"
                )

                # --- Send angles to ESP ---
                '''
                try:
                    esp.update_angle(TARGET_ID, yaw, tilt)
                except Exception as e:
                    print(f"\n[ESP send failed at frame {frame_count}]: {e}")
                    break
                '''
                
                cv2.circle(color_image, (tx, ty), 5, (0, 0, 255), -1)
                cv2.putText(color_image, f"Y:{yaw:.1f} T:{tilt:.1f}",
                            (tx + 10, ty - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(color_image, target_label,
                            (tx + 10, ty - 5),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

            cv2.imshow("Robot Frame Tracking", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        esp.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
