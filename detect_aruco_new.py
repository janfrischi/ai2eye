import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import numpy as np
import json
import os
from collections import deque
import serial.tools.list_ports
from ESP_serial import EspConfig


CALIB_FILE = "calibration.json"

# ---------------------------------------------------------------------------
# Calibration file helpers — all matrices live in one JSON, keyed by ID
# ---------------------------------------------------------------------------

def load_all_calibrations():
    """Load the entire calibration store. Returns empty dict if file missing."""
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_calibration(calib_id, matrix, description=""):
    """Save (or overwrite) a single calibration entry by ID."""
    store = load_all_calibrations()
    store[calib_id] = {
        "transformation_matrix": matrix.flatten().tolist(),
        "description": description
    }
    with open(CALIB_FILE, 'w') as f:
        json.dump(store, f, indent=2)
    print(f"\nCalibration '{calib_id}' saved to {CALIB_FILE}")


def load_calibration(calib_id):
    """Retrieve a single calibration matrix by ID. Returns None if not found."""
    store = load_all_calibrations()
    if calib_id not in store:
        return None
    entry = store[calib_id]
    if isinstance(entry, dict):
        return np.array(entry["transformation_matrix"]).reshape(3, 4)
    return None


def list_calibrations():
    """Print all stored calibration IDs with optional descriptions."""
    store = load_all_calibrations()
    if not store:
        print("  (no calibrations stored yet)")
        return

    if any(isinstance(v, list) for v in store.values()):
        print("  (legacy calibration.json detected — enter a new ID to migrate to new format)")
        return

    print(f"  {'ID':<20} Description")
    print(f"  {'-'*20} -----------")
    for cid, entry in store.items():
        desc = entry.get("description", "")
        print(f"  {cid:<20} {desc}")


def delete_calibration(calib_id):
    """Remove a calibration entry by ID."""
    store = load_all_calibrations()
    if calib_id in store:
        del store[calib_id]
        with open(CALIB_FILE, 'w') as f:
            json.dump(store, f, indent=2)
        print(f"Calibration '{calib_id}' deleted.")
    else:
        print(f"ID '{calib_id}' not found.")


# ---------------------------------------------------------------------------
# Camera & ArUco setup
# ---------------------------------------------------------------------------

pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile  = pipeline.start(config)

intr          = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0,       1        ]])
dist_coeffs   = np.array(intr.coeffs)

aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters  = aruco.DetectorParameters()
parameters.cornerRefinementMethod    = aruco.CORNER_REFINE_SUBPIX
parameters.adaptiveThreshWinSizeStep = 3
detector    = aruco.ArucoDetector(aruco_dict, parameters)
marker_length = 0.10

OBJ_POINTS = np.array([[-marker_length/2,  marker_length/2, 0],
                        [ marker_length/2,  marker_length/2, 0],
                        [ marker_length/2, -marker_length/2, 0],
                        [-marker_length/2, -marker_length/2, 0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# COM port auto-detection
# ---------------------------------------------------------------------------

def detect_esp_port():
    """
    Scan available serial ports and return the best candidate for an ESP32.

    Strategy (in order):
      1. Any port whose description or hardware ID contains a known ESP32/CH34x/CP210x
         USB-UART chip string.
      2. If exactly one port exists overall, offer it as a fallback.
      3. If multiple ports are found but none match, list them and let the user pick.

    Returns the selected port string (e.g. 'COM3' or '/dev/ttyUSB0'), or None.
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

    # No keyword match — fall back to showing all ports
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
# ESP write workflow
# ---------------------------------------------------------------------------

def run_write_workflow():
    """
    Select a stored calibration and flash it to a connected ESP32.

    Steps:
      1. Show stored calibrations and ask the user to pick one.
      2. Auto-detect (or manually select) the COM port.
      3. Connect via EspConfig, read the robot ID from the ESP.
      4. Confirm, then upload the 12-value transformation matrix.
    """
    print("\n=== WRITE CALIBRATION TO ESP ===")

    # --- Step 1: pick a calibration ---
    store = load_all_calibrations()
    valid = {k: v for k, v in store.items() if isinstance(v, dict) and "transformation_matrix" in v}

    if not valid:
        print("No valid calibrations found in calibration.json. Run the calibration workflow first.")
        return

    print("\nAvailable calibrations:")
    ids = list(valid.keys())
    for i, cid in enumerate(ids):
        desc = valid[cid].get("description", "")
        print(f"  [{i}] {cid:<20} {desc}")

    choice = input("\nSelect calibration index to write: ").strip()
    try:
        calib_id = ids[int(choice)]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    matrix = np.array(valid[calib_id]["transformation_matrix"])  # flat 12-element list
    print(f"\nSelected: '{calib_id}'")
    print(f"Matrix (flat, 12 values): {matrix}")

    # --- Step 2: COM port ---
    print("\nDetecting serial ports...")
    port = detect_esp_port()
    if not port:
        port = input("Enter COM port manually (e.g. COM3 or /dev/ttyUSB0): ").strip()
    if not port:
        print("No port specified. Aborting.")
        return

    # --- Step 3: connect and read robot ID ---
    print(f"\nConnecting to {port} ...")
    try:
        esp = EspConfig(port)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Reading Robot ID from ESP...")
    robot_id = esp.get_robot_id()
    if robot_id is None:
        print("  Could not read Robot ID (check firmware / baud rate).")
        answer = input("  Continue anyway? [y/n]: ").strip().lower()
        if answer != 'y':
            esp.close()
            return
    else:
        print(f"  Robot ID: {robot_id}")

    # --- Step 4: confirm and upload ---
    confirm = input(f"\nUpload calibration '{calib_id}' to ESP (Robot ID {robot_id})? [y/n]: ").strip().lower()
    if confirm != 'y':
        print("Upload cancelled.")
        esp.close()
        return

    success = esp.upload_matrix(matrix.tolist())
    if success:
        print(f"\n✓ Calibration '{calib_id}' successfully written to ESP (Robot ID {robot_id}).")
    else:
        print("\n✗ Upload failed. Check the serial output for details.")

    esp.close()


# ---------------------------------------------------------------------------
# Calibration routine — returns a 3x4 transformation matrix
# ---------------------------------------------------------------------------

def run_calibration_routine():
    """Interactively collect 4 point pairs and compute the transformation."""
    pts_camera, pts_robot = [], []
    smooth_buffer = deque(maxlen=10)

    print("\n--- PHASE 1: STABILIZED CAPTURE ---")
    print("Point the camera at the ArUco marker and press 'c' to capture each point.")

    while len(pts_camera) < 4:
        frames      = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame           = np.asanyarray(color_frame.get_data())
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if ids is not None:
            _, rvec, tvec = cv2.solvePnP(OBJ_POINTS, corners[0], camera_matrix, dist_coeffs)
            smooth_buffer.append(tvec.flatten())
            avg_xyz = np.mean(smooth_buffer, axis=0)

            aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
            cv2.putText(frame, f"Captured: {len(pts_camera)}/4  |  press 'c' to capture",
                        (10, 30), 1, 1.3, (0, 255, 0), 2)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and ids is not None:
            pts_camera.append(avg_xyz)
            print(f"  Captured Point {len(pts_camera)}: {avg_xyz}")

        if key == ord('q'):
            print("Calibration aborted.")
            cv2.destroyWindow("Calibration")
            return None

    cv2.destroyWindow("Calibration")

    # Phase 2: robot coordinates
    print("\n--- PHASE 2: ENTER ROBOT COORDINATES ---")
    for i in range(4):
        val = input(f"  Enter Robot (x,y,z) for Point {i+1}: ")
        pts_robot.append([float(x.strip()) for x in val.split(',')])

    # Phase 3: compute
    print("\n--- PHASE 3: COMPUTING TRANSFORMATION ---")
    ret, M_transform, _ = cv2.estimateAffine3D(np.array(pts_camera), np.array(pts_robot))

    if ret:
        print("Computed Transformation Matrix:\n", M_transform)
        return M_transform
    else:
        print("Calibration failed — could not estimate transformation.")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("\n=== Multi-Instance Calibration System ===")
    print(f"Calibration file: {os.path.abspath(CALIB_FILE)}\n")

    # Top-level workflow selection
    print("Select workflow:")
    print("  [c] Calibration  — capture ArUco points and save transformation")
    print("  [w] Write        — flash a stored calibration to an ESP32")
    print("  [q] Quit")
    workflow = input("\nChoice: ").strip().lower()

    if workflow == 'w':
        run_write_workflow()
        pipeline.stop()
        cv2.destroyAllWindows()
        return

    if workflow == 'q' or workflow not in ('c', 'w'):
        print("Exiting.")
        pipeline.stop()
        return

    # ------------------------------------------------------------------ #
    #  CALIBRATION WORKFLOW                                                 #
    # ------------------------------------------------------------------ #
    print("\nStored calibrations:")
    list_calibrations()

    print()
    calib_id = input("Enter calibration ID to load or create (e.g. 'robot_01'): ").strip()
    if not calib_id:
        print("No ID entered. Exiting.")
        pipeline.stop()
        return

    M_transform = load_calibration(calib_id)

    if M_transform is not None:
        print(f"\nCalibration '{calib_id}' found.")
        choice = input("  [u] Use it   [r] Re-calibrate   [d] Delete   : ").strip().lower()

        if choice == 'r':
            M_transform = None
        elif choice == 'd':
            delete_calibration(calib_id)
            pipeline.stop()
            cv2.destroyAllWindows()
            return

    if M_transform is None:
        M_transform = run_calibration_routine()
        if M_transform is None:
            pipeline.stop()
            cv2.destroyAllWindows()
            return
        description = input("Optional description for this calibration (press Enter to skip): ").strip()
        save_calibration(calib_id, M_transform, description)

    # Offer to immediately write to ESP after a fresh calibration
    write_now = input("\nWrite this calibration to an ESP now? [y/n]: ").strip().lower()
    if write_now == 'y':
        run_write_workflow()

    # --- PHASE 4: LIVE TRACKING ---
    print(f"\n--- PHASE 4: TRACKING ACTIVE  [ID: {calib_id}] ---")
    print("Press 'q' to quit.\n")

    try:
        while True:
            frames      = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame           = np.asanyarray(color_frame.get_data())
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            if ids is not None:
                _, rvec, tvec = cv2.solvePnP(OBJ_POINTS, corners[0], camera_matrix, dist_coeffs)

                p_robot    = M_transform @ np.append(tvec.flatten(), 1.0)
                xr, yr, zr = p_robot
                pan        = np.degrees(np.arctan2(yr, xr))
                tilt       = np.degrees(np.arctan2(zr, np.sqrt(xr**2 + yr**2)))

                cv2.putText(frame, f"[{calib_id}]  PAN: {pan:.1f}  TILT: {tilt:.1f}",
                            (10, 30), 1, 1.5, (0, 255, 0), 2)
                print(f"Robot Frame: {p_robot}    Pan: {pan:.2f}    Tilt: {tilt:.2f}", end='\r')

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()