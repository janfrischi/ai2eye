import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json
import os
import serial  # Added for ESP32 communication
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

def main():
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

    spatial = rs.spatial_filter()
    hole_filling = rs.hole_filling_filter()

    prev_time = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame: continue

            depth_frame = spatial.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame).as_depth_frame()
            color_image = np.asanyarray(color_frame.get_data())
            
            results = model.track(color_image, persist=True, classes=[0], verbose=False)

            for result in results:
                if result.keypoints is not None:
                    for person_kpts in result.keypoints.xy:
                        if len(person_kpts) < 1: continue

                        nose_kpt = person_kpts[0]
                        ix, iy = int(nose_kpt[0]), int(nose_kpt[1])
                        
                        if 0 < ix < width and 0 < iy < height:
                            depth_z = depth_frame.get_distance(ix, iy)
                            
                            if depth_z > 0:
                                p_cam = rs.rs2_deproject_pixel_to_point(intrinsics, [ix, iy], depth_z)
                                p_robot = M_transform @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
                                rx, ry, rz = p_robot

                                yaw = np.degrees(np.arctan2(ry, rx))
                                horizontal_dist = np.sqrt(rx**2 + ry**2)
                                tilt = np.degrees(np.arctan2(rz, horizontal_dist))

                                # --- SEND TO ESP32 ---
                                if ser and ser.is_open:
                                    data_packet = f"{yaw:.2f},{tilt:.2f}\n"
                                    print(f"Sending Yaw and Tilt Angles: {data_packet.strip()}")
                                    ser.write(data_packet.encode('utf-8'))

                                # Visuals
                                cv2.circle(color_image, (ix, iy), 5, (0, 0, 255), -1)
                                cv2.putText(color_image, f"Y:{yaw:.1f} T:{tilt:.1f}", (ix + 10, iy - 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("Robot Frame Tracking", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if ser: ser.close()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()