import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json
import os
from ultralytics import YOLO

# The final pipeline uses the calibration matrix from "calibration.json" and tracks nose of person, depicts coordinates in the camera frame
# and directly calculates the tilt and yaw angles for the robotic platform
CALIB_FILE = "calibration.json"

def load_calibration():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, 'r') as f:
            data = json.load(f)
            return np.array(data["transformation_matrix"])
    return None

def main():
    # 1. Initialization
    # Initialize YOLO11-pose model
    model = YOLO("yolo11n-pose.pt")
    # Load calibration matrix from JSON 
    M_transform = load_calibration()
    
    if M_transform is None:
        print("Error: calibration.json not found. Please run the calibration script first.")
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
    spatial = rs.spatial_filter()
    hole_filling = rs.hole_filling_filter()

    prev_time = 0

    # Console Header including Robot Z
    print(f"\n{'Target':<10} | {'Rob X':<7} | {'Rob Y':<7} | {'Rob Z':<7} | {'Yaw (deg)':<10} | {'Tilt (deg)':<10}")
    print("-" * 75)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame: continue

            # Filter depth
            depth_frame = spatial.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame).as_depth_frame()

            color_image = np.asanyarray(color_frame.get_data())
            
            # YOLO Inference
            results = model.track(color_image, persist=True, classes=[0], verbose=False)

            for result in results:
                if result.keypoints is not None:
                    for person_kpts in result.keypoints.xy:
                        if len(person_kpts) < 1: continue

                        # Focus on the Nose (Index 0)
                        nose_kpt = person_kpts[0]
                        ix, iy = int(nose_kpt[0]), int(nose_kpt[1])
                        
                        if 0 < ix < width and 0 < iy < height:
                            depth_z = depth_frame.get_distance(ix, iy)
                            
                            if depth_z > 0:
                                # Get Camera Frame Point
                                p_cam = rs.rs2_deproject_pixel_to_point(intrinsics, [ix, iy], depth_z)
                                
                                # Transform to Robot Frame: M * [x, y, z, 1]
                                p_robot = M_transform @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
                                rx, ry, rz = p_robot

                                # Calculate Angles
                                yaw = np.degrees(np.arctan2(ry, rx))
                                horizontal_dist = np.sqrt(rx**2 + ry**2)
                                tilt = np.degrees(np.arctan2(rz, horizontal_dist))

                                # Terminal Output including rz
                                print(f"{'Nose':<10} | {rx:>7.2f} | {ry:>7.2f} | {rz:>7.2f} | {yaw:>10.2f} | {tilt:>10.2f}", end='\r')

                                # Visuals (Nose point and Yaw/Tilt only)
                                cv2.circle(color_image, (ix, iy), 5, (0, 0, 255), -1)
                                angle_str = f"Yaw:{yaw:.1f} Tilt:{tilt:.1f}"
                                cv2.putText(color_image, angle_str, (ix + 10, iy - 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Performance UI (FPS)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(color_image, f"FPS: {int(fps)}", (20, 40), 1, 1.5, (100, 255, 0), 2)

            cv2.imshow("Robot Frame Tracking", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()