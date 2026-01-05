import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO

def main():
    # 1. Initialize YOLO11-Pose
    model = YOLO("yolo11n-pose.pt") 

    # 2. Configure RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    width, height = 640, 480
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    pipeline.start(config)

    # 3. Setup Alignment and Filters
    align = rs.align(rs.stream.color)
    spatial = rs.spatial_filter()
    hole_filling = rs.hole_filling_filter(1)

    prev_time = 0

    print(f"{'Point':<10} | {'X':<5} | {'Y':<5} | {'Depth (m)':<10}")
    print("-" * 40)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            raw_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not raw_depth_frame or not color_frame:
                continue

            # Apply filters and cast to depth_frame
            filtered = spatial.process(raw_depth_frame)
            filtered = hole_filling.process(filtered)
            depth_frame = filtered.as_depth_frame()

            color_image = np.asanyarray(color_frame.get_data())
            
            # --- Tracking ---
            results = model.track(color_image, persist=True, classes=[0], verbose=False)

            for result in results:
                if result.keypoints is not None:
                    # Keypoints.xy is (N, 17, 2)
                    for person_idx, person_kpts in enumerate(result.keypoints.xy):
                        if len(person_kpts) < 5: continue

                        # Nose=0, L Eye=1, R Eye=2
                        facial_points = {
                            "Nose": person_kpts[0],
                            "L-Eye": person_kpts[1],
                            "R-Eye": person_kpts[2]
                        }

                        for label, pt in facial_points.items():
                            ix, iy = int(pt[0]), int(pt[1])
                            
                            # Bounds check
                            if 0 < ix < width and 0 < iy < height:
                                dist = depth_frame.get_distance(ix, iy)
                                
                                # --- PRINTING VALUES ---
                                # We only print if depth is valid (> 0)
                                if dist > 0:
                                    print(f"{label:<10} | {ix:<5} | {iy:<5} | {dist:.3f}m")

                                # Visuals on frame
                                cv2.circle(color_image, (ix, iy), 4, (0, 255, 255), -1)
                                
                                # Overlay text: (X, Y) and Depth
                                data_str = f"{ix},{iy} | {dist:.2f}m"
                                cv2.putText(color_image, data_str, (ix + 10, iy - 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            # FPS Display
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(color_image, f"FPS: {int(fps)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("RealSense Tracking Data", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()