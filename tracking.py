import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

def main():
    # 1. Initialize YOLOv8
    #model = YOLO('yolov8n.pt') # Using Nano for speed
    # 1.1 Initialize YOLO11n
    model = YOLO("yolo11n.pt")

    # 2. Configure RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    width, height = 640, 480
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    pipeline.start(config)

    # 3. Initialize Filters (Using the tuned settings from before)
    align = rs.align(rs.stream.color)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    hole_filling = rs.hole_filling_filter(1) # Mode 1: Farthest from neighbor

    try:
        while True:
            # Get frames and align
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Apply Post-Processing to Depth (to ensure accurate center-point reading)
            depth_frame = spatial.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            
            # 4. YOLO Tracking
            # We track "person" (class 0)
            results = model.track(color_image, persist=True, classes=[0], verbose=False)

            if results[0].boxes:
                for box in results[0].boxes:
                    # Get 2D Bounding Box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate 2D Center
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # 5. Get 3D Depth at the Center Pixel
                    # Ensure coordinates are within image bounds
                    if 0 <= cx < width and 0 <= cy < height:
                        distance = depth_frame.get_distance(cx, cy)
                    else:
                        distance = 0

                    # Draw Visuals
                    color = (0, 255, 0)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                    
                    label = f"Person: {distance:.2f}m"
                    cv2.putText(color_image, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Show results
            cv2.imshow("RealSense + YOLO Tracking", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()