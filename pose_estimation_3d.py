import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO

# Transform 2D video feed into real-time 3D tracking system
def main():
    model = YOLO("yolo11n-pose.pt") 

    pipeline = rs.pipeline()
    config = rs.config()
    width, height = 640, 480
    # Start two simultaneous streams: Color and depth
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    
    # Start pipeline and get the active profile
    profile = pipeline.start(config)

    # Get camera intrinsics
    # Since we align depth-to-color, we use the color intrinsics for deprojection
    color_profile = profile.get_stream(rs.stream.color)
    # Intrinsics contain focal length (fx,fy) and principal point (px,py)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    # Align the color image with the depth map
    align = rs.align(rs.stream.color)
    # Smoothen depth data to reduce noise
    spatial = rs.spatial_filter()
    # Hole filling: Intelligently guess the depth of black pixels cause by shadows
    hole_filling = rs.hole_filling_filter()

    prev_time = 0

    print(f"{'Point':<10} | {'X (m)':<8} | {'Y (m)':<8} | {'Z (m)':<8}")
    print("-" * 45)

    # YOLO Inference
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            raw_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Filtering and casting
            filtered = spatial.process(raw_depth_frame)
            filtered = hole_filling.process(filtered)
            depth_frame = filtered.as_depth_frame()

            color_image = np.asanyarray(color_frame.get_data())
            
            # Run YOLO Pose detection, persist=True allows model to remember unique ID's
            results = model.track(color_image, persist=True, classes=[0], verbose=False)

            # Iterate over all the detections
            for result in results:
                if result.keypoints is not None:
                    for person_kpts in result.keypoints.xy:
                        if len(person_kpts) < 5: continue

                        facial_points = {"Nose": person_kpts[0], "L-Eye": person_kpts[1], "R-Eye": person_kpts[2]}
                        for label, pt in facial_points.items():
                            # Get pixel coordinates of facial point
                            ix, iy = int(pt[0]), int(pt[1])
                            
                            if 0 < ix < width and 0 < iy < height:
                                # Get depth at pixel coordinate (u,v)
                                depth_z = depth_frame.get_distance(ix, iy)
                                
                                # Convert 2D pixel coord (u,v) to 3D point in space (X,Y,Z)
                                if depth_z > 0:
                                    # rs2_deproject_pixel_to_point returns [X, Y, Z] in meters
                                    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [ix, iy], depth_z)
                                    # Get world coordinates of the tracked point
                                    world_x, world_y, world_z = point_3d

                                    # Console Print
                                    print(f"{label:<10} | {world_x:>8.3f} | {world_y:>8.3f} | {world_z:>8.3f}")

                                    # Visuals - Add circular markers at keypoint locations
                                    cv2.circle(color_image, (ix, iy), 3, (0, 255, 0), -1)
                                    coord_str = f"X:{world_x:.2f} Y:{world_y:.2f} Z:{world_z:.2f}"
                                    cv2.putText(color_image, coord_str, (ix + 10, iy - 5), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(color_image, f"FPS: {int(fps)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 0), 2)

            cv2.imshow("3D World Coordinate Tracking", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()