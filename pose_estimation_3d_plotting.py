import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO

# Transform 2D video into real-time 3D tracking system for human faces. 
def main():
    # 1. Initialize YOLO11-Pose, model identifies keypoints/skeleton joints
    model = YOLO("yolo11n-pose.pt") 

    # 2. Configure RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    width, height = 640, 480
    # Start two simultaneous streams: Color and depth
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    profile = pipeline.start(config)

    # Get Intrinsics for 3D Deprojection
    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    align = rs.align(rs.stream.color)
    spatial = rs.spatial_filter()
    hole_filling = rs.hole_filling_filter(1)

    # --- PLOT SETUP (Points only) ---
    plt.ion()
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    prev_time = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            raw_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not raw_depth_frame or not color_frame:
                continue

            # Process depth
            filtered = hole_filling.process(spatial.process(raw_depth_frame))
            depth_frame = filtered.as_depth_frame()
            color_image = np.asanyarray(color_frame.get_data())

            # --- YOLO Inference ---
            results = model.track(color_image, persist=True, classes=[0], verbose=False)

            # Clear 3D plot for new frame
            ax.cla()
            ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0, 2.0)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title("Real-Time 3D Points")

            for result in results:
                if result.keypoints is not None:
                    # Keypoints.xy is (N, 17, 2)
                    for person_kpts in result.keypoints.xy:
                        if len(person_kpts) < 5: continue

                        # 0: Nose, 1: L-Eye, 2: R-Eye
                        facial_points = {
                            "Nose": (person_kpts[0], (0, 0, 255)),   # Red
                            "L-Eye": (person_kpts[1], (255, 0, 0)),  # Blue
                            "R-Eye": (person_kpts[2], (0, 255, 0))   # Green
                        }

                        for label, (pt, color_bgr) in facial_points.items():
                            ix, iy = int(pt[0]), int(pt[1])
                            
                            if 0 < ix < width and 0 < iy < height:
                                z = depth_frame.get_distance(ix, iy)
                                if z > 0:
                                    # 1. Calculate 3D World Coordinates
                                    p3d = rs.rs2_deproject_pixel_to_point(intrinsics, [ix, iy], z)
                                    
                                    # 2. Update 3D Plot (Points only, no traces)
                                    # Convert BGR to RGB for Matplotlib
                                    color_rgb = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)
                                    ax.scatter(p3d[0], p3d[1], p3d[2], color=color_rgb, s=50)

                                    # 3. Draw on 2D Image Frame
                                    cv2.circle(color_image, (ix, iy), 6, color_bgr, -1)
                                    cv2.putText(color_image, f"{label}: {z:.2f}m", (ix + 10, iy - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

            # Refresh Plot
            plt.pause(0.001)

            # Calculate and show FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(color_image, f"FPS: {int(fps)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("RealSense YOLO11-Pose", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        plt.close()

if __name__ == "__main__":
    main()