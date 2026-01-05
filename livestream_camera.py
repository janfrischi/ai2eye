import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)

    # Align depth to color
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # ---- Create colored depth map ----
            # Convert depth to 8-bit for visualization
            depth_8u = cv2.convertScaleAbs(depth_image, alpha=0.03)

            # Apply colormap (JET, TURBO, INFERNO all work well)
            depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

            # ---- Optional: center distance overlay ----
            h, w = depth_image.shape
            cx, cy = w // 2, h // 2
            dist_m = depth_frame.get_distance(cx, cy)

            cv2.drawMarker(color_image, (cx, cy), (255, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(color_image, f"{dist_m:.3f} m",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

            # ---- Show windows ----
            cv2.imshow("RealSense Color", color_image)
            cv2.imshow("RealSense Depth (Colormap)", depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
