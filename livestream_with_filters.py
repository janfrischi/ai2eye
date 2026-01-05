import pyrealsense2 as rs
import numpy as np
import cv2

def nothing(x):
    pass

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    width, height = 640, 480
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    pipeline.start(config)

    # Initialize Filters
    decimation = rs.decimation_filter()
    threshold = rs.threshold_filter()
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    
    colorizer = rs.colorizer()
    align = rs.align(rs.stream.color)

    # --- UI Formatting ---
    ctrl_win = "Depth Control Panel"
    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ctrl_win, 400, 300)

    # Trackbars with cleaner naming and logical ranges
    cv2.createTrackbar("Alpha (0-100)", ctrl_win, 50, 100, nothing) 
    cv2.createTrackbar("Magnitude (1-5)", ctrl_win, 2, 5, nothing)
    cv2.createTrackbar("Hole Fill (0-2)", ctrl_win, 1, 2, nothing)
    cv2.createTrackbar("Threshold (m)", ctrl_win, 3, 10, nothing)
    cv2.createTrackbar("Reset", ctrl_win, 0, 1, nothing)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Reset logic
            if cv2.getTrackbarPos("Reset", ctrl_win) == 1:
                cv2.setTrackbarPos("Alpha (0-100)", ctrl_win, 50)
                cv2.setTrackbarPos("Magnitude (1-5)", ctrl_win, 2)
                cv2.setTrackbarPos("Hole Fill (0-2)", ctrl_win, 1)
                cv2.setTrackbarPos("Reset", ctrl_win, 0)

            # Get trackbar values
            s_alpha = cv2.getTrackbarPos("Alpha (0-100)", ctrl_win) / 100.0
            s_mag = cv2.getTrackbarPos("Magnitude (1-5)", ctrl_win)
            h_mode = cv2.getTrackbarPos("Hole Fill (0-2)", ctrl_win)
            t_dist = cv2.getTrackbarPos("Threshold (m)", ctrl_win)

            # Update Filter Parameters
            spatial.set_option(rs.option.filter_smooth_alpha, max(0.25, s_alpha))
            spatial.set_option(rs.option.filter_magnitude, float(max(1, s_mag)))
            hole_filling.set_option(rs.option.holes_fill, float(h_mode))
            threshold.set_option(rs.option.max_distance, float(t_dist))

            # --- Processing ---
            processed = decimation.process(depth_frame)
            processed = threshold.process(processed)
            processed = spatial.process(processed)
            processed = temporal.process(processed)
            processed = hole_filling.process(processed)

            # --- Visuals ---
            depth_colorized = np.asanyarray(colorizer.colorize(processed).get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_resized = cv2.resize(depth_colorized, (width, height), interpolation=cv2.INTER_NEAREST)

            # Create a black bar at the bottom for status text (cleaner than overlaying on image)
            status_bar = np.zeros((40, width * 2, 3), dtype=np.uint8)
            info_text = f"Alpha: {s_alpha:.2f} | Mag: {s_mag} | Hole Mode: {h_mode} | Range: {t_dist}m"
            cv2.putText(status_bar, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Combine Main Output
            main_output = np.hstack((color_image, depth_resized))
            final_display = np.vstack((main_output, status_bar))

            cv2.imshow("RealSense Filter Tuning", final_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()