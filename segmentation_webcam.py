import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # 1. Load the YOLO11 Segmentation model (Nano version)
    # This model is optimized for real-time performance
    model = YOLO("yolo11n-seg.pt")

    # 2. Initialize the standard webcam
    cap = cv2.VideoCapture(0)

    # Set resolution (optional, helps with performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting YOLO11 Segmentation... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 3. Perform Segmentation and Tracking
        # persist=True maintains ID across frames
        # classes=[0] filters for 'person' only
        results = model.track(frame, persist=True, classes=[0], verbose=False)

        # 4. Process Results
        for result in results:
            if result.masks is not None:
                # Create a transparent overlay for masks
                mask_overlay = frame.copy()
                
                # Iterate through detected objects
                for i, mask_coords in enumerate(result.masks.xy):
                    # Draw the Segmentation Mask
                    # mask_coords is a list of [x, y] points defining the silhouette
                    polygon = np.array(mask_coords, dtype=np.int32)
                    cv2.fillPoly(mask_overlay, [polygon], (0, 255, 0)) # Green mask
                    
                    # Get Bounding Box for Center Calculation
                    x1, y1, x2, y2 = map(int, result.boxes[i].xyxy[0])
                    
                    # 5. Calculate 2D Center Coordinates
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Draw Bounding Box and Center Point
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Display Coordinates
                    coord_text = f"Center ID {i}: ({cx}, {cy})"
                    cv2.putText(frame, coord_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Blend the original frame with the mask overlay
                frame = cv2.addWeighted(mask_overlay, 0.4, frame, 0.6, 0)

        # 6. Show the output
        cv2.imshow("YOLO11 Webcam Segmentation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()