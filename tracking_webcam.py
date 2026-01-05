import cv2
from ultralytics import YOLO

def main():
    # 1. Load the YOLOv8 model (Nano version is fastest for real-time)
    model = YOLO('yolo11n.pt') 
    model = YOLO("yolo11n-seg.pt")

    # 2. Initialize the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 3. Perform tracking
        # persist=True ensures the model keeps the same ID for the face across frames
        results = model.track(frame, persist=True, classes=[0], verbose=False) # class 0 is 'person'

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 4. Calculate the 2D Center Coordinates
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw the center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Display the 2D coordinates
                coord_text = f"Center: ({center_x}, {center_y})"
                cv2.putText(frame, coord_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the output
        cv2.imshow("YOLOv8 Face Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()