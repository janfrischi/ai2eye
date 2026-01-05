import cv2
import time
import numpy as np
from ultralytics import YOLO

def main():
    # 1. Initialize Model
    model = YOLO("yolo11n-pose.pt") 

    # 2. Camera Setup
    cap = cv2.VideoCapture(0)
    
    # FPS Calculation Variables
    prev_time = 0
    new_time = 0

    print("Tracking Eyes and Nose... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # --- FPS Calculation ---
        new_time = time.time()
        # Ensure we don't divide by zero on the first frame
        if (new_time - prev_time) > 0:
            fps = 1 / (new_time - prev_time)
        else:
            fps = 0
        prev_time = new_time

        # 3. Pose Inference
        results = model.track(frame, persist=True, classes=[0], verbose=False)

        for result in results:
            if result.keypoints is not None:
                for person_kpts in result.keypoints.xy:
                    if len(person_kpts) < 5:
                        continue

                    # Point 0: Nose | Point 1: L Eye | Point 2: R Eye
                    nose = person_kpts[0]
                    l_eye = person_kpts[1]
                    r_eye = person_kpts[2]

                    # Draw Facial Features
                    for pt, label in zip([nose, l_eye, r_eye], ["Nose", "L-Eye", "R-Eye"]):
                        x, y = int(pt[0]), int(pt[1])
                        if x != 0 and y != 0:
                            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                            cv2.putText(frame, label, (x + 10, y - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 4. Display FPS on Frame
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (100, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("YOLO11 Pose - Eyes/Nose + FPS", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()