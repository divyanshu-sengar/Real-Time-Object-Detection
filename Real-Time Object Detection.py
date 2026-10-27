import cv2
from ultralytics import YOLO

def main():
    # Load a pretrained general-object detection model
    model = YOLO("yolo26n.pt")

    # Open the default webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                print("Failed to read frame from camera.")
                break

            # Detect objects in the webcam frame
            results = model.predict(
                source=frame,
                imgsz=640,
                conf=0.35,
                verbose=False
            )

            # Draw detection boxes and labels
            annotated_frame = results[0].plot()

            # Show the detection window
            cv2.imshow("Real-Time Object Detection", annotated_frame)

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

