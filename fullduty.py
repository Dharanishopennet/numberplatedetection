import cv2
import torch
from ultralytics import YOLO

# Load YOLOv5 model for car detection
model = YOLO("yolov5s.pt")

# Load Haarcascade for number plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Open video file
video_path = "video5.mp4"  # Change this to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Video Properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}")

# Define codec and create VideoWriter object
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ensure frame size matches the video resolution
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Run YOLOv5 inference on the frame
    results = model(frame)

    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw rectangle and label for car detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Crop car region for number plate detection
            car_roi = frame[y1:y2, x1:x2]

            # Convert to grayscale for number plate detection
            gray_car = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)

            # Detect number plates inside car region
            plates = plate_cascade.detectMultiScale(gray_car, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw number plate detection
            for (px, py, pw, ph) in plates:
                cv2.rectangle(car_roi, (px, py), (px + pw, py + ph), (255, 0, 0), 2)
                cv2.putText(car_roi, "Plate", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Ensure full video display without cropping
    full_frame = cv2.resize(frame, (frame_width, frame_height))

    # **Fix display issue by changing window properties**
    cv2.namedWindow("YOLOv5 Object Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("YOLOv5 Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Show the frame
    cv2.imshow("YOLOv5 Object Detection", full_frame)

    # Write the frame to output video
    out.write(full_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
