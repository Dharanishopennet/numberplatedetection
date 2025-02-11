import cv2
import torch
import numpy as np
import pytesseract
from ultralytics import YOLO

# Load YOLOv5 model for car detection
model = YOLO("yolov5s.pt")

# Load Haarcascade for number plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Set OCR path (Change this based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\vkdha\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

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

# Set full-screen mode
cv2.namedWindow("Number Plate Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Number Plate Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# Function to preprocess number plate for better OCR
def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Resize for clarity
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)  # Adjust contrast
    return processed


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
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Crop car region for number plate detection
            car_roi = frame[y1:y2, x1:x2]

            # Convert to grayscale for number plate detection
            gray_car = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)

            # Detect number plates inside car region
            plates = plate_cascade.detectMultiScale(gray_car, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (px, py, pw, ph) in plates:
                plate_img = car_roi[py:py + ph, px:px + pw]

                # Preprocess plate image for better OCR
                processed_plate = preprocess_plate(plate_img)

                # Convert number plate to text using OCR (with high accuracy settings)
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = pytesseract.image_to_string(processed_plate, config=custom_config).strip()

                # Draw number plate rectangle and OCR text
                cv2.rectangle(car_roi, (px, py), (px + pw, py + ph), (255, 0, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                print(f"Detected Number Plate: {text}")  # Print detected number plate

    # Display in full-screen mode
    cv2.imshow("Number Plate Detection", frame)
    out.write(frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
