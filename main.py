import cv2
import pyttsx3
import time
from ultralytics import YOLO

# ---------------- CONFIG ---------------- #
KNOWN_WIDTH = 30.0      # cm (average bag width)
FOCAL_LENGTH = 800      # adjust based on calibration
CONF_THRESHOLD = 0.4
VIDEO_SOURCE = "input_video.mp4"  # change to 0 for webcam
TARGET_OBJECTS = ["suitcase", "backpack", "handbag"]
# ---------------------------------------- #

# Text-to-Speech setup
tts_engine = pyttsx3.init()
last_spoken_time = 0
SPEAK_COOLDOWN = 3  # seconds


def speak(text):
    global last_spoken_time
    current_time = time.time()
    if current_time - last_spoken_time > SPEAK_COOLDOWN:
        tts_engine.say(text)
        tts_engine.runAndWait()
        last_spoken_time = current_time


def estimate_distance(pixel_width):
    if pixel_width == 0:
        return -1
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width


# Load YOLO model
model = YOLO("yolov8n.pt")

# QR detector
qr_detector = cv2.QRCodeDetector()

# Video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (w, h))

    results = model(frame)[0]
    qr_display_text = ""

    for box in results.boxes:
        cls_id = int(box.cls)
        label = model.names[cls_id]
        conf = float(box.conf)

        if label in TARGET_OBJECTS and conf > CONF_THRESHOLD:
            coords = box.xyxy.cpu().numpy()
            if coords.ndim > 1:
                coords = coords[0]
            x1, y1, x2, y2 = map(int, coords[:4])
            pixel_width = x2 - x1

            distance = estimate_distance(pixel_width)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} | {int(distance)} cm"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            speak(f"{label} detected at approximately {int(distance)} centimeters")

            # QR detection inside bounding box
            roi = frame[y1:y2, x1:x2]
            qr_data, _, _ = qr_detector.detectAndDecode(roi)

            if qr_data:
                qr_display_text = qr_data
                speak(f"QR Code Identified {qr_data}")
                cv2.putText(frame, f"QR: {qr_data}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if qr_display_text:
        cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"QR Info: {qr_display_text}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Smart Luggage Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()