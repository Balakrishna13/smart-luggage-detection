# Smart Luggage Detection System

This is a small project I built to detect luggage in videos or live camera feeds. It can identify various types of luggage, estimate their distance from the camera, and even read QR codes if present. I added voice feedback so the system announces what it sees in real time, continuously telling the type of luggage and its approximate distance until it moves out of view. The audio functionality is specifically designed to assist visually-impaired users. For example, they can place QR codes on their luggage, and the system will detect them and read the information out loud, making it easier to identify and grab the right bag.

At the same time, the system highlights each detected luggage with a green bounding box and displays the label and distance on screen, providing a visual cue. The system processes the video frame by frame, making it easy to follow what it is detecting at any moment.

---

## Features

- Detects different types of luggage  
- Estimates approximate distance in centimeters  
- Draws green bounding boxes and labels on detected objects  
- Detects QR codes on luggage and reads the info out loud  
- Provides continuous audio feedback using `pyttsx3`  
- Works with both a video file or directly from your webcam  
- Saves the processed output video for later viewing  

---

## Requirements

| Library          | Version       |
|------------------|---------------|
| Python           | 3.10+         |
| OpenCV           | Latest        |
| pyttsx3          | Latest        |
| Ultralytics YOLO | Latest        |
| NumPy            | Latest        |
| qrcode           | Latest        |

---

## Installation

Make sure you have **Python 3.10+** installed.  

Install the required libraries:

```bash
pip install opencv-python pyttsx3 ultralytics numpy qrcode
```

Download the YOLOv8 model (`yolov8n.pt`) from [Ultralytics YOLO](https://ultralytics.com/).

---

### Usage

1. Place your input video in the project folder (or use your webcam).
2. Update the `VIDEO_SOURCE` variable in the script if needed:

```python
VIDEO_SOURCE = "your_video.mp4"  # or 0 for webcam
```

3. Run the script:

```bash
python smart_luggage_detection.py
```

The system will process the video frame by frame:

* Highlight detected luggage with a green box
* Display the type and estimated distance on screen
* Announce the detected luggage and QR info out loud

The processed video will be saved as `output.mp4` in the same folder.

---

## Notes

* The system continuously announces the luggage and distance, so it works best with visually-impaired users.
* QR codes can be placed on luggage to provide additional identification.
* You can adjust the distance estimation parameters (`KNOWN_WIDTH` and `FOCAL_LENGTH`) for your camera setup.
* Press **q** to quit the live preview.
