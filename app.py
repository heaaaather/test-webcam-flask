from flask import Flask, jsonify, Response
import cv2
import torch
import easyocr
import time
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load the YOLO model
model = YOLO('/models/CustomLPR.pt')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# To simulate continuous plate detection every 5 seconds
last_detected_plate = ""

# Process video frames and detect license plate every 5 seconds
def detect_license_plate():
    global last_detected_plate
    cap = cv2.VideoCapture(0)  # Using the first camera device

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame with YOLO
        results = model(frame)

        # Extract the bounding boxes
        detections = results.pandas().xyxy[0]

        # If a license plate is detected, run OCR
        for _, row in detections.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            plate_image = frame[ymin:ymax, xmin:xmax]

            # Use EasyOCR to recognize the text in the bounding box
            ocr_result = reader.readtext(plate_image)
            if ocr_result:
                last_detected_plate = ocr_result[0][-2]
        
        # Sleep for 5 seconds before processing the next frame
        time.sleep(5)

    cap.release()

# Route to get the latest plate number
@app.route('/get-plate-number', methods=['GET'])
def get_plate_number():
    return jsonify({'plate': last_detected_plate})

# Main function to start the video capture thread
if __name__ == '__main__':
    from threading import Thread
    # Start the license plate detection in a separate thread
    detection_thread = Thread(target=detect_license_plate)
    detection_thread.daemon = True
    detection_thread.start()

    # Start the Flask server
    app.run(debug=True, host='0.0.0.0', port=5000)
