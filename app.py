from flask import request, jsonify, Blueprint, render_template, Flask
import base64
from io import BytesIO
from PIL import Image
import torch
from ultralytics import YOLO
import easyocr
import firebase_admin
from firebase_admin import db
import time
import cv2
import os
import numpy as np

app = Flask(__name__)

# decode the base64 image sent from the frontend
def decode_image_from_base64(image_base64):
    image_data = base64.b64decode(image_base64.split(',')[1])
    image = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # convert pil image to opencv format

model = YOLO('../models/CustomLPR.pt')
reader = easyocr.Reader(['en'])

# initialize image count
image_count = 0

# directory to save ROI text
ROI_DIR = './static/roi/'
os.makedirs(ROI_DIR, exist_ok=True)

# Define a directory to save the images
IMAGE_DIR = './static/images/'
os.makedirs(IMAGE_DIR, exist_ok=True)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def save_image(image, filename):
    filepath = os.path.join(IMAGE_DIR, filename)
    cv2.imwrite(filepath, image)

def object_detection(image):
    global image_count # use the global variable for counting images
    last_crop_time = time.time()
    crop_interval = 2
    detected_plates = set()

    results = model(image) # run the yolo model for object detection

    # run the yolo model for object detection
    if results:
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Extract the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                plate_number = image[y1:y2, x1:x2]
                plate_hash = hash(plate_number.tobytes())

                if plate_hash not in detected_plates:
                    detected_plates.add(plate_hash)
                    processed_plate = preprocess_image(plate_number)

                    save_image(plate_number, f'original_image_{image_count}.jpg')
                    save_image(processed_plate, f'processed_image_{image_count}.jpg')
                    image_count += 1

                    extracted_text = reader.readtext(processed_plate)

                    if extracted_text:
                        texts = [item[1] for item in extracted_text]
                        text = " ".join(texts).replace(" ", "").upper()
                    else:
                        text = "No text detected"
                    
                    print("Extracted Text:", text)
                    return text # return the detected text

    return "No plate detected"

# api to process the frame and return the detected plate
@app.route('/process-frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    frame = data['frame']
    image = decode_image_from_base64(frame)

    # pass the image through the ALPR model
    plate_number = object_detection(image)

    # return the detected plate number as JSON
    return jsonify({'plate': plate_number})

# save detected to firebase
def save_plate_to_firebase(plate_number, camera_id):
    ref = db.reference('testing-webrtc')
    plate_data = {
        'plate_number': plate_number,
        'camera_id': camera_id,
        'timestamp': time.time()
    }
    ref.push(plate_data)

# load the test-webrtc html
@app.route('/webrtc-testing')
def webrtc_testing():
    return render_template('test-webrtc.html')

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
