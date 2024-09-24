from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import base64
import os
import cv2
import numpy as np

## flask-socketio = to add support for real-time communication between the client and server

app = Flask(__name__, static_folder="./templates/static")
app.config["SECRET_KEY"] = "secret!" # used to sign session cookies
socketio = SocketIO(app)

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, "static"),"favicon.ico",mimetype="image/vnd.microsoft.icon",)

# convert a base64-encoded image to a numpy array that can be processed using opencv
def base64_to_image(base64_string):
    # extract the base64 encoded binary data from the input string
    base64_data = base64_string.split("","")[1]

    # decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_data)

    # convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # decode the numpy array as an image using opencv
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image

# to handle incoming connections from the clients
@socketio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})

# to handle incoming images from the client
@socketio.on("image")
def receive_image(image):
    # decode the base64-encoded image data
    image = base64_to_image(image)

    # perform image processing using opencv
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resive(gray, (630, 360))

    # encode the processed image as a jpeg-encoded base64 string
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, frame_encoded = cv2.imencode(".jpg", frame_resized, encode_param)
    processed_img_data = base64.b64encode(frame_encoded).decode()

    # prepend the base64-encoded string with the data URL prefix
    b64_src = "data:image/jpg;base64,"
    processed_img_data = b64_src + processed_img_data

    # send the processed image back to the client
    emit("processed_image", processed_img_data)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')


