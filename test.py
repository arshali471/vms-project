import cv2
import time
from flask import Flask, Response

app = Flask(__name__)

# RTSP_URL = "rtsp://admin:admin$123@122.160.111.249:554/cam/realmonitor?channel=5&subtype=1"
RTSP_URL = "rtsp://admin:admin$123@122.160.111.249:554/cam/realmonitor?channel=11&subtype=1"

def connect_to_stream():
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer size to avoid delays
    return cap

def generate_frames():
    cap = connect_to_stream()
    while True:
        if not cap.isOpened():
            print("Reconnecting to the RTSP stream...")
            cap.release()
            time.sleep(5)  # Wait before reconnecting
            cap = connect_to_stream()
            continue

        success, frame = cap.read()
        if not success:
            print("Failed to read frame. Reconnecting...")
            cap.release()
            time.sleep(5)
            cap = connect_to_stream()
            continue

        # Optional: Process the frame here

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>RTSP Stream</title>
        </head>
        <body>
            <h1>RTSP Live Stream</h1>
            <img src="/video_feed" width="720" />
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
