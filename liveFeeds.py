import cv2
import os
import time
import numpy as np
import collections
import requests
import mediapipe as mp
from datetime import datetime
import face_recognition
from flask import Flask, Response, request, jsonify, render_template
from modelSchema import SettingsSchema
from model import Settings
from database import db

# RTSP Stream URL
# RTSP_URL = "rtsp://admin:admin$123@122.160.111.249:554/cam/realmonitor?channel=11&subtype=1"
# RTSP_URL = "/Users/mdarshadali/Downloads/manwithlaptop.mp4"

# Background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# YOLO (Object & Person Detection)
net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
fire_cascade = cv2.CascadeClassifier("fire_detection_cascade_model.xml")

# Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Tracking movement for stopped persons
person_states = collections.defaultdict(lambda: {"positions": [], "last_moved": None, "state": "walking"})

# Store last detected faces
last_saved_faces = []

# yolo setup for electronics detection
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]



# 🔥 Send Notification Function
def send_notification(api_url, api_key, event_name, message):
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"event_name": event_name, "message": message}
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            print(f"Notification sent: {event_name} - {message}")
    except Exception as e:
        print(f"Error sending notification: {e}")


# 📡 Connect to RTSP Stream
def connect_to_stream(RTSP_URL):
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap


# 🎥 Live Stream with Detection Functions
# def generate_frames():
#     cap = connect_to_stream()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Reconnecting to RTSP stream...")
#             cap.release()
#             time.sleep(5)
#             cap = connect_to_stream()
#             continue

#         # Apply detection algorithms
#         frame = process_motion(frame)
#         frame = detect_person_motion(frame)
#         frame = detect_faces(frame)
#         frame = detect_fire(frame)
#         # frame = detect_pose(frame)
#         frame = detect_electronic_devices(frame)
#         # frame = detect_stopped_after_walking(frame)

#         # Encode frame for streaming
#         ret, buffer = cv2.imencode('.jpg', frame)
#         if not ret:
#             continue

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

#     cap.release()



def generate_frames(rtsp_url, settings):
    cap = connect_to_stream(rtsp_url)
    fps = 30  # Target FPS
    frame_time = 1.0 / fps  # Time per frame
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Reconnecting to RTSP stream...")
            cap.release()
            time.sleep(1)
            cap = connect_to_stream(rtsp_url)
            continue

        # Apply detection algorithms (keeping it lightweight for performance)

        if settings.normal_motion:
            frame = process_motion(frame)
        if settings.person_motion:
            frame = detect_person_motion(frame)
        if settings.faces:
            frame = detect_faces(frame)
        if settings.fire_detections:
            frame = detect_fire(frame)
        if settings.electronic_devices:
            frame = detect_electronic_devices(frame)
        if settings.pose:
            frame = detect_pose(frame)
        if settings.stopped_persons:
            frame = detect_stopped_after_walking(frame, people_boxes, orig_fps)

        # frame = detect_person_motion(frame)
        # frame = detect_faces(frame)
        # frame = detect_fire(frame)
        # frame = detect_electronic_devices(frame)
        # frame = detect_pose(frame)
        # frame = detect_stopped_after_walking(frame)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        # **Ensure 30 FPS by adjusting sleep time**
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed_time)
        time.sleep(sleep_time)  # Sleep to maintain 30 FPS

    cap.release()
# 🚶 Normal Motion Detection
def process_motion(frame):
    fg_mask = fgbg.apply(frame)
    if np.sum(fg_mask > 0) > 5000:
        cv2.putText(frame, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


# 🏃 Person Motion Detection
def detect_person_motion(frame):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for detection in detections:
        for obj in detection:
            confidence = obj[5:]
            class_id = np.argmax(confidence)
            if confidence[class_id] > 0.5 and class_id == 0:
                cv2.putText(frame, "Person Moving", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame


# 🧑 Face Detection
def detect_faces(frame):
    """
    Detect faces in the frame, draw bounding boxes, and return the updated frame.
    
    Args:
        frame (numpy.ndarray): The video frame.

    Returns:
        numpy.ndarray: The frame with detected face annotations.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for face_recognition
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Detect faces

    for (top, right, bottom, left) in face_locations:
        # Draw bounding box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Display label
        cv2.putText(frame, "Face Detected", (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame  # Return the frame with detections


# 🔥 Fire Detection
def detect_fire(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fire_regions = fire_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in fire_regions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Fire Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return frame


# 📱 Electronic Device Detection
# def detect_electronic_devices(frame):
#     height, width, _ = frame.shape
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)

#     electronic_classes = ["tv", "laptop", "cell phone", "keyboard", "mouse", "remote", "book"]

#     try:
#         outputs = net.forward(output_layers)
#         for output in outputs:
#             for detection in output:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]

#                 if confidence > 0.5 and classes[class_id] in electronic_classes:
#                     x, y, w, h = (int(detection[0] * width), int(detection[1] * height),
#                                   int(detection[2] * width), int(detection[3] * height))
#                     x, y = max(0, x - w // 2), max(0, y - h // 2)
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
#                     cv2.putText(frame, f"{classes[class_id]} {confidence:.2f}",
#                                 (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
#         return frame
#     except Exception as e:
#         print(f"Error in electronic detection: {e}")
#         return frame
def detect_electronic_devices(frame):
    """
    Optimized function to detect electronic devices using YOLO.
    
    Args:
        frame (numpy.ndarray): Input video frame.
    
    Returns:
        numpy.ndarray: Frame with detected electronic devices highlighted.
    """
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Set of electronic classes for fast lookup
    electronic_classes_set = {"tv", "laptop", "cell phone", "keyboard", "mouse", "remote", "book"}

    try:
        outputs = net.forward(output_layers)
        boxes, confidences, class_ids = [], [], []

        # Process detections efficiently
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] in electronic_classes_set:
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = center_x - w // 2, center_y - h // 2

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maxima Suppression to remove duplicate boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Draw only the best detections
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    except Exception as e:
        print(f"Error in electronic detection: {e}")

    return frame

# 🚷 Stopped Person Detection
def detect_stopped_after_walking(frame, people_boxes, fps, stop_duration=2, movement_tolerance=10):
    global person_states
    stop_frames = int(stop_duration * fps)  # Convert seconds to frames

    stopped_positions = []
    for i, (x, y, w, h) in enumerate(people_boxes):
        person_id = f"person_{i}"
        current_position = (x, y)

        # Store position history
        person_states[person_id]["positions"].append(current_position)
        if len(person_states[person_id]["positions"]) > stop_frames:
            person_states[person_id]["positions"].pop(0)

        # Check if person has moved significantly
        if len(person_states[person_id]["positions"]) > 1:
            movements = [
                abs(person_states[person_id]["positions"][j + 1][0] - person_states[person_id]["positions"][j][0]) +
                abs(person_states[person_id]["positions"][j + 1][1] - person_states[person_id]["positions"][j][1])
                for j in range(len(person_states[person_id]["positions"]) - 1)
            ]

            # If movement is below threshold for the given duration, mark as stopped
            if movements and max(movements) < movement_tolerance:
                if person_states[person_id]["state"] == "walking":
                    if person_states[person_id]["last_moved"] is None:
                        person_states[person_id]["last_moved"] = time.time()
                    elif time.time() - person_states[person_id]["last_moved"] >= stop_duration:
                        person_states[person_id]["state"] = "stopped"
                        stopped_positions.append((x, y, w, h))
            else:
                # Reset if movement resumes
                person_states[person_id]["state"] = "walking"
                person_states[person_id]["last_moved"] = None

    # Draw bounding boxes for stopped persons
    for (x, y, w, h) in stopped_positions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(frame, "Stopped Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Pose Detection Function
def detect_pose(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        try:
            # Extract key landmarks
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

            # Calculate angles
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

            # Define angle ranges
            standing_range = (160, 200)
            sitting_range = (65, 115)

            # Check if standing
            is_standing = (
                standing_range[0] <= right_hip_angle <= standing_range[1] and
                standing_range[0] <= left_hip_angle <= standing_range[1] and
                standing_range[0] <= right_knee_angle <= standing_range[1] and
                standing_range[0] <= left_knee_angle <= standing_range[1]
            )

            # Check if sitting
            is_sitting = (
                sitting_range[0] <= right_hip_angle <= sitting_range[1] and
                sitting_range[0] <= left_hip_angle <= sitting_range[1] and
                sitting_range[0] <= right_knee_angle <= sitting_range[1] and
                sitting_range[0] <= left_knee_angle <= sitting_range[1]
            )

            # Define bounding box coordinates
            x_min = int(min(right_shoulder.x, left_shoulder.x, right_hip.x, left_hip.x) * frame.shape[1])
            y_min = int(min(right_shoulder.y, left_shoulder.y, right_hip.y, left_hip.y) * frame.shape[0])
            x_max = int(max(right_shoulder.x, left_shoulder.x, right_hip.x, left_hip.x) * frame.shape[1])
            y_max = int(max(right_shoulder.y, left_shoulder.y, right_hip.y, left_hip.y) * frame.shape[0])

            # Draw bounding box and label
            if is_standing:
                label = "Standing"
                color = (0, 255, 0)  # Green
            elif is_sitting:
                label = "Sitting"
                color = (0, 255, 255)  # Yellow
            else:
                label = "Unknown Pose"
                color = (0, 0, 255)  # Red

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        except Exception as e:
            print(f"Pose detection error: {e}")

    return frame

# 🎥 Flask Routes
def video_feed(id):
    # Fetch the settings from the database
    settings = Settings.query.get(id)

    if not settings:
        return jsonify({"status": "error", "message": f"Settings with id {id} not found"}), 404

    # Extract the RTSP URL
    rtsp_url = settings.rtspUrl

    # Validate the RTSP URL
    if not rtsp_url:
        return jsonify({"status": "error", "message": "RTSP URL is missing in settings"}), 400

    return Response(generate_frames(rtsp_url, settings), mimetype='multipart/x-mixed-replace; boundary=frame')


def index(id):
    return render_template("index.html", id=id)

