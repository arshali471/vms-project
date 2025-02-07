from flask import request, jsonify, make_response, send_from_directory
from modelSchema import SettingsSchema
from model import Settings
from database import db
import cv2
import os
import numpy as np
import collections
import mediapipe as mp
from datetime import datetime
import time
import face_recognition
import requests
import torch
from torchvision import transforms
from PIL import Image
import warnings




# Stopped-walking detection variables
stopped_person_folder = "stopped_person"
os.makedirs(stopped_person_folder, exist_ok=True)

# Data structure to track movement and state
person_states = collections.defaultdict(lambda: {"positions": [], "last_moved": None, "state": "walking"})



# Directories for output
output_dirs = ["normal_motion", "person_motion", "faces", "high_person_count", "pose", "fire_detections", "electronic_devices", "stopped_person", "motion_images"]

for output_dir in output_dirs:
    os.makedirs(output_dir, exist_ok=True)

# YOLO and face detection setup
net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
fire_cascade = cv2.CascadeClassifier("fire_detection_cascade_model.xml")

# Motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
last_saved_faces = []  # For avoiding duplicate face saves

#media pipe setup for pose estimation
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# yolo setup for electronics detection
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]




def upload_rtsp_data():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Fetch the existing record from the database
        existing_setting = Settings.query.filter_by(rtspUrl=data.get("rtspUrl")).first()
        if existing_setting:
            return make_response(
                jsonify({"message": f"Settings with rtspUrl {data.get('rtspUrl')} already exit"}), 400
            )

        # Validate and deserialize the input data
        schema = SettingsSchema()
        new_setting = schema.load(data)  # Returns a Settings instance

        # Use the create method to insert the new record
        created_rtsp_info = Settings.create(new_setting)

        # Serialize the created object for the response
        result = schema.dump(created_rtsp_info)

        return make_response(jsonify({"message": "RTSP Created Successfully", "data": result}), 200)
    except Exception as e:
        # Handle and return the exception
        return make_response(
            jsonify({"message": "Error Occurred", "error": str(e)}),
            400
        )

def update_rtsp_data(id):
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Fetch the existing record from the database
        existing_setting = Settings.query.get(id)

        if not existing_setting:
            return make_response(
                jsonify({"message": f"Settings with id {id} not found"}), 404
            )

        # Validate and deserialize the input data
        schema = SettingsSchema()
        updated_data = schema.load(data, instance=existing_setting, partial=True)

        # Commit changes to the database
        db.session.commit()

        # Serialize the updated object for the response
        result = schema.dump(updated_data)

        return make_response(
            jsonify({"message": "RTSP Updated Successfully", "data": result}), 200
        )
    except Exception as e:
        # Handle and return the exception
        return make_response(
            jsonify({"message": "Error Occurred", "error": str(e)}),
            400
        )

def get_rtsp_data(id=None):
    try:
        schema = SettingsSchema(many=True) if id is None else SettingsSchema()
        
        # Fetch all records or a single record
        if id is None:
            records = Settings.query.all()
        else:
            records = Settings.query.get(id)
            if not records:
                return jsonify({"message": f"Settings with id {id} not found"}), 404

        # Serialize and return data
        result = schema.dump(records)
        return jsonify({"message": "Data fetched successfully", "data": result}), 200
    except Exception as e:
        return jsonify({"message": "Error Occurred", "error": str(e)}), 400


def delete_rtsp_data(id):
    try:
        # Fetch the record by id
        record = Settings.query.get(id)

        if not record:
            return jsonify({"message": f"Settings with id {id} not found"}), 404

        # Delete the record
        db.session.delete(record)
        db.session.commit()

        return jsonify({"message": f"Settings with id {id} deleted successfully"}), 200
    except Exception as e:
        return jsonify({"message": "Error Occurred", "error": str(e)}), 400


def process_rtsp_data(id):
    """
    Process a video stream based on the settings stored in the database.
    Args:
        id (int): The ID of the settings to use for processing.
    """
    try:
        # Fetch the settings from the database
        settings = Settings.query.get(id)

        if not settings:
            return jsonify({"status": "error", "message": f"Settings with id {id} not found"}), 404

        # Extract the RTSP URL
        rtsp_url = settings.rtspUrl

        # Validate the RTSP URL
        if not rtsp_url:
            return jsonify({"status": "error", "message": "RTSP URL is missing in settings"}), 400

        # Pass the settings to the video processor
        process_video_based_on_settings(rtsp_url, settings)

        return jsonify({"status": "success", "message": "Video processing started"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# def process_video_based_on_settings(rtsp_url, settings):
#     """
#     Process the video stream based on the provided settings.
#     Args:
#         rtsp_url (str): The RTSP URL of the video stream.
#         settings (Settings): Settings object with detection parameters.
#     """
#     cap = cv2.VideoCapture(rtsp_url)
#     if not cap.isOpened():
#         print(f"Error opening video stream: {rtsp_url}")
#         return

#     frame_count = 0
#     orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
#     video_writer = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1

#         # Normal motion detection
#         if settings.normal_motion:
#             fg_mask = fgbg.apply(frame)
#             if np.sum(fg_mask > 0) > 5000:  # Threshold
#                 cv2.imwrite(f"normal_motion/frame_{frame_count}.jpg", frame)

#         # Person motion detection
#         if settings.person_motion:
#             video_writer, motion_detected = process_person_motion(frame, net, output_layers, video_writer)

#         # Face detection
#         if settings.faces:
#             global last_saved_faces  # Use global variable to track saved faces

#             # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # Detect face locations using face_recognition
#             face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # or "cnn" for GPU-accelerated detection

#             new_faces = []

#             for (top, right, bottom, left) in face_locations:
#                 is_new_face = True

#                 # Compare with previously saved faces
#                 for (p_top, p_right, p_bottom, p_left) in last_saved_faces:
#                     if (
#                         abs(p_top - top) < 50 and  # Adjust tolerance as needed
#                         abs(p_right - right) < 50 and
#                         abs(p_bottom - bottom) < 50 and
#                         abs(p_left - left) < 50
#                     ):
#                         is_new_face = False
#                         break

#                 # Save new face if it's not a duplicate
#                 if is_new_face:
#                     face_region = frame[top:bottom, left:right]  # Crop the face region
#                     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#                     face_file = f"faces/face_{timestamp}.jpg"
#                     cv2.imwrite(face_file, face_region)
#                     print(f"New face saved: {face_file}")
#                     new_faces.append((top, right, bottom, left))

#             # Update the list of saved faces
#             last_saved_faces.extend(new_faces)

#             # Limit the size of `last_saved_faces` for performance
#             if len(last_saved_faces) > 100:
#                 last_saved_faces = last_saved_faces[-50:]


#         # High person count detection
#         if settings.high_person_count:
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
#             if len(faces) >= 5:  # Example threshold
#                 cv2.imwrite(f"high_person_count/frame_{frame_count}.jpg", frame)

#         # Pose detection
#         if settings.pose:
#             pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#             detect_pose(frame, pose)

#         # Fire detection
#         if settings.fire_detections:
#             detect_fire(frame)

#         # Electronic devices detection
#         if settings.electronic_devices:
#             detect_electronic_devices(frame)

#         # Stopped person detection
#         if settings.stopped_persons:
#             blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
#             net.setInput(blob)
#             detections = net.forward(output_layers)
#             people_boxes = []

#             height, width = frame.shape[:2]  # Get the dimensions of the frame

#             for detection in detections:
#                 for obj in detection:
#                     scores = obj[5:]
#                     class_id = np.argmax(scores)
#                     confidence = scores[class_id]

#                     if confidence > 0.6 and class_id == 0:  # Class 0 corresponds to 'person'
#                         center_x = int(obj[0] * width)
#                         center_y = int(obj[1] * height)
#                         box_width = int(obj[2] * width)
#                         box_height = int(obj[3] * height)

#                         # Ensure bounding box coordinates are within the frame boundaries
#                         x = max(0, int(center_x - box_width / 2))
#                         y = max(0, int(center_y - box_height / 2))
#                         box_width = min(width - x, box_width)
#                         box_height = min(height - y, box_height)

#                         people_boxes.append((x, y, box_width, box_height))

#             # Call detect_stopped_after_walking only if people_boxes is not empty
#             if people_boxes:
#                 detect_stopped_after_walking(frame, people_boxes, orig_fps)

#     if video_writer:
#         video_writer.release()
#         print("Person motion video saved.")
#     cap.release()



def send_notification(api_url, api_key, event_name, message):
    """
    Send a notification to the API endpoint.

    Args:
        api_url (str): The API endpoint URL.
        api_key (str): The API key for authentication.
        event_name (str): The name of the detected event.
        message (str): The message to send.
    """
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"event_name": event_name, "message": message}
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            print(f"Notification sent: {event_name} - {message}")
        else:
            print(f"Failed to send notification: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending notification: {e}")


def connect_to_stream(RTSP_URL):
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer size to avoid delays
    return cap


def process_video_based_on_settings(rtsp_url, settings):
    """
    Process the video stream based on the provided settings.

    Args:
        rtsp_url (str): The RTSP URL of the video stream.
        settings (Settings): Settings object with detection parameters.
    """
    # cap = cv2.VideoCapture(rtsp_url)
    cap = connect_to_stream(rtsp_url)
    last_frame_time = 0
    # # Load YOLOv5 model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True, verbose=False)

    # # Use GStreamer pipeline
    # gst_pipeline = f"rtspsrc location={rtsp_url} latency=200 ! decodebin ! videoconvert ! appsink"
    # cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    # Retry logic
    retries = 5
    while not cap.isOpened() and retries > 0:
        print(f"Error opening RTSP stream: {rtsp_url}. Retrying... ({5 - retries}/5)")
        time.sleep(2)
        cap = cv2.VideoCapture()
        retries -= 1

    if not cap.isOpened():
        print(f"Error opening video stream: {rtsp_url}")
        return

    frame_count = 0
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    video_writer = None

    # Initialize variables
    global last_saved_faces
    last_saved_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 1. Normal Motion Detection
        # if settings.normal_motion:
        #     fg_mask = fgbg.apply(frame)
        #     if np.sum(fg_mask > 0) > 5000:  # Threshold
        #         event_name = "Normal Motion Detected"
        #         message = f"Motion detected at frame {frame_count}"
        #         send_notification(settings.api_url, settings.api_key, event_name, message)
        #         cv2.imwrite(f"normal_motion/frame_{frame_count}.jpg", frame)

        if settings.normal_motion:
            video_writer, motion_detected = process_motion(frame, fgbg, video_writer)
            if motion_detected:
                event_name = "Normal Motion Detected"
                message = f"Normal motion detected at frame {frame_count}"
                send_notification(settings.api_url, settings.api_key, event_name, message)


        # 2. Person Motion Detection
        if settings.person_motion:
            video_writer, motion_detected = process_person_motion(frame, net, output_layers, video_writer)
            if motion_detected:
                event_name = "Person Motion Detected"
                message = f"Person motion detected at frame {frame_count}"
                send_notification(settings.api_url, settings.api_key, event_name, message)

        if settings.faces:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")

            new_faces = []
            for (top, right, bottom, left) in face_locations:
                is_new_face = all(
                    abs(p_top - top) >= 50 or abs(p_right - right) >= 50 or
                    abs(p_bottom - bottom) >= 50 or abs(p_left - left) >= 50
                    for (p_top, p_right, p_bottom, p_left) in last_saved_faces
                )

                if is_new_face:
                    # Expand the bounding box to include the full face region
                    padding = 20  # Add padding to ensure the full face is captured
                    top = max(0, top - padding)
                    left = max(0, left - padding)
                    bottom = min(frame.shape[0], bottom + padding)
                    right = min(frame.shape[1], right + padding)

                    face_region = frame[top:bottom, left:right]

                    # Ensure the face region is within bounds
                    if top < 0 or left < 0 or bottom > frame.shape[0] or right > frame.shape[1]:
                        continue

                    # Resize face region for better quality (optional: upscale slightly)
                    face_region = cv2.resize(face_region, (400, 400), interpolation=cv2.INTER_CUBIC)

                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    face_file = f"faces/face_{timestamp}.jpg"

                    # Save with high quality
                    cv2.imwrite(face_file, face_region, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"New face saved: {face_file}")

                    new_faces.append((top, right, bottom, left))

                    # Send notification
                    event_name = "Person Face Detected"
                    message = f"Person face detected at frame face_{timestamp}.jpg"
                    send_notification(settings.api_url, settings.api_key, event_name, message)

            last_saved_faces.extend(new_faces)
            if len(last_saved_faces) > 100:
                last_saved_faces = last_saved_faces[-50:]


        # 4. High Person Count Detection
        # if settings.high_person_count:
        #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=2)
        #     if len(faces) >= 5:
        #         event_name = "High Person Count Detected"
        #         message = f"High person count ({len(faces)} persons) detected at frame {frame_count}"
        #         send_notification(settings.api_url, settings.api_key, event_name, message)
        #         cv2.imwrite(f"high_person_count/frame_{frame_count}.jpg", frame)

        # if settings.high_person_count:
        #     # Load a pre-trained model like YOLOv5 for accurate person detection
        #     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True) # YOLOv5 small model

        #     # Convert frame to PIL image
        #     pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        #     # Perform inference
        #     results = model(pil_image)

        #     # Extract detected classes and their confidence scores
        #     detections = results.xyxy[0].cpu().numpy()  # xyxy format [x1, y1, x2, y2, confidence, class]
        #     person_count = sum(1 for det in detections if int(det[5]) == 0 and det[4] > 0.6)  # Class 0 = person

        #     # Crowd threshold from settings
        #     crowd_threshold = settings.crowd_threshold if hasattr(settings, 'crowd_threshold') else 2

        #     if person_count > crowd_threshold:
        #         # Add text indicating the count of persons
        #         text = f"Crowd Detected: {person_count}"
        #         cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #         # Trigger event notification
        #         event_name = "Crowd Detected"
        #         message = f"Crowd detected with {person_count} persons exceeding threshold of {crowd_threshold} at frame {frame_count}"
        #         send_notification(settings.api_url, settings.api_key, event_name, message)

        #         # Save the image with crowd count
        #         cv2.imwrite(f"high_person_count/frame_{frame_count}.jpg", frame)

        if settings.high_person_count:
            warnings.filterwarnings("ignore", category=FutureWarning)

            # Load a pre-trained model like YOLOv5 for accurate person detection
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True, verbose=False)  # YOLOv5 small model

            # Convert frame to PIL image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Perform inference
            results = model(pil_image)

            # Extract detected classes and their confidence scores
            detections = results.xyxy[0].cpu().numpy()  # xyxy format [x1, y1, x2, y2, confidence, class]
            person_count = sum(1 for det in detections if int(det[5]) == 0 and det[4] > 0.6)  # Class 0 = person

            # Crowd threshold from settings
            crowd_threshold = settings.crowd_threshold if hasattr(settings, 'crowd_threshold') else 2

            # Initialize video writer and detection state if not already set
            if not hasattr(settings, 'video_writer'):
                settings.video_writer = None
            if not hasattr(settings, 'crowd_detected'):
                settings.crowd_detected = False

            if person_count > crowd_threshold:
                # Start saving video if not already started
                if settings.video_writer is None:
                    video_filename = f"high_person_count/crowd_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
                    settings.video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                    print(f"Started recording video: {video_filename}")

                # Send notification only when the crowd is first detected
                if not settings.crowd_detected:
                    event_name = "Crowd Detected"
                    message = f"Crowd detected with {person_count} persons exceeding threshold of {crowd_threshold}"
                    send_notification(settings.api_url, settings.api_key, event_name, message)
                    settings.crowd_detected = True

                # Write the frame to the video
                settings.video_writer.write(frame)

            else:
                # Stop saving video if crowd is no longer detected
                if settings.video_writer is not None:
                    settings.video_writer.release()
                    settings.video_writer = None

                    # Send notification when the crowd is no longer detected
                    if settings.crowd_detected:
                        event_name = "Crowd Cleared"
                        message = f"Crowd cleared. No longer exceeding threshold of {crowd_threshold}"
                        send_notification(settings.api_url, settings.api_key, event_name, message)
                        settings.crowd_detected = False

                    print("Stopped recording video due to no crowd detected.")

        # 5. Pose Detection
        if settings.pose:
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            if detect_pose(frame, pose):
                event_name = "Pose Detected"
                message = f"Pose detected at frame {frame_count}"
                send_notification(settings.api_url, settings.api_key, event_name, message)

        # 6. Fire Detection
        if settings.fire_detections:
            if detect_fire(frame):
                event_name = "Fire Detected"
                message = f"Fire detected at frame {frame_count}"
                send_notification(settings.api_url, settings.api_key, event_name, message)

        # 7. Electronic Devices Detection
        if settings.electronic_devices:
            if detect_electronic_devices(frame):
                event_name = "Electronic Device Detected"
                message = f"Electronic device detected at frame {frame_count}"
                send_notification(settings.api_url, settings.api_key, event_name, message)

        # 8. Stopped Person Detection
        if settings.stopped_persons:
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward(output_layers)
            people_boxes = []

            height, width = frame.shape[:2]
            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.6 and class_id == 0:  # Class 0 corresponds to 'person'
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        box_width = int(obj[2] * width)
                        box_height = int(obj[3] * height)

                        x = max(0, int(center_x - box_width / 2))
                        y = max(0, int(center_y - box_height / 2))
                        box_width = min(width - x, box_width)
                        box_height = min(height - y, box_height)

                        people_boxes.append((x, y, box_width, box_height))

            if people_boxes:
                detect_stopped_after_walking(frame, people_boxes, orig_fps)
                event_name = "Stopped Person Detected"
                message = f"Stopped person detected at frame {frame_count}"
                send_notification(settings.api_url, settings.api_key, event_name, message)

    if video_writer:
        video_writer.release()
        print("Person motion video saved.")

    cap.release()


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def detect_pose(frameOg, pose):
    frame = frameOg.copy()
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark


        # print("saving frame for debug")
        # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # cv2.imwrite(f"pose/temp_{timestamp}.jpg", frame)
        # print(f"Frame saved: temp_{timestamp}.jpg")

        # Get all required landmarks
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]

        # Calculate all required angles
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        
        # Enhanced visibility check
        min_visibility = 0.65
        landmarks_to_check = [right_shoulder, left_shoulder, right_hip, left_hip, 
                            right_knee, left_knee, right_ankle, left_ankle,
                            right_foot, left_foot]
        
        if all(landmark.visibility > min_visibility for landmark in landmarks_to_check):
            # Get bounding box
            bbox = get_bounding_box(landmarks_to_check, frame.shape)
            
            # Enhanced standing pose detection
            vertical_aligned = (
                right_shoulder.y < right_hip.y < right_knee.y < right_ankle.y and
                left_shoulder.y < left_hip.y < left_knee.y < left_ankle.y and
                abs(right_shoulder.y - left_shoulder.y) < 0.05 and  # Shoulders level
                abs(right_hip.y - left_hip.y) < 0.05  # Hips level
            )
            
            hip_angle_range = (160, 200)  # More precise range for standing
            knee_angle_range = (160, 200)
            
            angles_correct = (
                hip_angle_range[0] <= right_hip_angle <= hip_angle_range[1] and
                hip_angle_range[0] <= left_hip_angle <= hip_angle_range[1] and
                knee_angle_range[0] <= right_knee_angle <= knee_angle_range[1] and
                knee_angle_range[0] <= left_knee_angle <= knee_angle_range[1]
            )
            
            # Enhanced lateral sway check
            shoulder_center_x = (right_shoulder.x + left_shoulder.x) / 2
            hip_center_x = (right_hip.x + left_hip.x) / 2
            max_lateral_sway = 0.08  # Reduced tolerance
            
            minimal_sway = abs(shoulder_center_x - hip_center_x) < max_lateral_sway
            
            # Enhanced sitting pose detection
            sitting_hip_angle_range = (65, 115)  # Adjusted for sitting
            sitting_knee_angle_range = (65, 115)
            
            # Additional sitting checks
            hip_level = (right_hip.y - right_knee.y) / (right_ankle.y - right_knee.y)
            knees_level = abs(right_knee.y - left_knee.y) < 0.05
            
            is_sitting = (
                sitting_hip_angle_range[0] <= right_hip_angle <= sitting_hip_angle_range[1] and
                sitting_hip_angle_range[0] <= left_hip_angle <= sitting_hip_angle_range[1] and
                sitting_knee_angle_range[0] <= right_knee_angle <= sitting_knee_angle_range[1] and
                sitting_knee_angle_range[0] <= left_knee_angle <= sitting_knee_angle_range[1] and
                hip_level < 0.7 and  # Hips are lower relative to knees
                knees_level  # Knees are approximately at the same level
            )
            
            # Enhanced walking detection
            step_size = 0.08  # Minimum step size threshold
            leg_height_diff = abs(right_ankle.y - left_ankle.y)
            knee_height_diff = abs(right_knee.y - left_knee.y)
            foot_distance = abs(right_foot.x - left_foot.x)
            
            # Check for dynamic movement patterns
            knee_bent = (right_knee_angle < 160 or left_knee_angle < 160)
            legs_apart = foot_distance > 0.15
            
            is_walking = (
                (leg_height_diff > step_size or knee_height_diff > step_size) and
                knee_bent and
                legs_apart and
                not vertical_aligned and
                not is_sitting
            )

            # Draw bounding box with appropriate color and label
            if vertical_aligned and angles_correct and minimal_sway:
                # Standing - Green box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, 'Standing', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
            elif is_sitting:
                # Sitting - Yellow box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
                cv2.putText(frame, 'Sitting', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                
            elif is_walking:
                # Walking - Red box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, 'Walking', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Save frame if standing pose is detected
            if vertical_aligned and angles_correct and minimal_sway:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                standing_pose_file = f"pose/standing_pose_{timestamp}.jpg"
                cv2.imwrite(standing_pose_file, frame)
                print(f"Standing pose saved: {standing_pose_file}")
                return True
            elif is_sitting:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                sitting_pose_file = f"pose/sitting_pose_{timestamp}.jpg"
                cv2.imwrite(sitting_pose_file, frame)
                print(f"Sitting pose saved: {sitting_pose_file}")
                return True
            elif is_walking:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                walking_pose_file = f"pose/walking_pose_{timestamp}.jpg"
                cv2.imwrite(walking_pose_file, frame)
                print(f"Walking pose saved: {walking_pose_file}")
                return True

    return False


def detect_fire(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fire_regions = fire_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)
    temp_frame = frame.copy()

    if(len(fire_regions) > 0):
        for(x, y, w, h) in fire_regions:
            cv2.rectangle(temp_frame, (x -20, y-20), (x + w +20, y + h +20), (0, 0, 255), 2)
            # cv2.putText(temp_frame, "Fire", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fire_file = f"fire_detections/fire_detections_{timestamp}.jpg"
        cv2.imwrite(fire_file, temp_frame)
        print(f"Fire detections saved: {fire_file}")
        return True
    return False


# def detect_electronic_devices(frameOg):
#     frame = frameOg.copy()
#     height, width, _ = frame.shape
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)

#     boxes = []
#     confidences = []
#     class_ids = []
#     detections = []

#     electronic_classes = [
#         "tv", "laptop", "cell phone", "keyboard", "mouse", 
#         "remote", "book"
#     ]

#     try:
#         outputs = net.forward(output_layers)

#         for output in outputs:
#             for detection in output:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]

#                 if confidence > 0.5 and classes[class_id].lower() in electronic_classes:
#                     center_x = int(detection[0] * width)
#                     center_y = int(detection[1] * height)
#                     w = int(detection[2] * width)
#                     h = int(detection[3] * height)

#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)

#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)
#                     detections.append({
#                         'class': classes[class_id],
#                         'confidence': float(confidence)
#                     })

#         indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#         colors = np.random.uniform(0, 255, size=(len(boxes), 3))

#         if len(indexes) > 0:
#             for i in range(len(boxes)):
#                 if i in indexes:
#                     x, y, w, h = boxes[i]
#                     label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
#                     color = colors[i]

#                     cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                     cv2.putText(frame, label, (x, y - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             try:
#                 timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#                 electronic_devices_file = f"electronic_devices/electronic_devices_{timestamp}.jpg"
#                 cv2.imwrite(electronic_devices_file, frame)
#                 print(f"Electronic devices saved: {electronic_devices_file}")
#                 return true
#             except Exception as e:
#                 print(f"Error saving image: {e}")
#                 return false

#     except Exception as e:
#         print(f"Error during detection: {e}")
#         return None, None, []

def detect_electronic_devices(frameOg):
    frame = frameOg.copy()
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    boxes = []
    confidences = []
    class_ids = []
    detections = []

    electronic_classes = [
        "tv", "laptop", "cell phone", "keyboard", "mouse", 
        "remote", "book"
    ]

    try:
        outputs = net.forward(output_layers)

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id].lower() in electronic_classes:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    detections.append({
                        'class': classes[class_id],
                        'confidence': float(confidence)
                    })

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        if len(indexes) > 0:
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                    color = colors[i]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            try:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                electronic_devices_file = f"electronic_devices/electronic_devices_{timestamp}.jpg"
                cv2.imwrite(electronic_devices_file, frame)
                print(f"Electronic devices saved: {electronic_devices_file}")
                return True
            except Exception as e:
                print(f"Error saving image: {e}")
                return False

        else:
            print("No electronic devices detected.")
            return False

    except Exception as e:
        print(f"Error during detection: {e}")
        return False

def process_person_motion(frame, net, output_layers, video_writer, output_folder="person_motion", motion_threshold=5000, min_area=500):
    """
    Detect motion and process frames for saving images and video.
    
    Args:
        frame (np.array): The current frame of the video.
        net: Preloaded YOLO model.
        output_layers (list): YOLO output layers.
        video_writer (cv2.VideoWriter): Video writer to save motion video.
        output_folder (str): Folder to save the images and video.
        motion_threshold (int): Minimum pixel difference to detect motion.
        min_area (int): Minimum area for contours to be considered as motion.
    
    Returns:
        cv2.VideoWriter: Updated video writer object.
        bool: Whether motion was detected in the current frame.
    """
    height, width, _ = frame.shape
    fg_mask = fgbg.apply(frame)  # Apply background subtraction
    motion_detected = np.sum(fg_mask > 0) > motion_threshold

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    if motion_detected:
        # Find contours to highlight motion areas
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > min_area:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the image
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_file = f"{output_folder}/motion_frame_{timestamp}.jpg"
        cv2.imwrite(image_file, frame)
        print(f"Motion image saved: {image_file}")

        # Start recording video if not already recording
        if video_writer is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_file = f"{output_folder}/motion_video_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_file, fourcc, 30, (width, height))
            print(f"Started recording motion video: {video_file}")

        # Write the frame to the video
        video_writer.write(frame)
    else:
        # Stop recording video if no motion is detected
        if video_writer:
            video_writer.release()
            video_writer = None
            print("Stopped recording motion video.")

    return video_writer, motion_detected


def process_motion(frame, fgbg, video_writer=None, output_folder="normal_motion", motion_threshold=5000, min_area=500, fps=30):
    """
    Detect and process motion for saving images and videos.

    Args:
        frame (np.array): The current video frame.
        fgbg (cv2.BackgroundSubtractor): Background subtractor for motion detection.
        video_writer (cv2.VideoWriter): Current video writer object.
        output_folder (str): Folder to save motion images and videos.
        motion_threshold (int): Minimum threshold for pixel changes to detect motion.
        min_area (int): Minimum area for contours to be considered as motion.
        fps (int): Frame rate for the output video.

    Returns:
        cv2.VideoWriter: Updated video writer object.
        bool: Whether motion was detected.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    height, width, _ = frame.shape
    fg_mask = fgbg.apply(frame)  # Apply background subtraction

    # Reduce noise in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Close gaps
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)   # Remove noise

    # Detect contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > min_area:  # Filter small areas
            motion_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if motion_detected:
        # Save motion image
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_file = f"{output_folder}/motion_frame_{timestamp}.jpg"
        cv2.imwrite(image_file, frame)
        print(f"Motion image saved: {image_file}")

        # Start recording video if not already recording
        if video_writer is None:
            video_file = f"{output_folder}/motion_video_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
            print(f"Started recording motion video: {video_file}")

        # Write the current frame to the video
        video_writer.write(frame)
    else:
        # Stop recording if no motion is detected
        if video_writer is not None:
            video_writer.release()
            print("Stopped recording motion video.")
            video_writer = None

    return video_writer, motion_detected


def detect_stopped_after_walking(frame, people_boxes, fps, stop_duration=2, movement_tolerance=10):
    """
    Detect persons who have stopped moving after walking and save their images with a green frame outline.
    
    Args:
        frame (np.array): Current video frame.
        people_boxes (list): List of bounding boxes [(x, y, w, h)] for detected people.
        fps (int): Frames per second of the video.
        stop_duration (float): Duration (in seconds) to classify as stopped.
        movement_tolerance (int): Maximum allowed movement (in pixels) to classify as stationary.
    """
    global person_states

    stopped_positions = []
    stop_frames = int(stop_duration * fps)  # Number of frames to check for stopped motion

    for i, (x, y, w, h) in enumerate(people_boxes):
        person_id = f"person_{i}"
        current_position = (x, y)

        # Add the current position to the history
        person_states[person_id]["positions"].append(current_position)
        if len(person_states[person_id]["positions"]) > stop_frames:
            person_states[person_id]["positions"].pop(0)

        # Ensure there are enough positions to calculate movements
        if len(person_states[person_id]["positions"]) > 1:
            movements = [
                abs(person_states[person_id]["positions"][j + 1][0] - person_states[person_id]["positions"][j][0]) +
                abs(person_states[person_id]["positions"][j + 1][1] - person_states[person_id]["positions"][j][1])
                for j in range(len(person_states[person_id]["positions"]) - 1)
            ]

            # Check movements only if the list is not empty
            if movements and max(movements) < movement_tolerance:
                if person_states[person_id]["state"] == "walking":
                    # Transition to "stopped" state if stationary for the duration
                    if person_states[person_id]["last_moved"] is None:
                        person_states[person_id]["last_moved"] = time.time()
                    elif time.time() - person_states[person_id]["last_moved"] >= stop_duration:
                        person_states[person_id]["state"] = "stopped"
                        stopped_positions.append((x, y, w, h))
            else:
                # Reset state to "walking" if movement is detected
                person_states[person_id]["state"] = "walking"
                person_states[person_id]["last_moved"] = None

    # Save images for stopped persons
    for (x, y, w, h) in stopped_positions:
        stopped_frame = frame.copy()
        cv2.rectangle(stopped_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green frame
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stopped_file = f"{stopped_person_folder}/stopped_person_{timestamp}.jpg"
        cv2.imwrite(stopped_file, stopped_frame)
        print(f"Stopped person saved: {stopped_file}")
        return true





def list_files(folder):
    if folder not in output_dirs:
        return jsonify({'status': 'error', 'message': 'Invalid folder'})
    try:
        files = os.listdir(folder)  # List all files in the specified folder
        return jsonify({'status': 'success', 'files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def serve_file(folder, filename):
    """
    Serve a specific file from the specified folder.
    """
    if folder not in output_dirs:
        return jsonify({'status': 'error', 'message': 'Invalid folder'}), 400

    try:
        folder_path = os.path.abspath(folder)  # Get the absolute path of the folder
        file_path = os.path.join(folder_path, filename)  # Construct the file path

        if not os.path.isfile(file_path):
            return jsonify({'status': 'error', 'message': 'File not found'}), 404

        # Serve the file with appropriate mimetype
        mimetype = 'video/mp4' if filename.endswith('.mp4') else None
        return send_from_directory(folder_path, filename, mimetype=mimetype)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


def delete_all_files():
    """
    Delete all files from the specified output directories.
    """
    try:
        for folder in output_dirs:
            folder_path = os.path.abspath(folder)  # Get the absolute path of the folder
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    os.remove(file_path)  # Remove the file
        return jsonify({'status': 'success', 'message': 'All files deleted'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def api_test():
    data = request.get_json()
    print(data)
    return make_response(jsonify({'status': 'success', 'message': 'API test successful'}), 200)
