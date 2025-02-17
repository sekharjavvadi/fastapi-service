import cv2
from dotenv import load_dotenv
import mediapipe as mp
import requests
import tempfile
import os
import numpy as np
from azure.storage.blob import BlobServiceClient

# Load environment variables
from ultralytics import YOLO
load_dotenv()

# YOLOv8 Model Initialization
yolo_model = YOLO('yolov8x.pt')  # Using the more accurate model

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# Azure Blob Storage Configuration
AZURE_CONNECTION_STRING = os.getenv("proxy_connect_str")
AZURE_CONTAINER_NAME = os.getenv("proxy_container_name")

# Azure Blob Storage Client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
blob_container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

def download_video(url):
    """Download video to a temporary file and return the path."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, headers=headers)

    if response.status_code == 200:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return temp_file.name
    else:
        return None

def upload_to_azure(file_path, blob_name):
    """Upload a file to Azure Blob Storage and return the file URL."""
    try:
        with open(file_path, "rb") as data:
            blob_client = blob_container_client.get_blob_client(blob_name)
            blob_client.upload_blob(data, overwrite=True, content_type="image/jpeg")

        return f"https://{blob_service_client.account_name}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{blob_name}"
    except Exception as e:
        print(f"Failed to upload to Azure: {e}")
        return None

def save_screenshot(frame, timestamp, video_name):
    """Save the current frame as a screenshot and upload it to Azure."""
    screenshot_name = f"{video_name}screenshot{timestamp}.jpg"
    local_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    cv2.imwrite(local_path, frame)  # Save frame locally
    
    azure_url = upload_to_azure(local_path, screenshot_name)
    os.remove(local_path)  # Clean up local file
    
    return azure_url

# ---------------------------------------------------
# Face Detection Method
# ---------------------------------------------------
def analyze_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Fallback if FPS is not provided
    results_list = []  # To store the results as per the desired format

    last_processed_second = -1  # Track the last second processed to avoid duplicates
    video_name = os.path.basename(video_path).split('.')[0]  # Extract video name for blob naming

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp_in_seconds = int(frame_count / fps)  # Get timestamp in seconds

            # Process only once per second
            if timestamp_in_seconds > last_processed_second:
                last_processed_second = timestamp_in_seconds


                # Convert frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                # Default values
                head_position = "unknown"
                multiple_face_detection = False

                if results.detections:
                    face_count = len(results.detections)
                    if face_count > 1:
                        multiple_face_detection = True
                    
                    for detection in results.detections:
                        # Improved head orientation analysis
                        bbox = detection.location_data.relative_bounding_box
                        if bbox.xmin < 0.3:
                            head_position = "left"
                        elif bbox.xmin > 0.7:
                            head_position = "right"
                        else:
                            head_position = "forward"
                else:
                    head_position = "away"  # No face detected
                if head_position != "forward" or multiple_face_detection == True:
                        # Save the current frame and upload to Azure
                        azure_url = save_screenshot(frame, timestamp_in_seconds, video_name)
                        results_list.append({
                    "time": timestamp_in_seconds,
                    "head_position": head_position,
                    "multiple_face_detection": multiple_face_detection,
                    "screenshot_url": azure_url      })
                        
                else:
                # Append results for the current timestamp
                    results_list.append({
                    "time": timestamp_in_seconds,
                    "head_position": head_position,
                    "multiple_face_detection": multiple_face_detection
                })

    cap.release()
    return results_list

def detect_tab_switch(prev_frame, current_frame, threshold=50):
    """Detect tab switching by measuring the difference between consecutive frames."""
    if prev_frame is None or current_frame is None:
        return False
 
    # Resize frames to the same dimensions
    height, width = current_frame.shape[:2]
    prev_frame_resized = cv2.resize(prev_frame, (width, height))
 
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame_resized, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
 
    # Calculate absolute difference
    diff = cv2.absdiff(prev_gray, curr_gray)
    mean_diff = np.mean(diff)
 
    return mean_diff > threshold  # If large difference, tab switched

# ---------------------------------------------------
# Object Detection Method
# ---------------------------------------------------
def object_detection(video_path, input_seconds):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30

    results_list = []
    last_processed_second = -1
    video_name = os.path.basename(video_path).split('.')[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp_in_seconds = int(frame_count / fps)

        if timestamp_in_seconds > last_processed_second:
            last_processed_second = timestamp_in_seconds

            # YOLOv8 Object Detection
            yolo_results = yolo_model(frame, conf=0.2)[0]
            detected_objects = []
            target_classes = ["laptop", "cell phone"]

            for box in yolo_results.boxes:
                cls_id = int(box.cls[0].item())
                label = yolo_model.names[cls_id].lower().replace("-", " ")

                if label in target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf[0].item()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detected_objects.append(label)

            if detected_objects:
                azure_url = save_screenshot(frame, timestamp_in_seconds, video_name)
                results_list.append({
                    "time": timestamp_in_seconds,
                    "detected_objects": detected_objects,
                    "screenshot_url": azure_url
                })

    cap.release()
    return results_list

# ---------------------------------------------------
# Tab Shifting Method
# ---------------------------------------------------
def tab_shift_detection(video_path, input_seconds):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30

    results_list = []
    last_processed_second = -1
    prev_frame = None
    video_name = os.path.basename(video_path).split('.')[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp_in_seconds = int(frame_count / fps)

        if timestamp_in_seconds > last_processed_second:
            last_processed_second = timestamp_in_seconds

            tab_switched = detect_tab_switch(prev_frame, frame)
            screenshot_url = None

            if tab_switched:
                screenshot_url = save_screenshot(frame, timestamp_in_seconds, video_name)

            results_list.append({
                "time": timestamp_in_seconds,
                "tab_switched": str(tab_switched).lower(),
                "screenshot_url": screenshot_url
            })

            prev_frame = frame.copy()

    cap.release()
    return results_list


# ---------------------------------------------------
# Endpoint Functions for API
# ---------------------------------------------------
def object_detection_endpoint(video_url, input_seconds):
    video_path = download_video(video_url)
    if not video_path:
        return {"error": "Failed to download video"}
    
    result = object_detection(video_path, input_seconds)
    os.remove(video_path)  # Clean up downloaded video
    return {"object_detection_result": result}

def tab_shift_endpoint(tabshifted_url, input_seconds):
    video_path = download_video(tabshifted_url)
    if not video_path:
        return {"error": "Failed to download video"}
    
    result = tab_shift_detection(video_path, input_seconds)
    os.remove(video_path)  # Clean up downloaded video
    return {"tab_shift_result": result}

def analyze_video_endpoint(video_url):
    # data = await request.json()
    # video_url = data.get("video_url")
    if not video_url:
        return {"error": "No video URL provided"}

    # Download the video temporarily if needed
    video_path = download_video(video_url)
    if not video_path:
        return {"error": "Failed to download video from the URL provided"}

    # Perform analysis on the video
    result = analyze_video_stream(video_path)

    # Clean up the downloaded video file
    os.remove(video_path)

    return {"analysis_result": result}

# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------
# tabshifted_url = "https://testingreaidy.blob.core.windows.net/recording/MicrosoftTeams-video.mp4"
video_url = "https://testingreaidy.blob.core.windows.net/recording/1739446444519_original-8a20c2e2-6da5-447a-ab21-f1379a13e2c1.mp4"
input_seconds = 3

# Object Detection
result = object_detection_endpoint(video_url , input_seconds)
print(result)

# Tab Shifting
# tab_shift_result = tab_shift_endpoint(tabshifted_url, input_seconds)
# print(tab_shift_result)

# Face Detection
# face_result = analyze_video_endpoint(video_url)
# print(face_result)