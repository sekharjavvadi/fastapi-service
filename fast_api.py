import cv2
from dotenv import load_dotenv
import mediapipe as mp
import requests
import tempfile
import os
import numpy as np
from azure.storage.blob import BlobServiceClient,ContentSettings

# Load environment variables
from ultralytics import YOLO

load_dotenv()

# Initialize MediaPipe for face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load a more accurate YOLOv8 model
yolo_model = YOLO('yolov8x.pt')  # Use the extra-large model for better accuracy

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

        # Construct the correct URL for the uploaded file
        return f"https://{blob_service_client.account_name}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{blob_name}"
    except Exception as e:
        print(f"Failed to upload to Azure: {e}")
        return None

def save_screenshot(frame, timestamp, video_name):
    """Save the current frame as a screenshot and upload it to Azure."""
    screenshot_name = f"{video_name}_screenshot_{timestamp}.jpg"
    local_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    cv2.imwrite(local_path, frame)  # Save frame locally
    
    # Upload to Azure
    azure_url = upload_to_azure(local_path, screenshot_name)
    
    # Clean up local file
    os.remove(local_path)
    
    return azure_url

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

def analyze_video_stream(video_path, input_seconds):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30  # Default fallback FPS

    results_list = []  # To store the results as per the desired format
    last_processed_second = -1  # Track the last second processed to avoid duplicates
    video_name = os.path.basename(video_path).split('.')[0]  # Extract video name for blob naming
    prev_frame = None  # Store previous frame for tab detection

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp_in_seconds = int(frame_count / fps)

            # Process only once per second
            if timestamp_in_seconds > last_processed_second:
                last_processed_second = timestamp_in_seconds

                # Detect tab switch
                tab_switched = detect_tab_switch(prev_frame, frame)

                # Convert frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Default values for face detection
                head_position = "unknown"
                multiple_face_detection = "false"

                if results.detections:
                    face_count = len(results.detections)
                    if face_count > 1:
                        multiple_face_detection = "true"
                    
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        if bbox.xmin < 0.3:
                            head_position = "left"
                        elif bbox.xmin > 0.7:
                            head_position = "right"
                        else:
                            head_position = "forward"
                else:
                    head_position = "away"  # No face detected

                # Determine if a screenshot should be taken
                azure_url = None
                if head_position == "forward" and (tab_switched or multiple_face_detection == "true") or head_position != "forward":
                    azure_url = save_screenshot(frame, timestamp_in_seconds, video_name)  # Upload to Azure

                # Append results for the current timestamp
                # results_list.append({
                #     "time": timestamp_in_seconds,
                #     "head_position": head_position,
                #     "multiple_face_detection": multiple_face_detection,
                #     "tab_switched": str(tab_switched).lower(),
                #     "screenshot_url": azure_url if azure_url else None
                # })
                # YOLOv8 Object Detection
                yolo_results = yolo_model(frame, conf=0.2)[0]  # Set to 0.2 for more detections
                detected_objects = []
                target_classes = ["laptop", "cell phone"]

                for box in yolo_results.boxes:
                    # x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # confidence = box.conf[0].item()
                    # cls_id = int(box.cls[0].item())
                    # label = yolo_model.names[cls_id]
                    cls_id = int(box.cls[0].item())
                    label = yolo_model.names[cls_id]
                    label = label.lower().replace("-", " ")

                    # Filter for laptop and cell phone only
                    if label in target_classes:
                        # Draw green bounding box and label
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = box.conf[0].item()
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {confidence:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                        detected_objects.append(label)

                # Save Screenshot if conditions met
                if head_position != "forward" or detected_objects or multiple_face_detection == "true":
                    azure_url = save_screenshot(frame, timestamp_in_seconds, video_name)
                    results_list.append({
                        "time": timestamp_in_seconds,
                        "head_position": head_position,
                        "multiple_face_detection": multiple_face_detection,
                        "tab_switched": str(tab_switched).lower(),
                        "detected_objects": detected_objects,
                        "screenshot_url": azure_url
                    })
                else:
                    results_list.append({
                        "time": timestamp_in_seconds,
                        "head_position": head_position,
                        "multiple_face_detection": multiple_face_detection,
                        "tab_switched": str(tab_switched).lower(),
                        "detected_objects": detected_objects,
                        "screenshot_url": azure_url if azure_url else None
                    })

                prev_frame = frame.copy()  # Ensure deep copy to avoid memory issues

    cap.release()
    return results_list

def analyze_video_endpoint(video_url, input_seconds):
    """Process the video and return the analysis result."""
    if not video_url:
        return {"error": "No video URL provided"}

    # Download the video temporarily
    video_path = download_video(video_url)
    if not video_path:
        return {"error": "Failed to download video from the URL provided"}

    # Perform analysis on the video
    result = analyze_video_stream(video_path, input_seconds)

    # Clean up the downloaded video file
    os.remove(video_path)
    return {"analysis_result": result}

def analyze_tab_shift_master_func(video_url, input_seconds):
    """Process the video and return the analysis result."""
    if not video_url:
        return {"error": "No video URL provided"}

    # Download the video temporarily
    video_path = download_video(video_url)
    if not video_path:
        return {"error": "Failed to download video from the URL provided"}

    # Perform analysis on the video
    result = analyze_video_stream(video_path, input_seconds)

    # Clean up the downloaded video file
    os.remove(video_path)
    return {"analysis_result": result}





# Example usage
# if __name__ == "__main__":
video_url = "https://testingreaidy.blob.core.windows.net/recording/1739446444519_original-8a20c2e2-6da5-447a-ab21-f1379a13e2c1.mp4"
screen_share_video = "https://reaidystorage.blob.core.windows.net/recordings/679e05c2ba3c82edd31d87af.mp4"
input_seconds = 3

# video analysis functionality like object detection
result = analyze_video_endpoint(video_url, input_seconds)
print(result)


#  tab shifting functionality
result2 = analyze_tab_shift_master_func(screen_share_video, input_seconds)
print(result2)



# Requirements
# opencv-python
# mediapipe 
# requests
# azure_functions
# azure_storage_blob 
# python_dotenv
# numpy
