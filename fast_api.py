import cv2
from dotenv import load_dotenv
import mediapipe as mp
import requests
import tempfile
import os
from azure.storage.blob import BlobServiceClient, ContentSettings
from ultralytics import YOLO

load_dotenv()

# Initialize MediaPipe for face and landmark detection
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
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_file.name, 'wb') as f:
            f.write(response.content)
        return temp_file.name
    else:
        return None

def upload_to_azure(file_path, blob_name):
    """Upload a file to Azure Blob Storage and return the file URL."""
    try:
        with open(file_path, "rb") as data:
            content_settings = ContentSettings(content_type="image/jpeg")
            blob_client = blob_container_client.get_blob_client(blob_name)
            blob_client.upload_blob(data, overwrite=True, content_settings=content_settings)
        return blob_client.url
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

def analyze_video_stream(video_path, input_seconds):
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
            timestamp_in_seconds = int(frame_count / fps)

            # Process only once per second
            if timestamp_in_seconds > last_processed_second:
                last_processed_second = timestamp_in_seconds

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
                        "detected_objects": detected_objects,
                        "screenshot_url": azure_url
                    })
                else:
                    results_list.append({
                        "time": timestamp_in_seconds,
                        "head_position": head_position,
                        "multiple_face_detection": multiple_face_detection,
                        "detected_objects": detected_objects
                    })

    cap.release()
    return results_list

def analyze_video_endpoint(video_url, input_seconds):
    if not video_url:
        return {"error": "No video URL provided"}

    video_path = download_video(video_url)
    if not video_path:
        return {"error": "Failed to download video from the URL provided"}

    result = analyze_video_stream(video_path, input_seconds)
    os.remove(video_path)
    return {"analysis_result": result}

# Example usage
video_url = "https://testingreaidy.blob.core.windows.net/recording/1739446444519_original-8a20c2e2-6da5-447a-ab21-f1379a13e2c1.mp4" 
input_seconds = 3
result = analyze_video_endpoint(video_url, input_seconds)
print(result)  