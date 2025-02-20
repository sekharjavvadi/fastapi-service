import cv2
from dotenv import load_dotenv
import mediapipe as mp 
import requests
import tempfile
import os 
import numpy as np
from azure.storage.blob import BlobServiceClient, ContentSettings
from scipy.signal import find_peaks
from moviepy.video.io.VideoFileClip import VideoFileClip 
from pydub import AudioSegment 
from ultralytics import YOLO
AudioSegment.converter = "C:\\Users\\91949\\Downloads\\ffmpeg-2025-02-10-git-a28dc06869-full_build\\ffmpeg-2025-02-10-git-a28dc06869-full_build\\bin\\ffmpeg.exe"
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
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_file.name, 'wb') as f:
            f.write(response.content)
        return temp_file.name 
    return None

def upload_to_azure(file_path, blob_name, content_type):
    """Upload a file to Azure Blob Storage and return the file URL."""
    try:
        with open(file_path, "rb") as data:
            content_settings = ContentSettings(content_type=content_type)
            blob_client = blob_container_client.get_blob_client(blob_name)
            blob_client.upload_blob(data, overwrite=True, content_settings=content_settings)
        return blob_client.url
    except Exception as e:
        print(f"Failed to upload to Azure: {e}") 
        return None

def save_screenshot(frame, timestamp_in_seconds, video_name):
    """Save a screenshot of a frame and upload it to Azure."""
    screenshot_name = f"{video_name}_{timestamp_in_seconds}.png"
    local_path = f"{tempfile.gettempdir()}/{screenshot_name}"

    cv2.imwrite(local_path, frame)
    azure_url = upload_to_azure(local_path, screenshot_name, "image/png")

    return azure_url
# ---------------------------------------------------
# Audio Extraction Method
# ---------------------------------------------------
def extract_audio_from_video(video_path):
    """Extract audio from video and save it as an audio file."""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        audio.write_audiofile(audio_path, codec="pcm_s16le") 
        return audio_path 
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None
# ---------------------------------------------------
# Multiple Voice Detection Method
# ---------------------------------------------------
def detect_dual_voices(audio_path):
    """Detect dual voices dynamically without a fixed height threshold."""
    try:
        audio = AudioSegment.from_wav(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000) 
        samples = np.array(audio.get_array_of_samples())

        samples_per_second = len(samples) // int(audio.duration_seconds)
        voice_detection_results = {}  
        merged_clips = []  
        audio_clips = {}  
        ongoing_clip_start = None

        # **Dynamic threshold calculation**
        mean_amplitude = np.mean(np.abs(samples))  # Mean absolute amplitude
        std_dev_amplitude = np.std(np.abs(samples))  # Standard deviation of amplitudes
        dynamic_threshold = mean_amplitude + (std_dev_amplitude * 1.5)  # Adjust factor if needed

        for second in range(int(audio.duration_seconds)):
            start = second * samples_per_second
            end = (second + 1) * samples_per_second
            segment = samples[start:end]
            peaks, _ = find_peaks(segment, height=dynamic_threshold, distance=8000)  # No fixed height
            voice_detected = len(peaks) > 1
            voice_detection_results[second] = voice_detected

            if voice_detected:
                if ongoing_clip_start is None:
                    ongoing_clip_start = second  
            else:
                if ongoing_clip_start is not None:
                    merged_clips.append((ongoing_clip_start, second - 1))
                    ongoing_clip_start = None  

        if ongoing_clip_start is not None:
            merged_clips.append((ongoing_clip_start, int(audio.duration_seconds) - 1))

        for start, end in merged_clips:
            clip = audio[start * 1000:(end + 1) * 1000]  
            mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            clip.export(mp3_path, format="mp3")  

            audio_blob_name = f"merged_audio_{start}-{end}.mp3"
            audio_url = upload_to_azure(mp3_path, audio_blob_name, "audio/mpeg")

            audio_clips[f"{start}-{end}"] = audio_url  
            os.remove(mp3_path)  

        return voice_detection_results, merged_clips, audio_clips  

    except Exception as e:
        print(f"Error detecting dual voices: {e}")
        return {}, [], {}


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


def detect_multiple_voices_endpoint(video_url):
    """Endpoint for multiple voice detection."""
    if not video_url:
        return {"error": "No video URL provided"}

    video_path = download_video(video_url)
    if not video_path:
        return {"error": "Failed to download video"}

    audio_path = extract_audio_from_video(video_path)
    
    try:
        os.remove(video_path)  # Remove video file after extracting audio
    except Exception as e:
        print(f"Warning: Unable to delete video file. {e}")

    if not audio_path:
        return {"error": "Failed to extract audio"}

    voice_results, timestamps, audio_clips = detect_dual_voices(audio_path)
    os.remove(audio_path)  # Remove extracted audio after processing

    output = {
        "multiple_voice_detection_results": voice_results,
        "dual_voice_segments": []  # List to store merged timestamps and audio URLs
    }

    for start, end in timestamps:
        audio_url = audio_clips.get(f"{start}-{end}", None)
        output["dual_voice_segments"].append({
            "timestamp": start,
            "start_time": start,
            "end_time": end,
            "audio_url": audio_url,
            "detected": True
        })

    # Add False detection cases when no multiple voices were detected
    for second, detected in voice_results.items():
        if not detected:
            output["dual_voice_segments"].append({ 
                "timestamp": second,
                "detected": False
            })
    
    return output


# ---------------------------------------------------

# Example Usage
# ---------------------------------------------------
# tabshifted_url = "https://reaidystorage.blob.core.windows.net/recordings/679e05c2ba3c82edd31d87af.mp4"
video_url = "https://reaidytesting.blob.core.windows.net/recordings/VID_20250217_170726.mp4"
input_seconds = 3

# Object Detection
# result = object_detection_endpoint(video_url , input_seconds)
# print(result)

# Tab Shifting
# tab_shift_result = tab_shift_endpoint(tabshifted_url, input_seconds)
# print(tab_shift_result)

# Face Detection
# face_result = analyze_video_endpoint(video_url)
# print(face_result)

# Multiple Voice Detection
# voice_analysis = detect_multiple_voices_endpoint(video_url)
# print("Voice Analysis:", voice_analysis) 


