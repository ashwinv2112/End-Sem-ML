import cv2
import time
import numpy as np
import yagmail
from datetime import datetime
from ultralytics import YOLO
# from playsound import playsound
import threading
import winsound  # Alternative audio library for Windows

# === CONFIGURATION ===

# File paths
animal_model_path = r"dataset1"
traffic_model_path = r"dataset2"

animal_video_paths = [
    r"Cam1",
    r"Cam2",
    r"Cam3",
    r"Cam4"
]

traffic_video_paths = [
    r"Cam5",
    r"Cam6"
]

# Email Setup
EMAIL_SENDER = "sendersemail@gmail.com"
EMAIL_PASSWORD = "password"  # Make sure you set your real password or app password
EMAIL_RECEIVERS = ["recieversemail@gmail.com"]
ALERT_AUDIO_PATH = r"audio.mp3"

yag = yagmail.SMTP(user=EMAIL_SENDER, password=EMAIL_PASSWORD)

# Alert Configuration
animal_classes = ['Elephant', 'Gorilla', 'Hippopotamus', 'Leopard', 'Lion', 'Tiger', 'Wolf']

# Flags to ensure alerts are sent only once PER EVENT PER CAMERA
animal_alert_sent = {animal: {f"cam{i}": False for i in range(1, 5)} for animal in animal_classes}
traffic_alert_sent = {f"cam{i}": False for i in range(1, 3)}
audio_alert_played = False

# Load models
animal_model = YOLO(animal_model_path)
traffic_model = YOLO(traffic_model_path)

# === HELPERS ===

def send_email_alert(subject, body, image_path=None):
    try:
        if image_path:
            yag.send(to=EMAIL_RECEIVERS, subject=subject, contents=[body, image_path])
        else:
            yag.send(to=EMAIL_RECEIVERS, subject=body)
        print(f"[EMAIL SENT] {subject}")
    except Exception as e:
        print("[EMAIL FAILED]", e)

# def play_alert_audio():
#     threading.Thread(target=playsound, args=(ALERT_AUDIO_PATH,)).start()

def play_alert_audio():
    global audio_alert_played
    if not audio_alert_played:
        threading.Thread(target=play_sound_windows, args=(ALERT_AUDIO_PATH,)).start()
        audio_alert_played = True

def play_sound_windows(path):
    try:
        winsound.PlaySound(path, winsound.SND_ASYNC)
    except Exception as e:
        print(f"[AUDIO PLAYBACK FAILED]: {e}")

# === DETECTION ===

def process_animal_detection(frame, frame_id, cam_index):
    global animal_alert_sent
    results = animal_model(frame)
    annotated = results[0].plot()
    detected_animals = set()
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for box in results[0].boxes.data:
        cls_id = int(box[5])
        class_name = animal_model.names[cls_id]
        if class_name in animal_classes:
            animal_key = class_name
            cam_key = f"cam{cam_index + 1}"
            if not animal_alert_sent[animal_key][cam_key]:
                filename = f"animal_{animal_key}_cam{cam_index}_{frame_id}.jpg"
                cv2.imwrite(filename, frame)
                subject = f"ALERT: {animal_key} spotted"
                body = f"{animal_key} spotted at {timestamp_str} in Camera {cam_index+1}"
                send_email_alert(subject, body, filename)
                play_alert_audio()
                animal_alert_sent[animal_key][cam_key] = True

    return cv2.resize(annotated, (640, 360))

def process_traffic_detection(frame, frame_id, cam_index):
    global traffic_alert_sent, audio_alert_played, traffic_stopped_start

    results = traffic_model(frame)
    annotated = results[0].plot()
    boxes = results[0].boxes.xywh.cpu().numpy()
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    now = time.time()
    cam_key = f"cam{cam_index + 1}"

    is_stopped = False
    if len(boxes) > 0:
        current_positions = np.round(boxes[:, :2], 1)
        positions_history[cam_index].append(current_positions)
        if len(positions_history[cam_index]) > 10:
            positions_history[cam_index].pop(0)

        if len(positions_history[cam_index]) == 10:
            diffs = [np.sum(np.abs(positions_history[cam_index][i] - positions_history[cam_index][i + 1])) for i in range(9)]
            mean_movement = np.mean(diffs)
            if mean_movement < 1:
                is_stopped = True
                if traffic_stopped_start[cam_index] is None:
                    traffic_stopped_start[cam_index] = now
            else:
                traffic_stopped_start[cam_index] = None
    else:
        traffic_stopped_start[cam_index] = None

    if is_stopped and traffic_stopped_start[cam_index] is not None and (now - traffic_stopped_start[cam_index] > traffic_alert_delay):
        if not traffic_alert_sent[cam_key]:
            filename = f"traffic_stop_cam{cam_index}_{frame_id}.jpg"
            cv2.imwrite(filename, frame)
            subject = f"ALERT: Traffic Jam Camera {cam_index+5}"
            body = f"Traffic has been stagnant since {timestamp_str} in Camera {cam_index+5}"
            send_email_alert(subject, body, filename)
            play_alert_audio()
            traffic_alert_sent[cam_key] = True

    return cv2.resize(annotated, (640, 360))

# === MAIN ===

def main():
    cap_animal = [cv2.VideoCapture(path) for path in animal_video_paths]
    cap_traffic = [cv2.VideoCapture(path) for path in traffic_video_paths]
    frame_id = 0

    while True:
        frame_id += 1
        processed_frames = []

        # Animal feeds
        for i in range(4):
            ret, frame = cap_animal[i].read()
            if not ret:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                text = f"No Feed (Animal {i+1})"
                cv2.putText(frame, text, (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                processed_frames.append(frame)
            else:
                processed_frames.append(process_animal_detection(frame, frame_id, i))

        # Traffic feeds
        for i in range(2):
            ret, frame = cap_traffic[i].read()
            if not ret:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                text = f"No Feed (Traffic {i+1})"
                cv2.putText(frame, text, (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                processed_frames.append(frame)
            else:
                processed_frames.append(process_traffic_detection(frame, frame_id, i))

        # Display grid
        row1 = cv2.hconcat(processed_frames[0:3])
        row2 = cv2.hconcat(processed_frames[3:6])
        grid = cv2.vconcat([row1, row2])

        cv2.imshow("6-Camera Detection Grid", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in cap_animal + cap_traffic:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()