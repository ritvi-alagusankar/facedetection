import os
import cv2
import dlib
import numpy as np
import time
import json
import argparse

# --- Configuration ---
DATASET_PATH = "png_dataset/png_dataset"
FACE_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 100

# Tracking parameters
TRACK_MATCH_DISTANCE_THRESHOLD = 50    # Maximum pixel distance for face track matching
STABLE_COUNT_THRESHOLD = 5             # Number of consecutive frames for a stable prediction
CONFIDENCE_TOLERANCE = 10              # Allowed variation in confidence
MAX_MISSED_FRAMES = 10                 # Maximum missed frames before a track is removed

MODEL_FILENAME = "lbph_model_augmented.yml"
LABEL_MAP_FILENAME = "label_map_augmented.json"


# --- Model & Detector Loading ---
def load_models():
    """Loads the dlib face detector, LBPH recognizer, and label map."""
    print("[INFO] Loading face detector...")
    detector = dlib.get_frontal_face_detector()
    try:
        print(f"[INFO] Loading trained LBPH model from: {MODEL_FILENAME}")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_FILENAME)
        print(f"[INFO] Loading label map from: {LABEL_MAP_FILENAME}")
        with open(LABEL_MAP_FILENAME, 'r') as f:
            label_to_name_str_keys = json.load(f)
            label_to_name = {int(k): v for k, v in label_to_name_str_keys.items()}
        print(f"[INFO] Loaded label map: {label_to_name}")
    except FileNotFoundError:
        print(f"[ERROR] Model file '{MODEL_FILENAME}' or label map '{LABEL_MAP_FILENAME}' not found.")
        exit(1)
    except Exception as e:
        print(f"[ERROR] Error loading model or label map: {e}")
        exit(1)
    return detector, recognizer, label_to_name


# Load models globally
detector, recognizer, label_to_name = load_models()


# --- Helper Functions ---
def safe_resize(face_roi, size=FACE_SIZE):
    """Resizes the face region safely using INTER_AREA. Returns None on error."""
    try:
        return cv2.resize(face_roi, size, interpolation=cv2.INTER_AREA)
    except cv2.error:
        return None


def extract_face(image, detector, min_size=10):
    """
    Detects faces in the image and extracts the first valid face ROI.
    
    Returns:
        A tuple (resized_face, bbox) or (None, None) if no valid face is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        faces = detector(gray, 1)
    except Exception as e:
        print(f"[ERROR] Face detection failed: {e}")
        return None, None

    if not faces:
        return None, None

    # Use the first detected face
    face_rect = faces[0]
    x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
    x, y = max(0, x), max(0, y)
    img_h, img_w = gray.shape
    if w <= 0 or h <= 0 or x + w > img_w or y + h > img_h or w < min_size or h < min_size:
        print("[WARNING] Invalid face bounds detected.")
        return None, None

    face_roi = gray[y:y+h, x:x+w]
    if face_roi.size == 0:
        return None, None

    resized_face = safe_resize(face_roi)
    return resized_face, (x, y, w, h) if resized_face is not None else (None, None)


def get_centroid(bbox):
    """Calculates the centroid of a given bounding box."""
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def euclidean_distance(p1, p2):
    """Returns the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# --- Dataset Testing ---
def test_model_on_dataset(data_folder_path, detector, recognizer, label_to_name):
    """
    Evaluates the LBPH model on images in the dataset folder.
    """
    print("\n" + "="*30)
    print("[INFO] Starting model testing on dataset...")
    print("="*30)

    if not label_to_name:
        print("[ERROR] Label map not loaded.")
        return
    if not os.path.isdir(data_folder_path):
        print(f"[ERROR] Invalid dataset path: {data_folder_path}")
        return

    # Create a mapping from name to label
    name_to_label = {v: k for k, v in label_to_name.items()}
    total_tested_faces = 0
    correct_predictions = 0
    total_detected_faces = 0

    for person_name in sorted(os.listdir(data_folder_path)):
        person_folder = os.path.join(data_folder_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        if person_name not in name_to_label:
            print(f"[WARNING] Person '{person_name}' not found in label map. Skipping.")
            continue

        expected_label = name_to_label[person_name]
        print(f"[TESTING] Evaluating images for: {person_name} (Expected Label: {expected_label})")
        images = os.listdir(person_folder)
        correct_in_folder = 0

        for image_file in images:
            image_path = os.path.join(person_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue

            resized_face, bbox = extract_face(image, detector)
            if resized_face is None:
                continue

            total_detected_faces += 1
            try:
                label_id, confidence = recognizer.predict(resized_face)
            except Exception as e:
                print(f"[TESTING-ERROR] Prediction error on {image_path}: {e}")
                continue

            total_tested_faces += 1
            if confidence < CONFIDENCE_THRESHOLD and label_id == expected_label:
                correct_predictions += 1
                correct_in_folder += 1
            elif label_id == expected_label:
                print(f"  - {image_file}: Correct label but high confidence ({confidence:.2f}) - Not counted.")
            else:
                predicted_name = label_to_name.get(label_id, "Unknown")
                print(f"  - {image_file}: Incorrect prediction: {predicted_name} ({label_id}), confidence {confidence:.2f}")

        print(f"[TESTING] {person_name}: Correct predictions: {correct_in_folder}/{len(images)}")

    print("\n" + "="*30)
    print("[INFO] Dataset Testing Summary")
    print("="*30)
    print(f"Total detected faces: {total_detected_faces}")
    print(f"Total faces processed: {total_tested_faces}")
    print(f"Total correct recognitions: {correct_predictions}")
    if total_tested_faces:
        accuracy = (correct_predictions / total_tested_faces) * 100
        print(f"Recognition Success Rate: {accuracy:.2f}%")
    else:
        print("Recognition Success Rate: N/A")
    print("="*30 + "\n")


# --- Live Recognition with Tracking ---
def process_detections(frame, detector, recognizer, label_to_name):
    """
    Processes the current video frame: detects faces,
    performs prediction, and returns a list of detection dictionaries.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        faces = detector(gray_frame, 0)
    except Exception as e:
        print(f"[LIVE-ERROR] Face detection error: {e}")
        return []

    detections = []
    img_h, img_w = frame.shape[:2]
    for face_rect in faces:
        # Clamp coordinates to image bounds
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        w_adj, h_adj = x2 - x1, y2 - y1

        if w_adj <= 10 or h_adj <= 10:
            continue

        face_roi = gray_frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        resized_face = safe_resize(face_roi)
        if resized_face is None:
            continue

        try:
            label_id, confidence = recognizer.predict(resized_face)
        except Exception as e:
            print(f"[LIVE-ERROR] Prediction error: {e}")
            continue

        detections.append({
            'bbox': (x1, y1, w_adj, h_adj),
            'centroid': get_centroid((x1, y1, w_adj, h_adj)),
            'label': label_id,
            'confidence': confidence,
            'name': label_to_name.get(label_id, "Unknown")
        })
    return detections


def update_tracks(detections, tracks, next_track_id):
    """
    Updates or creates face tracks based on current detections.
    
    Returns:
        A tuple (updated_tracks, next_track_id)
    """
    # Mark all current tracks as not updated this frame
    for track in tracks:
        track['updated'] = False

    matched_track_ids = set()
    for detection in detections:
        detection_centroid = detection['centroid']
        best_track = None
        min_dist = TRACK_MATCH_DISTANCE_THRESHOLD

        for track in tracks:
            if track['id'] in matched_track_ids:
                continue
            dist = euclidean_distance(get_centroid(track['bbox']), detection_centroid)
            if dist < min_dist:
                min_dist = dist
                best_track = track

        if best_track is not None:
            # Update existing track
            best_track['bbox'] = detection['bbox']
            best_track['updated'] = True
            matched_track_ids.add(best_track['id'])
            # Increase stability count if prediction remains consistent
            if best_track['label'] == detection['label'] and abs(best_track['confidence'] - detection['confidence']) < CONFIDENCE_TOLERANCE:
                best_track['stable_count'] += 1
            else:
                best_track['stable_count'] = 1
            best_track['label'] = detection['label']
            best_track['confidence'] = detection['confidence']
            best_track['name'] = detection['name']
            best_track['missed_frames'] = 0
        else:
            # Create new track
            new_track = {
                'id': next_track_id,
                'bbox': detection['bbox'],
                'label': detection['label'],
                'confidence': detection['confidence'],
                'name': detection['name'],
                'stable_count': 1,
                'missed_frames': 0,
                'updated': True
            }
            tracks.append(new_track)
            next_track_id += 1

    # Update missed frames and remove stale tracks
    active_tracks = []
    for track in tracks:
        if not track.get('updated'):
            track['missed_frames'] += 1
        if track['missed_frames'] <= MAX_MISSED_FRAMES:
            active_tracks.append(track)
    return active_tracks, next_track_id


def run_live_recognition(detector, recognizer, label_to_name):
    """
    Opens the camera, performs detection, recognition, and tracking,
    and displays the live prediction results.
    """
    if not label_to_name:
        print("[ERROR] Label map not loaded.")
        return

    print("[INFO] Starting live video stream...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    # Warm up the camera sensor
    time.sleep(1.0)
    tracks = []
    next_track_id = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARNING] Frame grab failed, retrying...")
            time.sleep(0.5)
            continue

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time - prev_time > 1e-3 else 0
        prev_time = current_time

        # Process detections on the current frame
        detections = process_detections(frame, detector, recognizer, label_to_name)
        tracks, next_track_id = update_tracks(detections, tracks, next_track_id)

        # Display tracking results with refined display logic
        for track in tracks:
            x, y, w, h = track['bbox']
            if track['stable_count'] >= STABLE_COUNT_THRESHOLD:
                if track['confidence'] < CONFIDENCE_THRESHOLD:
                    display_text = f"{track['name']} ({track['confidence']:.2f})"
                    box_color = (0, 255, 0)  # Confident match
                else:
                    display_text = f"Unknown ({track['confidence']:.2f})"
                    box_color = (0, 165, 255)  # Low confidence
            else:
                display_text = f"({track['name']}?) ({track['confidence']:.2f})"
                box_color = (255, 165, 0)  # Tentative

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(frame, display_text, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Tracks: {len(tracks)}", (frame.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[INFO] Stopping video stream and cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Live recognition terminated.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run face recognition either live from camera or for dataset testing."
    )
    parser.add_argument("--mode", type=str, choices=["live", "test"], default="live",
                        help="Mode: 'live' for camera or 'test' for dataset evaluation.")
    args = parser.parse_args()

    if args.mode == "test":
        test_model_on_dataset(DATASET_PATH, detector, recognizer, label_to_name)
    else:
        run_live_recognition(detector, recognizer, label_to_name)
