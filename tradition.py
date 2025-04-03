import cv2
import dlib
import os
import numpy as np
from PIL import Image

# --- Configuration ---
dataset_path = 'dataset'  # Path to the folder containing training images
output_trainer_file = 'trainer/trainer.yml'  # Where to save the trained model
confidence_threshold = 75  # LBPH confidence threshold (lower value means better match)

# --- Initialization ---
print("[INFO] Initializing dlib face detector...")
detector = dlib.get_frontal_face_detector()

# Ensure trainer directory exists
output_dir = os.path.dirname(output_trainer_file)
os.makedirs(output_dir, exist_ok=True)

# Initialize OpenCV's LBPH Face Recognizer
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
except AttributeError:
    print("[ERROR] Could not initialize LBPH Recognizer. Ensure opencv-contrib-python is installed.")
    exit()

# --- Training Phase ---
def get_images_and_labels(path):
    print(f"[INFO] Reading training images from: {path}")
    face_samples, ids, names = [], [], {}
    current_id = 0
    
    if not os.path.exists(path):
        print(f"[ERROR] Dataset path does not exist: {path}")
        return [], [], {}
    
    for person_name in os.listdir(path):
        person_path = os.path.join(path, person_name)
        if not os.path.isdir(person_path):
            continue  # Skip non-folder files
        
        person_id = current_id
        names[person_id] = person_name
        current_id += 1
        print(f"[INFO] Mapping ID {person_id} to Name '{person_name}'")

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            try:
                pil_img = Image.open(image_path).convert('L')
                img_numpy = np.array(pil_img, 'uint8')
                
                if img_numpy is None or img_numpy.size == 0:
                    print(f"[ERROR] Cannot read image: {image_path}")
                    continue
                
                dets = detector(img_numpy, 1)
                if not dets:
                    print(f"[WARNING] No face detected in: {image_name}")
                    continue
                
                for d in dets:
                    x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
                    if x1 >= x2 or y1 >= y2:
                        print(f"[WARNING] Skipping invalid face detection in {image_name}")
                        continue
                    
                    face_roi = img_numpy[y1:y2, x1:x2]
                    if face_roi.size == 0:
                        print(f"[WARNING] Skipping empty face ROI in {image_name}")
                        continue
                    
                    face_samples.append(face_roi)
                    ids.append(person_id)
            except Exception as e:
                print(f"[ERROR] Failed to process image {image_path}: {e}")
    
    return face_samples, ids, names

faces, ids, names_map = get_images_and_labels(dataset_path)
if faces and ids:
    print("[INFO] Training LBPH recognizer...")
    recognizer.train(faces, np.array(ids))
    recognizer.write(output_trainer_file)
    print(f"[INFO] Trained model saved to {output_trainer_file}")
else:
    print("[ERROR] Training failed due to lack of training data.")
    exit()

# --- Real-time Detection ---
print("[INFO] Starting video capture...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame from webcam.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    
    for d in dets:
        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        if x1 >= x2 or y1 >= y2:
            continue
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        face_roi_gray = gray[y1:y2, x1:x2]
        
        try:
            id_, confidence = recognizer.predict(face_roi_gray)
            name = names_map.get(id_, "Unknown")
            text_color = (255, 255, 255) if confidence < confidence_threshold else (0, 0, 255)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x1, y1 - 10), font, 0.7, text_color, 2)
        except Exception as e:
            print(f"[WARNING] Recognition failed: {e}")
    
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Application finished.")