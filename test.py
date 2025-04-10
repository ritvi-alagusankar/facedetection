import os
import cv2
import dlib
import numpy as np
import time 
import json 
import argparse 
dataset_path = "png_dataset/png_dataset" 

face_size = (100, 100)

confidence_threshold = 100

model_filename = "lbph_model_augmented.yml"
label_map_filename = "label_map_augmented.json"


print("[INFO] Loading face detector...")
detector = dlib.get_frontal_face_detector()


recognizer = None
label_to_name = None

try:
    print(f"[INFO] Loading trained LBPH model from: {model_filename}")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_filename) 
    print(f"[INFO] Loading label map from: {label_map_filename}")
    with open(label_map_filename, 'r') as f:
        label_to_name_str_keys = json.load(f)
        label_to_name = {int(k): v for k, v in label_to_name_str_keys.items()}
        print(f"[INFO] Loaded label map: {label_to_name}")

except FileNotFoundError:
    print(f"[ERROR] Model file '{model_filename}' or label map '{label_map_filename}' not found.")
    print("[ERROR] Please run the train.py script first.")
    exit()
except Exception as e:
    print(f"[ERROR] Error loading model or label map: {e}")
    exit()

def test_model_on_dataset(data_folder_path):
    """
    Tests the loaded LBPH model on the images in the dataset folder
    and calculates the success rate. Uses globally loaded model/map.
    """
    print("\n" + "="*30)
    print("[INFO] Starting model testing on the dataset...")
    print("="*30)

    if not label_to_name: 
        print("[ERROR] Label map is not loaded. Cannot perform testing.")
        return

    if not os.path.isdir(data_folder_path):
         print(f"[ERROR] Dataset path for testing not found or is not a directory: {data_folder_path}")
         return

    name_to_label = {name: label_id for label_id, name in label_to_name.items()}

    total_tested_faces = 0
    correct_predictions = 0
    faces_detected_count = 0


    person_names = sorted(os.listdir(data_folder_path)) 
    for person_name in person_names:
        person_folder_path = os.path.join(data_folder_path, person_name)

        if not os.path.isdir(person_folder_path):
            continue

        # Check if this person was part of the training set (i.e., in the loaded map)
        if person_name not in name_to_label:
            print(f"[WARNING] Person '{person_name}' found in dataset but not in the loaded label map. Skipping testing for this person.")
            continue

        true_label_id = name_to_label[person_name]
        print(f"[TESTING] Evaluating images for: {person_name} (Expected Label: {true_label_id})")

        images_in_folder = 0
        detections_in_folder = 0
        correct_in_folder = 0

        # Iterate through images in the person's folder
        for image_name in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_name)
            images_in_folder += 1

            image = cv2.imread(image_path)
            if image is None:
                continue # Skip unreadable images

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            try:
                detected_faces = detector(gray, 1)
            except Exception as e:
                 print(f"[TESTING-ERROR] Face detection failed for {image_path}: {e}. Skipping.")
                 continue

            if len(detected_faces) == 0:
                continue 

            faces_detected_count += 1
            detections_in_folder += 1

            face_rect = detected_faces[0]
            x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()

            x = max(0, x)
            y = max(0, y)
            if w <= 0 or h <= 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
                print(f"[TESTING-WARN] Invalid face bounds in: {image_path}. Skipping.")
                continue 

            face_roi = gray[y:y+h, x:x+w]
            if face_roi.size == 0:
                print(f"[TESTING-WARN] Empty face ROI in: {image_path}. Skipping.")
                continue 

            try:
                resized_face = cv2.resize(face_roi, face_size, interpolation=cv2.INTER_AREA)
            except cv2.error:
                 print(f"[TESTING-WARN] Resize error for face in: {image_path}. Skipping.")
                 continue 


            predicted_label_id, confidence = recognizer.predict(resized_face)

            total_tested_faces += 1 
  
            prediction_is_correct = False
        
            if confidence < confidence_threshold and predicted_label_id == true_label_id:
                correct_predictions += 1
                correct_in_folder += 1
                prediction_is_correct = True
        
            elif predicted_label_id == true_label_id:
                 print(f"  - {image_name}: Correct Label ({person_name}) but HIGH confidence ({confidence:.2f}) - Not counted.")
            # Handle case where prediction is wrong label (confidence doesn't matter as much here for accuracy metric)
            else:
                predicted_name = label_to_name.get(predicted_label_id, "Unknown Label")
                print(f"  - {image_name}: Incorrectly Predicted as {predicted_name}({predicted_label_id}) with confidence {confidence:.2f}")

        print(f"[TESTING] Finished {person_name}: Found faces in {detections_in_folder}/{images_in_folder} images. Correctly recognized (below threshold): {correct_in_folder}/{detections_in_folder}")


    print("\n" + "="*30)
    print("[INFO] Dataset Testing Summary")
    print("="*30)
    print(f"Total faces detected across all tested images: {faces_detected_count}")
    print(f"Total detected faces subjected to recognition: {total_tested_faces}")
    print(f"Total correct recognitions (within threshold {confidence_threshold}): {correct_predictions}")

    if total_tested_faces > 0:
        accuracy = (correct_predictions / total_tested_faces) * 100
        print(f"Recognition Success Rate: {accuracy:.2f}%")
    else:
        print("Recognition Success Rate: N/A (No faces were successfully processed for recognition testing)")
    print("="*30 + "\n")


def run_live_recognition():
    """
    Opens the camera, detects faces, and performs recognition using the loaded model.
    Uses globally loaded model/map.
    """
    if not label_to_name: 
        print("[ERROR] Label map is not loaded. Cannot run live recognition.")
        return

    print("[INFO] Starting video stream for live recognition...")
    cap = cv2.VideoCapture(0)
    time.sleep(1.0)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Exiting.")
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Failed to grab frame. Exiting.")
            break

        current_time = time.time()
        time_diff = current_time - prev_time
        fps = 1 / time_diff if time_diff > 0 else 0
        prev_time = current_time

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            detected_faces = detector(gray_frame, 0)
        except Exception as e:
            print(f"[LIVE-ERROR] Face detection failed: {e}")
            detected_faces = [] # Continue with empty list


        # Loop over detected faces
        for face_rect in detected_faces:
            x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()


            x = max(0, x)
            y = max(0, y)
 
            if w <= 0 or h <= 0:
                continue
    
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
     
            if w <= 0 or h <= 0:
                continue

            face_roi = gray_frame[y:y+h, x:x+w]

            if face_roi.size == 0:
                continue

            try:
                resized_face = cv2.resize(face_roi, face_size, interpolation=cv2.INTER_AREA)
            except cv2.error:
                 continue 


            # Perform prediction using the loaded LBPH recognizer
            label_id, confidence = recognizer.predict(resized_face)


            name = "Unknown" # Default
            display_text = ""

            # LBPH confidence is distance - LOWER is better.
            if confidence < confidence_threshold:
                if label_id in label_to_name:
                     name = label_to_name[label_id]
                     display_text = f"{name} ({confidence:.2f})"
                else:
                    # Recognizer predicted a label ID not in our map (should be rare if trained correctly)
                    display_text = f"Known? ID:{label_id} ({confidence:.2f})"
            else:
                name = label_to_name[label_id]
                display_text = f"Unknown maybe {name}? ({confidence:.2f})"


            # Draw bounding box around the face on the *original color* frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put the predicted name (and confidence) text above the bounding box
            text_y = y - 10 if y - 10 > 10 else y + h + 20 # Adjust position if too close to top
            cv2.putText(frame, display_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display FPS on frame (optional)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the resulting frame
        cv2.imshow("Live Face Recognition (Press 'q' to quit)", frame)

        # Check for exit key ('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    print("[INFO] Stopping video stream and cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application finished.")

# --- Main Execution ---
if __name__ == "__main__":
    # Check if model was loaded successfully before proceeding
    if recognizer is None or label_to_name is None:
        print("[ERROR] Model or label map failed to load. Exiting.")
    else:
        # --- Add Argument Parsing to Choose Mode ---
        parser = argparse.ArgumentParser(description="Run face recognition: either test on dataset or live from camera.")
        parser.add_argument("--mode", type=str, default="live", choices=["live", "test"],
                            help="Operation mode: 'live' for camera feed, 'test' for dataset evaluation (default: live)")
        args = parser.parse_args()

        if args.mode == "test":
            # Run testing on the dataset
            test_model_on_dataset(dataset_path)
        elif args.mode == "live":
            # Run live recognition
            run_live_recognition()
        else:
            # Should not happen due to 'choices' in argparse, but good practice
            print(f"[ERROR] Invalid mode specified: {args.mode}. Choose 'live' or 'test'.")