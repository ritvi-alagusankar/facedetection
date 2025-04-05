import os
import cv2
import dlib
import numpy as np
import time 

# --- Configuration ---
# Main dataset folder containing subfolders named after people
# Example structure:
# /path/to/your/dataset/
#  |- Person_A/
#  |  |- img1.jpg
#  |  |- img2.png
#  |- Person_B/
#  |  |- pic1.jpeg
# ...
dataset_path = "dataset/dataset"

# Size to resize cropped faces to (consistency is key for LBPH)
face_size = (100, 100)

# LBPH Confidence threshold (lower values mean higher confidence)
# Adjust this based on testing - might need values between 50 and 100
confidence_threshold = 50 # Adjusted slightly for potentially better testing results

# --- NEW: Testing Configuration ---
# Set to True to test the model on the dataset after training
activate_test = True

# --- Global Variables ---
# Load Dlib's pre-trained frontal face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Create LBPH Face Recognizer
# Note: You might need to install opencv-contrib-python
# pip install opencv-contrib-python
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Dictionary to map numerical labels to names
label_to_name = {}

# --- Function to Prepare Training Data and Train LBPH ---
def train_model(data_folder_path):
    """
    Scans the dataset folder, detects faces, prepares training data,
    and trains the LBPH recognizer.
    Returns True if training was successful, False otherwise.
    """
    print(f"[INFO] Preparing training data from: {data_folder_path}")
    faces = []
    labels = []
    current_label_id = 0
    global label_to_name # Allow modification of the global dictionary
    label_to_name.clear() # Clear mapping for potential re-training runs

    # Iterate through each person's folder in the dataset path
    for person_name in os.listdir(data_folder_path):
        person_folder_path = os.path.join(data_folder_path, person_name)

        # Skip if it's not a directory
        if not os.path.isdir(person_folder_path):
            continue

        print(f"[INFO] Processing images for: {person_name}")
        # Assign a numerical label to this person
        if person_name not in label_to_name.values():
            label_to_name[current_label_id] = person_name
            person_label = current_label_id
            current_label_id += 1
        else:
            # This case should ideally not happen if folder names are unique identifiers
            person_label = [id for id, name in label_to_name.items() if name == person_name][0]


        # Iterate through images in the person's folder
        image_count = 0
        for image_name in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_name)

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] Could not read image: {image_path}. Skipping.")
                continue

            # Convert to grayscale (Dlib detector and LBPH work better with grayscale)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Optional: Histogram Equalization can sometimes improve detection/recognition
            # gray = cv2.equalizeHist(gray)

            # Detect faces using Dlib detector
            # Using upsampling=1 consistent with potential quality needs during training data prep
            detected_faces = detector(gray, 1)

            if len(detected_faces) == 0:
                print(f"[WARNING] No face detected in: {image_path} for {person_name}. Skipping.")
                continue
            elif len(detected_faces) > 1:
                 print(f"[WARNING] Multiple faces ({len(detected_faces)}) detected in: {image_path} for {person_name}. Using the first one.")

            # Process only the first detected face for consistency
            face_rect = detected_faces[0]
            x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()

            # Ensure coordinates are valid
            x = max(0, x)
            y = max(0, y)
            if w <= 0 or h <= 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
                print(f"[WARNING] Invalid face bounds detected or calculated in: {image_path}. Skipping.")
                continue

            # Crop the face region from the grayscale image
            face_roi = gray[y:y+h, x:x+w]

            # Check if cropping resulted in an empty image (due to edge cases)
            if face_roi.size == 0:
                 print(f"[WARNING] Empty face ROI after cropping in: {image_path}. Skipping.")
                 continue

            # Resize the face ROI to the standard size
            try:
                resized_face = cv2.resize(face_roi, face_size, interpolation=cv2.INTER_AREA)
            except cv2.error as e:
                 print(f"[WARNING] Error resizing face ROI from {image_path}: {e}. Skipping.")
                 continue


            # Append the processed face and its corresponding label
            faces.append(resized_face)
            labels.append(person_label)
            image_count += 1

        print(f"[INFO] Added {image_count} face images for {person_name}.")


    if not faces:
        print("[ERROR] No faces found or processed for training. Check dataset path, image content, and warnings.")
        return False

    if len(label_to_name) < 2:
         print("[WARNING] Training requires images from at least two different people for the recognizer to be effective.")
         # Decide if you want to stop here or proceed with a potentially less useful model
         # return False # Option to stop if less than 2 people

    print(f"\n[INFO] Total faces prepared for training: {len(faces)}")
    print(f"[INFO] Total unique persons (labels): {len(label_to_name)}")
    print(f"[INFO] Label map: {label_to_name}")
    print("[INFO] Training LBPH model...")

    # Train the recognizer
    # Ensure labels is a NumPy array of type int32
    recognizer.train(faces, np.array(labels, dtype=np.int32))

    print("[INFO] LBPH model trained successfully.")
    # Optional: Save the trained model and label mapping for later use
    # recognizer.save("lbph_model.yml")
    # with open("label_map.txt", "w") as f:
    #  for label_id, name in label_to_name.items():
    #     f.write(f"{label_id}:{name}\n")
    # print("[INFO] Model and label map saved.")
    return True

# --- NEW: Function to Test Model on Dataset ---
def test_model_on_dataset(data_folder_path):
    """
    Tests the trained LBPH model on the images in the dataset folder
    and calculates the success rate.
    """
    print("\n" + "="*30)
    print("[INFO] Starting model testing on the dataset...")
    print("="*30)

    global label_to_name, recognizer, detector, face_size, confidence_threshold

    if not label_to_name:
        print("[ERROR] Label map is empty. Cannot perform testing. Was the model trained?")
        return

    # Create a reverse map for convenience (Name -> Label ID)
    name_to_label = {name: label_id for label_id, name in label_to_name.items()}

    total_tested_faces = 0
    correct_predictions = 0
    faces_detected_count = 0

    # Iterate through each person's folder in the dataset path
    for person_name in os.listdir(data_folder_path):
        person_folder_path = os.path.join(data_folder_path, person_name)

        if not os.path.isdir(person_folder_path):
            continue

        # Check if this person was part of the training set
        if person_name not in name_to_label:
            print(f"[WARNING] Person '{person_name}' found in dataset but not in the trained label map. Skipping testing for this person.")
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
                # Warning already given during training, maybe skip here unless verbose testing needed
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # gray = cv2.equalizeHist(gray) # Apply if used in training

            # Use the same detection settings as training ideally
            detected_faces = detector(gray, 1)

            if len(detected_faces) == 0:
                 # Can log this if needed: print(f"[TESTING-WARN] No face detected in: {image_path}")
                 continue # Cannot test recognition if no face is found

            faces_detected_count += 1
            detections_in_folder += 1

            # Use only the first detected face for consistency with training
            face_rect = detected_faces[0]
            x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()

            x = max(0, x)
            y = max(0, y)
            if w <= 0 or h <= 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
                continue # Skip invalid bounds

            face_roi = gray[y:y+h, x:x+w]
            if face_roi.size == 0:
                 continue # Skip empty ROI

            try:
                resized_face = cv2.resize(face_roi, face_size, interpolation=cv2.INTER_AREA)
            except cv2.error:
                continue # Skip resize errors


            # Perform prediction
            predicted_label_id, confidence = recognizer.predict(resized_face)

            total_tested_faces += 1 # Count this face as tested for recognition

            # Evaluate prediction
            prediction_is_correct = False
            if confidence < confidence_threshold and predicted_label_id == true_label_id:
                correct_predictions += 1
                correct_in_folder += 1
                prediction_is_correct = True

            # Optional: Print detailed results per image (can be very verbose)
            # predicted_name = label_to_name.get(predicted_label_id, "Unknown Label")
            # print(f"  - {image_name}: Predicted={predicted_name}({predicted_label_id}), Confidence={confidence:.2f}, Correct={prediction_is_correct}")

        print(f"[TESTING] Finished {person_name}: Found faces in {detections_in_folder}/{images_in_folder} images. Correctly recognized: {correct_in_folder}/{detections_in_folder}")


    print("\n" + "="*30)
    print("[INFO] Dataset Testing Summary")
    print("="*30)
    print(f"Total faces detected across all tested images: {faces_detected_count}")
    print(f"Total detected faces subjected to recognition: {total_tested_faces}")
    print(f"Total correct recognitions (within threshold): {correct_predictions}")

    if total_tested_faces > 0:
        accuracy = (correct_predictions / total_tested_faces) * 100
        print(f"Recognition Success Rate: {accuracy:.2f}%")
    else:
        print("Recognition Success Rate: N/A (No faces were successfully processed for recognition testing)")
    print("="*30 + "\n")


# --- Function for Live Face Recognition ---
def run_live_recognition():
    """
    Opens the camera, detects faces, and performs recognition using the trained model.
    """
    global label_to_name # Access the global mapping

    print("[INFO] Starting video stream for live recognition...")
    # Use 0 for default camera, or change if you have multiple cameras
    cap = cv2.VideoCapture(0)
    time.sleep(1.0) # Allow camera sensor to warm up

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Exiting.")
        return

    prev_time = 0

    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Failed to grab frame. Exiting.")
            break

        # For FPS calculation
        current_time = time.time()
        # Avoid division by zero on the first frame
        time_diff = current_time - prev_time
        fps = 1 / time_diff if time_diff > 0 else 0
        prev_time = current_time

        # Convert frame to grayscale for detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Optional: Equalize histogram
        # gray_frame = cv2.equalizeHist(gray_frame)

        # Detect faces in the current frame
        # Using upsampling=0 for potentially faster live performance
        detected_faces = detector(gray_frame, 0)

        # Loop over detected faces
        for face_rect in detected_faces:
            x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()

            # Ensure coordinates are valid and within frame boundaries
            x = max(0, x)
            y = max(0, y)
            # Check width/height before calculating bottom-right corner
            if w <= 0 or h <= 0:
                continue
            # Adjust width/height if they extend beyond frame boundaries
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            # Re-check after adjustment
            if w <= 0 or h <= 0:
                continue


            # Crop the face region *from the grayscale frame* for recognition
            face_roi = gray_frame[y:y+h, x:x+w]

             # Check if cropping resulted in an empty image (should be rare with checks above)
            if face_roi.size == 0:
                continue

            # Resize the cropped face to the standard size used for training
            try:
                resized_face = cv2.resize(face_roi, face_size, interpolation=cv2.INTER_AREA)
            except cv2.error:
                continue # Skip if resize fails


            # Perform prediction using the trained LBPH recognizer
            label_id, confidence = recognizer.predict(resized_face)

            # Get the name associated with the predicted label ID
            # Use "Unknown" if confidence is too high (poor match) or label unknown
            name = "Unknown" # Default
            print(confidence)
            if label_id in label_to_name: # Check if label exists first
                 if confidence > confidence_threshold:
                      name = label_to_name[label_id]
                      display_text = f"{name} ({confidence:.2f})"
                 else:
                      # Known person, but low confidence
                      display_text = f"Maybe {label_to_name[label_id]}? ({confidence:.2f})"
                      # Or just stick with "Unknown":
                      # name = "Unknown"
                      # display_text = f"{name} ({confidence:.2f})"
            else:
                 # Label ID not in our map (shouldn't happen if predict works correctly)
                 display_text = f"Unknown Label {label_id} ({confidence:.2f})"


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
    if not os.path.isdir(dataset_path):
         print(f"[ERROR] Dataset path not found or is not a directory: {dataset_path}")
         print("[INFO] Please set the 'dataset_path' variable correctly.")
    else:
        # 1. Train the model
        training_successful = train_model(dataset_path)

        if training_successful:
            # 2. Test the model (if activated)
            if activate_test:
                test_model_on_dataset(dataset_path)
            else:
                 print("[INFO] Skipping dataset testing as 'activate_test' is False.")

            # 3. Run live recognition (only if training was successful)
            # Decide if you always want to run live, or maybe only if not testing, etc.
            # Current logic: Run live recognition after training and optional testing.
            run_live_recognition()
        else:
            print("[ERROR] Model training failed. Cannot proceed to testing or live recognition.")
            