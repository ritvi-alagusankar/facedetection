import cv2
import dlib

# Load HOG + SVM-based face detector from dlib
hog_detector = dlib.get_frontal_face_detector()

# Open webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
else:
    print("Webcam is successfully opened!")


print("Press 'q' to exit.")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to grayscale and equalize histogram
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect faces
    detected_faces = hog_detector(gray)

    # Draw bounding boxes
    for face in detected_faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Live Face Detection (Press 'q' to quit)", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


import cv2
import dlib
import os
import numpy as np
from PIL import Image # Used for easier image handling during training

# --- Configuration ---
dataset_path = 'dataset' # Path to the folder containing training images
output_trainer_file = 'trainer/trainer.yml' # Where to save the trained model
confidence_threshold = 75 # LBPH confidence threshold (lower value means better match) Adjust as needed.

# --- Initialization ---
# Initialize dlib's HOG-based face detector
print("[INFO] Initializing dlib face detector...")
detector = dlib.get_frontal_face_detector()

# Initialize OpenCV's LBPH Face Recognizer
# Make sure the directory for the trainer file exists
output_dir = os.path.dirname(output_trainer_file)
if not os.path.exists(output_dir):
    print(f"[INFO] Creating directory: {output_dir}")
    os.makedirs(output_dir)

# Create LBPH Recognizer instance
# Note: cv2.face was moved in OpenCV 4. If you have issues, try cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("[INFO] Initializing LBPH Recognizer (cv2.face.LBPHFaceRecognizer_create)...")
except AttributeError:
    try:
        recognizer = cv2.face_LBPHFaceRecognizer.create() # Older OpenCV versions
        print("[INFO] Initializing LBPH Recognizer (cv2.face_LBPHFaceRecognizer.create)...")
    except AttributeError:
        print("[ERROR] Could not find LBPHFaceRecognizer_create. Check your OpenCV installation and version (ensure opencv-contrib-python is installed if needed).")
        exit()


# --- Training Phase ---

# Function to get images and labels from the dataset directory
def get_images_and_labels(path):
    print(f"[INFO] Reading training images from: {path}")
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')] # Ignore hidden files
    face_samples = []
    ids = []
    names = {} # Dictionary to map ID -> Name
    current_id = 0
    label_ids = {} # Dictionary to map Name -> ID

    if not image_paths:
        print(f"[ERROR] No images found in the dataset path: {path}")
        print("Please ensure the 'dataset' folder exists and contains images named like 'SubjectName.UserID.ImageNumber.jpg'")
        return [], [], {}

    print(f"[INFO] Found {len(image_paths)} potential image files.")

    processed_image_count = 0
    face_detected_count = 0

    for image_path in image_paths:
        try:
            # Use PIL to open image (handles various formats robustly)
            pil_img = Image.open(image_path).convert('L') # Convert to grayscale
            img_numpy = np.array(pil_img, 'uint8')

            # Extract name and ID from filename (e.g., "Alice.1.1.jpg")
            filename = os.path.basename(image_path)
            parts = filename.split('.')
            if len(parts) < 3:
                print(f"[WARNING] Skipping file with unexpected format: {filename}. Expected 'Name.ID.Num.jpg'")
                continue

            name = parts[0]
            try:
                img_id = int(parts[1])
            except ValueError:
                print(f"[WARNING] Skipping file - could not parse ID as integer from: {filename}. Expected 'Name.ID.Num.jpg'")
                continue

            # Assign unique integer ID if needed (optional, using ID from filename is preferred)
            # if name not in label_ids:
            #     label_ids[name] = current_id
            #     names[current_id] = name
            #     current_id += 1
            # img_id = label_ids[name]

            # Store Name mapping if not already present for this ID
            if img_id not in names:
                 names[img_id] = name
                 print(f"[INFO] Mapping ID {img_id} to Name '{name}'")


            processed_image_count += 1

            # Detect faces using dlib HOG detector
            # The '1' indicates upsampling the image 1 time, making it larger and potentially finding more faces.
            dets = detector(img_numpy, 1)
            # print(f"[DEBUG] Detected {len(dets)} faces in {filename}")

            if not dets:
                 print(f"[WARNING] No face detected in training image: {filename}")

            for k, d in enumerate(dets):
                # dlib returns a rectangle object (left, top, right, bottom)
                x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()

                # Ensure coordinates are valid
                if x1 < 0 or y1 < 0 or x2 > img_numpy.shape[1] or y2 > img_numpy.shape[0]:
                    print(f"[WARNING] Skipping face with out-of-bounds coordinates in {filename}")
                    continue
                if x1 >= x2 or y1 >= y2:
                     print(f"[WARNING] Skipping face with invalid dimensions (w or h <= 0) in {filename}")
                     continue


                # Extract the face ROI (Region of Interest)
                face_roi = img_numpy[y1:y2, x1:x2]

                # --- Optional: Display face being trained ---
                # cv2.imshow("Training on image...", face_roi)
                # cv2.waitKey(10) # wait 10ms
                # --- End Optional ---

                face_samples.append(face_roi)
                ids.append(img_id)
                face_detected_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to process image {image_path}: {e}")

    # cv2.destroyAllWindows() # Close the optional training window
    print(f"[INFO] Processed {processed_image_count} images.")
    print(f"[INFO] Extracted {face_detected_count} faces for training.")

    if not face_samples:
        print("[ERROR] No faces were extracted from the dataset. Cannot train the recognizer.")
        print("Please check dataset images and dlib detection.")
        return [], [], {}

    return face_samples, ids, names

# --- Train the Recognizer ---
if not os.path.exists(dataset_path):
    print(f"[ERROR] Dataset path does not exist: {dataset_path}")
    print("Please create the 'dataset' folder and add training images.")
    exit()

faces, ids, names_map = get_images_and_labels(dataset_path)

if faces and ids:
    print("\n[INFO] Training LBPH recognizer...")
    recognizer.train(faces, np.array(ids))
    # Save the trained model
    recognizer.write(output_trainer_file)
    print(f"[INFO] Trained model saved to {output_trainer_file}")
    print(f"[INFO] Names mapped: {names_map}")
else:
    print("[ERROR] Training failed due to lack of training data.")
    exit()


# --- Real-time Detection and Identification ---

# Load the trained recognizer if not just trained (useful if running separately)
# recognizer.read(output_trainer_file)

print("\n[INFO] Starting video capture...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()

print("[INFO] Press 'q' to quit.")

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame from webcam.")
        break

    # Convert frame to grayscale for detection and recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using dlib HOG detector
    # Again, '1' is the upsample factor. Use '0' for faster but potentially less sensitive detection.
    dets = detector(gray, 1)
    # print(f"[DEBUG] Detected {len(dets)} faces in current frame.")


    # Loop over the detected faces
    for k, d in enumerate(dets):
        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()

        # Ensure coordinates are valid before drawing/cropping
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > frame.shape[1]: x2 = frame.shape[1]
        if y2 > frame.shape[0]: y2 = frame.shape[0]

        if x1 >= x2 or y1 >= y2: # Skip invalid boxes
            continue

        # Draw rectangle around the face in the original color frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- Identification using LBPH ---
        # Extract the grayscale face ROI for recognition
        face_roi_gray = gray[y1:y2, x1:x2]

        try:
            # Predict the ID and confidence
            id_, confidence = recognizer.predict(face_roi_gray)

            # Check if the prediction is confident enough (lower confidence is better)
            if confidence < confidence_threshold:
                # Map the predicted ID back to a name
                name = names_map.get(id_, "Unknown") # Get name or default to Unknown
                display_text = f"{name} ({confidence:.2f})"
                text_color = (255, 255, 255) # White text for known faces
            else:
                # If confidence is too high, label as Unknown
                name = "Unknown"
                display_text = f"{name} ({confidence:.2f})"
                text_color = (0, 0, 255) # Red text for unknown faces

            # Display the name and confidence
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10 # Position text above or below box
            cv2.putText(frame, display_text, (x1, text_y), font, 0.7, text_color, 2)

        except cv2.error as e:
            # This can happen if the face ROI is too small or invalid
            print(f"[WARNING] Could not predict face ROI: {e}")
            cv2.putText(frame, "Error", (x1, y1 - 5), font, 0.7, (0, 0, 255), 2)
        except Exception as e:
            print(f"[ERROR] Unexpected error during prediction: {e}")


    # Display the resulting frame
    cv2.imshow('Real-Time Face Detection and Identification (dlib HOG + LBPH)', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("\n[INFO] Cleaning up...")
cap.release()
cv2.destroyAllWindows()
print("[INFO] Application finished.")