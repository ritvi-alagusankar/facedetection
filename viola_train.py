# Processing best_train.py
import os
import cv2
# import dlib # Removed dlib import
import numpy as np
import json
import time
import random
import shutil  # Added for copying original files
from sklearn.model_selection import train_test_split
from collections import Counter

print(f"Using OpenCV version: {cv2.__version__}")
# print(f"Using dlib version: {dlib.__version__}") # Removed dlib version print
print(f"Using Numpy version: {np.__version__}")
try:
    import sklearn
    print(f"Using scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("[ERROR] scikit-learn not found. Please install it: pip install scikit-learn")
    exit()

# --- Configuration ---
dataset_path = "png_dataset/png_dataset"
face_size = (100, 100) # Size for cropped training faces
model_filename = "lbph_model_train_augmented.yml" # Keep LBPH model name for now
label_map_filename = "label_map_final.json" # Keep label map name for now
train_data_save_path = "viola_train_data_augmented" # Changed save path prefix
validation_data_save_path = "viola_validation_data_original_images" # Changed save path prefix

# Augmentation Settings (applied ONLY to training data AFTER split)
AUGMENT_TRAIN_DATA = True
AUG_FLIP_PROB = 0.5
AUG_BRIGHTNESS_CONTRAST_PROB = 0.4
AUG_CONTRAST_RANGE = (0.85, 1.15)
AUG_BRIGHTNESS_RANGE = (-15, 15)

TRAIN_TEST_SPLIT_RATIO = 0.25 # 25% for validation/test
RANDOM_STATE_SPLIT = 42       # For reproducible splits

print("[INFO] Loading Viola-Jones face detector...")
# detector = dlib.get_frontal_face_detector() # Replaced Dlib detector
# Load the Haar cascade file
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(haar_cascade_path):
     print(f"[ERROR] Haar cascade file not found at: {haar_cascade_path}")
     print("[INFO] Ensure OpenCV is installed correctly and the cascade file exists.")
     exit()
detector = cv2.CascadeClassifier(haar_cascade_path)
if detector.empty():
    print("[ERROR] Failed to load Haar cascade classifier.")
    exit()
print("[INFO] Viola-Jones face detector loaded.")


def adjust_brightness_contrast(image, alpha_range, beta_range):
    alpha = random.uniform(alpha_range[0], alpha_range[1])
    beta = random.uniform(beta_range[0], beta_range[1])
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# --- Function to save CROPPED face datasets (Used for Training Data) ---
def save_cropped_face_dataset(data, labels, base_path, label_map):
    """Saves a list of CROPPED face images and labels."""
    print(f"[INFO] Saving CROPPED face dataset to: {base_path}")
    if os.path.exists(base_path):
        shutil.rmtree(base_path) # Remove old data if exists
    os.makedirs(base_path, exist_ok=True)
    person_image_count = {}

    for i, (image, label) in enumerate(zip(data, labels)):
        person_name = label_map.get(label, f"unknown_label_{label}")
        person_dir = os.path.join(base_path, person_name)
        os.makedirs(person_dir, exist_ok=True)

        current_count = person_image_count.get(label, 0)
        # Save as PNG, as these are processed numpy arrays
        filename = f"face_{current_count:04d}.png"
        save_path = os.path.join(person_dir, filename)

        try:
            cv2.imwrite(save_path, image) # image is the cropped face numpy array
            person_image_count[label] = current_count + 1
        except Exception as e:
            print(f"[ERROR] Could not save cropped face image {save_path}: {e}")

    print(f"[INFO] Finished saving CROPPED face dataset to {base_path}")
    print(f"   - Total cropped faces saved: {len(data)}")

# --- Function to save ORIGINAL image files (Used for Validation/Test Data) ---
def save_original_images(data_items, labels, base_path, label_map):
    """Copies original image files based on paths stored in data_items."""
    print(f"[INFO] Saving ORIGINAL image files to: {base_path}")
    if os.path.exists(base_path):
        shutil.rmtree(base_path) # Remove old data if exists
    os.makedirs(base_path, exist_ok=True)
    saved_count = 0
    copy_errors = 0

    for i, (item, label) in enumerate(zip(data_items, labels)):
        original_path = item.get('path') # Get path from the dictionary
        if not original_path:
            print(f"[ERROR] Original path missing for item index {i}, label {label}. Skipping.")
            copy_errors += 1
            continue

        person_name = label_map.get(label, f"unknown_label_{label}")
        person_dir = os.path.join(base_path, person_name)
        os.makedirs(person_dir, exist_ok=True)

        original_filename = os.path.basename(original_path)
        dest_path = os.path.join(person_dir, original_filename)

        try:
            # Copy the original file
            # Make sure the source file still exists
            if os.path.exists(original_path):
                 shutil.copy2(original_path, dest_path) # copy2 preserves more metadata
                 saved_count += 1
            else:
                 print(f"[ERROR] Original file not found, cannot copy: {original_path}")
                 copy_errors +=1

        except Exception as e:
            print(f"[ERROR] Could not copy original file {original_path} to {dest_path}: {e}")
            copy_errors += 1

    print(f"[INFO] Finished saving ORIGINAL image files to {base_path}")
    print(f"   - Total original files copied: {saved_count}")
    if copy_errors > 0:
        print(f"   - Errors encountered during copy: {copy_errors}")


# --- Main Data Processing Function (Modified Workflow) ---
def process_balance_split_augment_train(data_folder_path):
    print(f"[INFO] Starting Data Processing Workflow:")
    # Workflow steps updated slightly for clarity
    print(f"       1. Detect Original Faces & Store Paths")
    print(f"       2. Balance (Originals + Paths)")
    print(f"       3. Split into Train/Test (Originals + Paths)")
    print(f"       4. Augment TRAINING Faces")
    print(f"       5. Save Augmented Train Faces & Original Test Images")
    print(f"       6. Train Model on Augmented Train Faces")
    print("-" * 40)
    start_time = time.time()

    # --- Phase 1: Detect ORIGINAL Faces & Store Paths ---
    print("[INFO] Phase 1: Detecting original faces & storing paths...")
    # Now stores dicts: {'face': resized_face_array, 'path': original_image_path}
    original_person_data_with_paths = {}
    label_to_name_map = {}
    current_label_id = 0
    original_faces_detected = 0
    face_detection_failures = 0

    try:
        person_names = sorted([d for d in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, d))])
    except FileNotFoundError:
        print(f"[ERROR] Dataset directory not found: {data_folder_path}")
        return None, None
    if not person_names:
        print(f"[ERROR] No person subdirectories found in: {data_folder_path}")
        return None, None

    for person_name in person_names:
        person_folder_path = os.path.join(data_folder_path, person_name)
        print(f"   - Processing images for: {person_name}")

        # Label Assignment
        existing_label = next((id for id, name in label_to_name_map.items() if name == person_name), None)
        if existing_label is not None:
            person_label = existing_label
        else:
            person_label = current_label_id
            label_to_name_map[person_label] = person_name
            original_person_data_with_paths[person_label] = [] # Initialize list for this new person
            current_label_id += 1
        if person_label not in original_person_data_with_paths:
             original_person_data_with_paths[person_label] = []

        # Image Processing (Originals Only)
        faces_added_this_person = 0
        image_files = [f for f in os.listdir(person_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if not image_files:
             print(f"     [WARNING] No image files found for {person_name}.")
             continue

        for image_name in image_files:
            image_path = os.path.join(person_folder_path, image_name) # Full path to original image
            image = cv2.imread(image_path)
            if image is None: continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # --- Apply preprocessing if needed (Viola-Jones often works well without much) ---
            # gray = cv2.equalizeHist(gray) # Example: equalize histogram

            try:
                # --- Use Viola-Jones detector ---
                # Parameters might need tuning based on image characteristics
                detected_faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1, # How much image size is reduced at each scale
                    minNeighbors=5,  # How many neighbors each candidate rectangle should have
                    minSize=(50, 50) # Minimum possible object size
                )
                # --- End Viola-Jones detection ---
            except Exception as e:
                print(f"[ERROR] Viola-Jones detection failed for {image_name}: {e}")
                face_detection_failures += 1; continue

            if len(detected_faces) == 0:
                face_detection_failures += 1; continue

            # --- Select the largest face if multiple are detected ---
            if len(detected_faces) > 1:
                # print(f"     [INFO] Multiple faces ({len(detected_faces)}) detected in {image_name}, using largest.")
                detected_faces = sorted(detected_faces, key=lambda rect: rect[2] * rect[3], reverse=True)
            (x, y, w, h) = detected_faces[0] # Get largest face
            # --- End Face Selection ---

            if w <= 0 or h <= 0: continue # Should not happen with Viola-Jones usually

            # Ensure ROI coordinates are within image bounds
            y_end = min(y + h, gray.shape[0])
            x_end = min(x + w, gray.shape[1])
            face_roi = gray[y:y_end, x:x_end]

            if face_roi.size == 0: continue

            try:
                resized_face = cv2.resize(face_roi, face_size, interpolation=cv2.INTER_AREA)
                # Store BOTH the resized face and the original path
                data_item = {'face': resized_face, 'path': image_path}
                original_person_data_with_paths[person_label].append(data_item)
                faces_added_this_person += 1
                original_faces_detected += 1
            except cv2.error as e:
                print(f"     [ERROR] OpenCV error during resize for {image_path}: {e}")
                continue

        print(f"     - Added {faces_added_this_person} original faces (with paths) for this person.")

    print("[INFO] Phase 1 finished.")
    print(f"[INFO] Total ORIGINAL faces detected: {original_faces_detected}")
    if face_detection_failures > 0:
        print(f"[WARNING] Face detection failed/skipped for {face_detection_failures} images.")

    # Remove persons with zero original samples
    print("\n[INFO] Pre-Balancing Check...")
    labels_to_remove = [label for label, items in original_person_data_with_paths.items() if not items]
    if labels_to_remove:
        for label in labels_to_remove:
            person_name = label_to_name_map.pop(label, f"Unknown {label}")
            del original_person_data_with_paths[label]
            print(f"   - Removed '{person_name}' (Label {label}) due to zero samples.")
    else:
        print("   - All persons have at least one sample.")

    if len(original_person_data_with_paths) < 2:
         print(f"\n[ERROR] Need at least 2 persons with samples. Aborting.")
         return None, None

    # --- Phase 2: Calculate Minimum Samples (Originals) ---
    print("\n[INFO] Phase 2: Calculating minimum samples per person (based on originals)...")
    min_samples_per_person = float('inf')
    for label, items in original_person_data_with_paths.items():
        count = len(items)
        person_name = label_to_name_map.get(label, f"L {label}")
        print(f"   - Person '{person_name}' (Label {label}): Has {count} ORIGINAL samples.")
        if count > 0: min_samples_per_person = min(min_samples_per_person, count)

    print(f"[INFO] Minimum number of ORIGINAL samples per person: {min_samples_per_person}")
    if min_samples_per_person < 2: # Need at least 2 for stratified split
         print(f"[ERROR] Minimum samples per person ({min_samples_per_person}) is less than 2. Cannot perform stratified split. Aborting.")
         return None, None
    elif min_samples_per_person < 5:
        print(f"[WARNING] Minimum original sample count ({min_samples_per_person}) is low.")
    print("[INFO] Phase 2 finished.")


    # --- Phase 3: Balance Dataset (Originals + Paths) & Split ---
    print("\n[INFO] Phase 3: Balancing dataset and Splitting...")
    balanced_data_items = [] # Will contain list of {'face':..., 'path':...} dicts
    balanced_labels = []
    total_after_balancing = 0

    for label in sorted(original_person_data_with_paths.keys()):
        items = original_person_data_with_paths[label]
        selected_items = random.sample(items, min_samples_per_person)
        balanced_data_items.extend(selected_items)
        balanced_labels.extend([label] * min_samples_per_person)
        total_after_balancing += min_samples_per_person

    print(f"   - Balanced dataset created with {total_after_balancing} total items (face+path).")

    # Perform Train-Test Split on the BALANCED list of DICTIONARIES
    print(f"   - Splitting balanced data ({100*(1-TRAIN_TEST_SPLIT_RATIO):.0f}% train / {100*TRAIN_TEST_SPLIT_RATIO:.0f}% test)...")
    try:
        # Split the list of dictionaries and the labels list
        train_data_items, test_data_items, y_train, y_test = train_test_split(
            balanced_data_items,  # List of {'face':..., 'path':...}
            balanced_labels,      # List of corresponding labels
            test_size=TRAIN_TEST_SPLIT_RATIO,
            random_state=RANDOM_STATE_SPLIT,
            stratify=balanced_labels # Stratify based on labels
        )
    except ValueError as e:
        print(f"[ERROR] Could not split data. Error: {e}")
        label_counts = Counter(balanced_labels)
        print("[DEBUG] Balanced label counts before split:", label_counts)
        return None, None

    print(f"   - Split complete:")
    print(f"     - Training items (face+path): {len(train_data_items)}")
    print(f"     - Validation items (face+path): {len(test_data_items)}")
    print("[INFO] Phase 3 finished.")

    # --- Phase 4: Augment ONLY the Training Faces ---
    print(f"\n[INFO] Phase 4: Augmenting ONLY the Training faces (from {len(train_data_items)} original items)...")
    X_train_augmented_faces = [] # List to store augmented face image arrays
    y_train_augmented_labels = []  # Corresponding labels

    if AUGMENT_TRAIN_DATA:
        for item, label in zip(train_data_items, y_train):
            orig_face = item['face'] # Extract the face numpy array from the dict
            if orig_face is None: continue # Should not happen, but safety check

            # Augmentation logic (same as before)
            rois_to_add = [orig_face]
            flipped_face = None
            if random.random() < AUG_FLIP_PROB:
                flipped_face = cv2.flip(orig_face, 1)
                rois_to_add.append(flipped_face)

            final_rois_for_this_image = []
            for current_roi in rois_to_add:
                final_rois_for_this_image.append(current_roi)
                if random.random() < AUG_BRIGHTNESS_CONTRAST_PROB:
                    try:
                        adjusted_roi = adjust_brightness_contrast(current_roi, AUG_CONTRAST_RANGE, AUG_BRIGHTNESS_RANGE)
                        final_rois_for_this_image.append(adjusted_roi)
                    except cv2.error as e:
                        print(f" [WARN] Augmentation (brightness/contrast) failed: {e}")

            X_train_augmented_faces.extend(final_rois_for_this_image)
            y_train_augmented_labels.extend([label] * len(final_rois_for_this_image))

        print(f"   - Augmentation complete. Final augmented training faces: {len(X_train_augmented_faces)}")
    else:
        print("   - Augmentation disabled. Using original training faces.")
        # Extract original faces if augmentation is off
        X_train_augmented_faces = [item['face'] for item in train_data_items]
        y_train_augmented_labels = y_train # Labels remain the same

    print("[INFO] Phase 4 finished.")


    # --- Phase 5: Save Datasets (Augmented Train Faces, Original Test Images) ---
    print("\n[INFO] Phase 5: Saving datasets...")
    # Save the AUGMENTED CROPPED faces for training
    save_cropped_face_dataset(X_train_augmented_faces, y_train_augmented_labels, train_data_save_path, label_to_name_map)
    # Save the ORIGINAL FULL images for validation/testing
    save_original_images(test_data_items, y_test, validation_data_save_path, label_to_name_map)
    print("[INFO] Phase 5 finished.")


    # --- Phase 6: Train LBPH Model ---
    print("\n[INFO] Phase 6: Training LBPH model on AUGMENTED training faces...")
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

    start_training_time = time.time()
    try:
        # Train only on the face image arrays and their labels
        recognizer.train(X_train_augmented_faces, np.array(y_train_augmented_labels, dtype=np.int32))
    except cv2.error as e:
        print(f"\n[ERROR] OpenCV error during training: {e}"); return None, None
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during training: {e}"); return None, None

    end_training_time = time.time()
    print(f"[INFO] LBPH model trained successfully in {end_training_time - start_training_time:.2f} seconds.")
    print("[INFO] Phase 6 finished.")

    end_process_time = time.time()
    print(f"\n[INFO] Entire workflow finished in {end_process_time - start_time:.2f} seconds.")
    print(f"[INFO] Final Label map: {label_to_name_map}")

    return recognizer, label_to_name_map


# --- Main execution block ---
if __name__ == "__main__":
    print("="*40)
    print("Starting LBPH Face Recognizer Script")
    print(" (Viola-Jones Detector, Save Original Test, Augment Train)")
    print("="*40)

    if not os.path.isdir(dataset_path):
        print(f"[ERROR] Dataset path not found: '{dataset_path}'")
    else:
        trained_recognizer, final_label_map = process_balance_split_augment_train(dataset_path)

        if trained_recognizer is not None and final_label_map is not None:
            print("\n[INFO] Saving trained model and the final label map...")
            try:
                trained_recognizer.save(model_filename)
                print(f"[SUCCESS] Trained model saved to: {model_filename}")
            except Exception as e:
                print(f"[ERROR] Failed to save model '{model_filename}': {e}")
            try:
                # Save map with integer keys for consistency with loading
                json_compatible_map = {int(k): v for k, v in final_label_map.items()}
                with open(label_map_filename, 'w') as f:
                    json.dump(json_compatible_map, f, indent=4)
                print(f"[SUCCESS] Label map saved to: {label_map_filename}")
            except Exception as e:
                print(f"[ERROR] Failed to save label map '{label_map_filename}': {e}")
            print("\n[INFO] Process completed successfully.")
        else:
            print("\n[ERROR] Model training process failed or was aborted.")

    print("="*40)
    print("Script Finished.")
    print("="*40)