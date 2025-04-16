import os
import cv2
import dlib
import numpy as np
import json
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print(f"Using OpenCV version: {cv2.__version__}")
print(f"Using dlib version: {dlib.__version__}")
print(f"Using Numpy version: {np.__version__}")
try:
    import sklearn
    print(f"Using scikit-learn version: {sklearn.__version__}")
    import matplotlib
    print(f"Using matplotlib version: {matplotlib.__version__}")
    import seaborn
    print(f"Using seaborn version: {seaborn.__version__}")
except ImportError as e:
    print(f"[ERROR] Missing library: {e}. Please install scikit-learn, matplotlib, and seaborn.")
    exit()

# --- Configuration ---
# !!! IMPORTANT: Set this to the path where your TEST/VALIDATION images are stored !!!
# This should contain the ORIGINAL images set aside during the train/test split
# (e.g., the directory created by best_train.py's validation_data_save_path)
TEST_DATA_PATH = "lbph_validation_data_original_images"

# Model and label map filenames (MUST MATCH the trained LBPH model)
MODEL_FILENAME = "lbph_model_train_augmented.yml" # Model trained by best_train.py
LABEL_MAP_FILENAME = "label_map_final.json"       # Label map corresponding to the model

# Parameters (MUST MATCH training/testing configuration)
FACE_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 100 # As used in test.py (lower distance = higher confidence)

# Output filenames
CONFUSION_MATRIX_FILENAME = "lbph_confusion_matrix.png"

# --- Load Necessary Components ---

def load_essentials(model_path, label_map_path):
    """Loads the LBPH recognizer, label map, and face detector."""
    recognizer = None
    label_to_name = None
    detector = None

    # Load Detector
    print("[INFO] Loading face detector...")
    try:
        detector = dlib.get_frontal_face_detector()
    except Exception as e:
        print(f"[ERROR] Failed to load dlib face detector: {e}")
        return None, None, None

    # Load Recognizer
    print(f"[INFO] Loading trained LBPH model from: {model_path}")
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)
    except cv2.error as e:
        print(f"[ERROR] OpenCV error loading model '{model_path}': {e}")
        print("[INFO] Ensure the model file exists and is a valid LBPH model.")
        return None, None, detector # Return detector even if model fails
    except Exception as e:
        print(f"[ERROR] Failed to load LBPH model: {e}")
        return None, None, detector

    # Load Label Map
    print(f"[INFO] Loading label map from: {label_map_path}")
    try:
        with open(label_map_path, 'r') as f:
            label_to_name_str_keys = json.load(f)
            # Convert JSON string keys back to integer keys
            label_to_name = {int(k): v for k, v in label_to_name_str_keys.items()}
        print(f"[INFO] Loaded label map: {label_to_name}")
    except FileNotFoundError:
        print(f"[ERROR] Label map file not found: {label_map_path}")
        # Return loaded components so far, maybe user can provide map manually
        return recognizer, None, detector
    except Exception as e:
        print(f"[ERROR] Failed to load label map: {e}")
        return recognizer, None, detector

    return recognizer, label_to_name, detector

# --- Evaluation Function ---

def evaluate_lbph_model(test_data_dir, recognizer, detector, label_map, face_size, confidence_threshold):
    """Evaluates the LBPH model on the test dataset and returns prediction lists."""
    print("\n" + "="*30)
    print("[INFO] Starting LBPH Model Evaluation on Test Set...")
    print("="*30)

    if not recognizer: print("[ERROR] Recognizer not loaded."); return [], []
    if not detector: print("[ERROR] Detector not loaded."); return [], []
    if not label_map: print("[ERROR] Label map not loaded."); return [], []
    if not os.path.isdir(test_data_dir):
        print(f"[ERROR] Test data directory not found: {test_data_dir}")
        return [], []

    name_to_label = {name: label_id for label_id, name in label_map.items()}
    all_true_labels = []
    all_pred_labels_confident = [] # Store only predictions meeting threshold

    persons = sorted([d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d))])
    if not persons:
         print(f"[ERROR] No person subdirectories found in: {test_data_dir}")
         return [], []

    print(f"[INFO] Evaluating on {len(persons)} persons found in test directory...")
    total_images = 0
    processed_faces = 0
    skipped_confidence = 0
    skipped_detection = 0
    read_errors = 0

    for person_name in persons:
        if person_name not in name_to_label:
            print(f"  [WARNING] Person '{person_name}' in test dir but not in label map. Skipping.")
            continue

        person_folder_path = os.path.join(test_data_dir, person_name)
        true_label_id = name_to_label[person_name]
        # print(f"  - Evaluating: {person_name} (Label: {true_label_id})") # Verbose

        for image_name in os.listdir(person_folder_path):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue

            image_path = os.path.join(person_folder_path, image_name)
            total_images += 1

            image = cv2.imread(image_path)
            if image is None:
                # print(f"    [WARN] Cannot read image: {image_name}") # Verbose
                read_errors +=1
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Note: Preprocessing like equalizeHist is NOT applied here,
            # matching the basic test.py logic. If training used more
            # preprocessing (like fischerDCVA_train), apply it here too.

            try:
                detected_faces = detector(gray, 0) # Use upsampling=0 for speed? Or 1 like test.py? Let's use 1 like test.py
            except Exception:
                skipped_detection += 1
                continue

            if len(detected_faces) == 0:
                skipped_detection += 1
                continue
            # --- Filter: ONLY use images where ONE face is detected (if desired) ---
            # if len(detected_faces) != 1:
            #     skipped_detection +=1
            #     continue
            # --- End Filter ---

            # Use the first (or largest) detected face
            face_rect = max(detected_faces, key=lambda rect: rect.width() * rect.height())
            x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
            x, y = max(0, x), max(0, y)
            if w <= 0 or h <= 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
                 skipped_detection += 1
                 continue

            face_roi = gray[y:min(y+h, gray.shape[0]), x:min(x+w, gray.shape[1])]
            if face_roi.size == 0:
                skipped_detection += 1
                continue

            try:
                resized_face = cv2.resize(face_roi, face_size, interpolation=cv2.INTER_AREA)
            except cv2.error:
                skipped_detection += 1
                continue

            # --- Prediction ---
            try:
                predicted_label_id, confidence = recognizer.predict(resized_face)
                processed_faces += 1
                # Store the true label for accuracy calculation
                all_true_labels.append(true_label_id)

                # Check confidence threshold (lower distance is better for LBPH)
                if confidence < confidence_threshold:
                    all_pred_labels_confident.append(predicted_label_id)
                else:
                    # Prediction is considered 'Unknown' or unreliable due to high distance
                    all_pred_labels_confident.append(-1) # Use -1 for unknown/below-threshold
                    skipped_confidence += 1

            except cv2.error as pred_err:
                print(f"    [WARN] Prediction failed for {image_name}: {pred_err}")
                # Need to decide how to handle this - append true label but maybe -2 for prediction?
                # For simplicity, let's skip appending if prediction fails
                continue

    print("\n[INFO] Evaluation Loop Summary:")
    print(f"  - Total images scanned: {total_images}")
    print(f"  - Faces processed for recognition: {processed_faces}")
    print(f"  - Skipped (no/multiple/bad face detection): {skipped_detection}")
    print(f"  - Skipped (failed image read): {read_errors}")
    print(f"  - Predictions ignored (distance >= threshold {confidence_threshold}): {skipped_confidence}")
    print("="*30 + "\n")

    # Ensure lists have the same length if predictions failed mid-way
    if len(all_true_labels) != len(all_pred_labels_confident):
         print("[ERROR] Mismatch between true label count and confident prediction count. Check evaluation loop logic.")
         # This shouldn't happen with the current logic, but as a failsafe:
         min_len = min(len(all_true_labels), len(all_pred_labels_confident))
         return all_true_labels[:min_len], all_pred_labels_confident[:min_len]

    return all_true_labels, all_pred_labels_confident


# --- Reporting and Visualization Function ---

def generate_results(true_labels, pred_labels, label_map, conf_matrix_filename):
    """Generates accuracy, classification report, and confusion matrix."""
    if not true_labels or not pred_labels:
        print("[INFO] No results to generate report from.")
        return

    # Calculate overall accuracy based *only* on predictions that passed the threshold
    # Note: We use the labels where prediction was not -1
    valid_indices = [i for i, p in enumerate(pred_labels) if p != -1]
    if not valid_indices:
        print("[INFO] No predictions passed the confidence threshold. Cannot calculate metrics.")
        return

    true_labels_valid = [true_labels[i] for i in valid_indices]
    pred_labels_valid = [pred_labels[i] for i in valid_indices]

    accuracy = accuracy_score(true_labels_valid, pred_labels_valid)
    print(f"Overall Accuracy (on predictions below threshold): {accuracy * 100:.2f}%")

    # Prepare labels for report and matrix
    # Include -1 if it exists in predictions, but map it to "Unknown"
    present_labels = sorted(list(set(true_labels) | set(pred_labels)))
    target_labels = [l for l in present_labels if l != -1 and l in label_map] # Known labels present
    target_names = [label_map.get(l, f"Unknown_{l}") for l in target_labels]

    if -1 in present_labels:
         target_labels_with_unknown = target_labels + [-1]
         target_names_with_unknown = target_names + ["Unknown/Rejected"]
    else:
         target_labels_with_unknown = target_labels
         target_names_with_unknown = target_names


    # --- Classification Report ---
    print("\n" + "="*40)
    print("Classification Report (Predictions Below Threshold vs. True Labels)")
    print(" (Precision/Recall/F1 calculated ONLY for known classes)")
    print("="*40)
    try:
        # Use labels=target_labels to only report on known classes
        report = classification_report(true_labels, pred_labels,
                                       labels=target_labels,
                                       target_names=target_names,
                                       zero_division=0)
        print(report)
    except Exception as e:
        print(f"[ERROR] Could not generate classification report: {e}")


    # --- Confusion Matrix ---
    print("\n" + "="*40)
    print("Generating Confusion Matrix (Predictions Below Threshold vs. True Labels)")
    print(" (Rows: True Label, Columns: Predicted Label)")
    print("="*40)
    try:
        # Use target_labels_with_unknown to include the rejected predictions
        cm = confusion_matrix(true_labels, pred_labels, labels=target_labels_with_unknown)

        plt.figure(figsize=(max(8, len(target_names_with_unknown)*0.8), max(6, len(target_names_with_unknown)*0.6)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names_with_unknown,
                    yticklabels=target_names_with_unknown,
                    annot_kws={"size": 10})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix (LBPH Model)', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(conf_matrix_filename)
        print(f"[SUCCESS] Confusion matrix saved to: {conf_matrix_filename}")
        plt.show()
    except Exception as e:
        print(f"[ERROR] Failed to generate Confusion Matrix: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    print("="*50)
    print(" LBPH Model Evaluation Script ")
    print("="*50)

    # Load components
    recognizer, label_map, detector = load_essentials(MODEL_FILENAME, LABEL_MAP_FILENAME)

    if recognizer and label_map and detector:
        # Perform evaluation
        true_labels, predicted_labels = evaluate_lbph_model(
            TEST_DATA_PATH,
            recognizer,
            detector,
            label_map,
            FACE_SIZE,
            CONFIDENCE_THRESHOLD
        )

        # Generate and display results
        generate_results(true_labels, predicted_labels, label_map, CONFUSION_MATRIX_FILENAME)
    else:
        print("\n[ERROR] Essential components failed to load. Cannot proceed with evaluation.")

    print("\n" + "="*50)
    print(" Evaluation Script Finished ")
    print("="*50)