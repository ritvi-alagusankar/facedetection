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
TEST_DATA_PATH = "lbph_validation_data_original_images" # Or your actual test set path

# Model and label map filenames (MUST MATCH the trained LBPH model)
MODEL_FILENAME = "lbph_model_train_augmented.yml"
LABEL_MAP_FILENAME = "label_map_final.json"

# Parameters (MUST MATCH training/testing configuration)
FACE_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 100 # Lower distance = higher confidence for LBPH

# Output filenames
CONFUSION_MATRIX_FILENAME = "lbph_confusion_matrix_confident.png" # Renamed output file

# --- Load Necessary Components ---

def load_essentials(model_path, label_map_path):
    """Loads the LBPH recognizer, label map, and face detector."""
    recognizer = None
    label_to_name = None
    detector = None
    # (Loading logic remains the same as before)
    print("[INFO] Loading face detector...")
    try:
        detector = dlib.get_frontal_face_detector()
    except Exception as e:
        print(f"[ERROR] Failed to load dlib face detector: {e}")
        return None, None, None
    print(f"[INFO] Loading trained LBPH model from: {model_path}")
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)
    except cv2.error as e:
        print(f"[ERROR] OpenCV error loading model '{model_path}': {e}")
        return None, None, detector
    except Exception as e:
        print(f"[ERROR] Failed to load LBPH model: {e}")
        return None, None, detector
    print(f"[INFO] Loading label map from: {label_map_path}")
    try:
        with open(label_map_path, 'r') as f:
            label_to_name_str_keys = json.load(f)
            label_to_name = {int(k): v for k, v in label_to_name_str_keys.items()}
        print(f"[INFO] Loaded label map: {label_to_name}")
    except FileNotFoundError:
        print(f"[ERROR] Label map file not found: {label_map_path}")
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
    # (Error checking remains the same)
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
            # print(f"  [WARNING] Person '{person_name}' in test dir but not in label map. Skipping.") # Verbose
            continue

        person_folder_path = os.path.join(test_data_dir, person_name)
        true_label_id = name_to_label[person_name]

        for image_name in os.listdir(person_folder_path):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue

            image_path = os.path.join(person_folder_path, image_name)
            total_images += 1

            image = cv2.imread(image_path)
            if image is None:
                read_errors +=1
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            try:
                detected_faces = detector(gray, 1) # Use upsampling=1
            except Exception:
                skipped_detection += 1
                continue

            if len(detected_faces) == 0:
                skipped_detection += 1
                continue

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

                # Check confidence threshold (lower distance is better for LBPH)
                if confidence < confidence_threshold:
                    # Only store pairs where prediction is confident
                    all_true_labels.append(true_label_id)
                    all_pred_labels_confident.append(predicted_label_id)
                else:
                    skipped_confidence += 1

            except cv2.error as pred_err:
                # print(f"    [WARN] Prediction failed for {image_name}: {pred_err}") # Verbose
                # Skip appending if prediction fails
                continue

    print("\n[INFO] Evaluation Loop Summary:")
    print(f"  - Total images scanned: {total_images}")
    print(f"  - Faces processed for recognition: {processed_faces}")
    print(f"  - Skipped (no/multiple/bad face detection/errors): {skipped_detection + read_errors}")
    print(f"  - Predictions ignored (distance >= threshold {confidence_threshold}): {skipped_confidence}")
    print(f"  - Predictions included in report/matrix: {len(all_true_labels)}") # Should match len(all_pred_labels_confident)
    print("="*30 + "\n")

    return all_true_labels, all_pred_labels_confident # Return lists containing only confident predictions

# --- Reporting and Visualization Function (MODIFIED) ---

def generate_results(true_labels, pred_labels, label_map, conf_matrix_filename):
    """Generates accuracy, classification report, and confusion matrix
       ONLY for predictions that passed the confidence threshold."""
    if not true_labels or not pred_labels:
        print("[INFO] No confident predictions to generate report from.")
        return

    # --- All labels passed to this function are already confident predictions ---
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"Overall Accuracy (on {len(true_labels)} predictions below threshold): {accuracy * 100:.2f}%")

    # Prepare labels for report and matrix
    # Use only the labels present in the confident predictions
    present_labels = sorted(list(set(true_labels) | set(pred_labels)))
    # Ensure we only include labels that are actually in the original label_map
    target_labels = [l for l in present_labels if l in label_map]
    target_names = [label_map.get(l, f"Unknown_{l}") for l in target_labels]

    if not target_labels:
        print("[ERROR] No known labels found among the confident predictions. Cannot generate report/matrix.")
        return

    # --- Classification Report ---
    print("\n" + "="*40)
    print("Classification Report (Only Confident Predictions)")
    print("="*40)
    try:
        # Report based ONLY on the confident predictions
        report = classification_report(true_labels, pred_labels,
                                       labels=target_labels,
                                       target_names=target_names,
                                       zero_division=0)
        print(report)
    except Exception as e:
        print(f"[ERROR] Could not generate classification report: {e}")


    # --- Confusion Matrix ---
    print("\n" + "="*40)
    print("Generating Confusion Matrix (Only Confident Predictions)")
    print(" (Rows: True Label, Columns: Predicted Label)")
    print("="*40)
    try:
        # Compute matrix based ONLY on confident predictions vs their true labels
        cm = confusion_matrix(true_labels, pred_labels, labels=target_labels)

        plt.figure(figsize=(max(8, len(target_names)*0.8), max(6, len(target_names)*0.6)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, # Use names without "Unknown/Rejected"
                    yticklabels=target_names, # Use names without "Unknown/Rejected"
                    annot_kws={"size": 10})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix (LBPH Model - Confident Predictions Only)', fontsize=14)
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
    print(" LBPH Model Evaluation Script (Confident Predictions Only) ")
    print("="*50)

    # Load components
    recognizer, label_map, detector = load_essentials(MODEL_FILENAME, LABEL_MAP_FILENAME)

    if recognizer and label_map and detector:
        # Perform evaluation - Now returns lists filtered by confidence
        true_labels_confident, predicted_labels_confident = evaluate_lbph_model(
            TEST_DATA_PATH,
            recognizer,
            detector,
            label_map,
            FACE_SIZE,
            CONFIDENCE_THRESHOLD
        )

        # Generate and display results using the filtered lists
        generate_results(true_labels_confident, predicted_labels_confident,
                         label_map, CONFUSION_MATRIX_FILENAME)
    else:
        print("\n[ERROR] Essential components failed to load. Cannot proceed with evaluation.")

    print("\n" + "="*50)
    print(" Evaluation Script Finished ")
    print("="*50)