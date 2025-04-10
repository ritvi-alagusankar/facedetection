# test_fisher_dcva.py
import os
import cv2
import dlib
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import defaultdict
from sklearn.preprocessing import normalize # For normalize function used in DCVA
# from scipy.spatial.distance import euclidean # Or implement simple euclidean
from tqdm.notebook import tqdm # Use standard tqdm if not in notebook

print(f"Using OpenCV version: {cv2.__version__}")
print(f"Using dlib version: {dlib.__version__}")
print(f"Using Numpy version: {np.__version__}")
# Ensure scikit-learn, matplotlib, seaborn are installed

# --- Configuration ---
# Use the SAME dataset for testing for now, ideally use a separate test set
dataset_path = "png_dataset/png_dataset"
face_size = (200, 200) # MUST MATCH training face_size
model_output_prefix = "fisher_dcva" # Prefix used during saving
model_filename = f"{model_output_prefix}_model.npz"
label_map_filename = f"{model_output_prefix}_label_map.json"
prediction_threshold = 0.5 # Cosine similarity threshold (higher is better)
prediction_method = "cosine" # MUST MATCH how model expects similarity ('cosine' or 'euclidean')

# --- Load Face Detector ---
print("[INFO] Loading face detector...")
try:
    hog_detector = dlib.get_frontal_face_detector()
except Exception as e:
    print(f"[ERROR] Failed to load Dlib face detector: {e}")
    exit()

# --- Preprocessing Function (MUST MATCH training) ---
def preprocess_face(img):
    if len(img.shape) > 2: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(img)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(blurred)

# --- DCVAHybridClassifier Class (MUST MATCH training, including helpers) ---
def simple_euclidean(v1, v2): return np.sqrt(np.sum((v1 - v2)**2))

class DCVAHybridClassifier:
    def __init__(self, eigenfaces, mean_vector):
        self.eigenfaces = eigenfaces
        self.mean = mean_vector
        self.class_vectors = {} # label_id -> normalized class vector
        self.label_names = {}   # label_id -> person_name

    # fit method is not needed for testing, only for training

    def predict(self, img, method="cosine"):
        n_features = self.eigenfaces.shape[0]
        if img.size != n_features: return None, 0.0
        flat = img.flatten().astype(np.float32).reshape(1, -1)
        proj = (flat - self.mean) @ self.eigenfaces
        test_vec = normalize(proj, norm='l2')[0]

        best_label = None
        best_score = -np.inf if method == "cosine" else np.inf

        if not self.class_vectors: return None, 0.0

        for lbl, class_vec in self.class_vectors.items():
            if method == "cosine":
                score = np.dot(test_vec, class_vec)
                if score > best_score:
                    best_score = score; best_label = lbl
            elif method == "euclidean":
                score = simple_euclidean(test_vec, class_vec)
                if score < best_score:
                    best_score = score; best_label = lbl
            else: raise ValueError("Unsupported method")

        if best_label is None:
            final_score = 0.0 if method == "cosine" else np.inf
            return None, final_score
        else:
            final_score = best_score
            if method == "cosine": final_score = np.clip(best_score, 0.0, 1.0)
            return best_label, final_score
# --- End DCVAHybridClassifier Class ---


# --- Load Model Components and Label Map ---
def load_dcva_model(model_path, label_map_path):
    """Loads DCVA components and label map."""
    print(f"[INFO] Loading model components from: {model_path}")
    print(f"[INFO] Loading label map from: {label_map_path}")
    label_names = None
    dcva = None

    # Load Label Map
    try:
        with open(label_map_path, 'r') as f:
            str_key_label_map = json.load(f)
            # Convert keys back to integers
            label_names = {int(k): v for k, v in str_key_label_map.items()}
            print(f"[INFO] Loaded label map: {label_names}")
    except FileNotFoundError:
        print(f"[ERROR] Label map file not found: {label_map_path}")
        return None, None
    except Exception as e:
        print(f"[ERROR] Failed to load label map: {e}")
        return None, None

    # Load Model Components from NPZ
    try:
        data = np.load(model_path)
        eigenfaces = data['eigenfaces']
        mean = data['mean']
        class_vector_labels = data['class_vector_labels']
        class_vector_data = data['class_vector_data']
        print("[INFO] Loaded model components (eigenfaces, mean, class vectors).")

        # Reconstruct the DCVA object
        dcva = DCVAHybridClassifier(eigenfaces, mean)
        dcva.label_names = label_names # Assign loaded label map
        # Reconstruct the class_vectors dictionary
        dcva.class_vectors = {int(lbl): vec for lbl, vec in zip(class_vector_labels, class_vector_data)}
        print(f"  - Reconstructed DCVA with {len(dcva.class_vectors)} class vectors.")

    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {model_path}")
        return None, label_names # Return label_names even if model fails
    except KeyError as e:
         print(f"[ERROR] Missing key in model file '{model_path}': {e}")
         return None, label_names
    except Exception as e:
        print(f"[ERROR] Failed to load model components: {e}")
        return None, label_names

    return dcva, label_names

# --- Testing Function ---
def test_fisher_dcva(dataset_dir, dcva_model, detector, target_size, threshold, method):
    """Evaluates the loaded DCVA model on the dataset."""
    print("\n" + "="*30)
    print("[INFO] Starting DCVA model evaluation...")
    print("="*30)

    if dcva_model is None: print("[ERROR] DCVA model not loaded."); return [], [], []
    if detector is None: print("[ERROR] Detector not loaded."); return [], [], []
    if not dcva_model.label_names: print("[ERROR] Label names missing in DCVA model."); return [], [], []
    if not os.path.isdir(dataset_dir): print(f"[ERROR] Dataset path not found: {dataset_dir}"); return [], [], []

    list_true_labels = []
    list_pred_labels = []
    list_scores = []

    name_to_label = {name: label_id for label_id, name in dcva_model.label_names.items()}
    known_persons = sorted(list(name_to_label.keys()))

    total_images_scanned = 0
    total_faces_detected = 0
    total_faces_processed = 0
    correct_confident_predictions = 0

    print(f"[INFO] Evaluating {len(known_persons)} persons from label map...")
    # Wrap with tqdm for progress if desired (install standard tqdm: pip install tqdm)
    # from tqdm import tqdm
    # for person_name in tqdm(known_persons, desc="Evaluating Persons"):
    for person_name in known_persons:
        if person_name not in name_to_label: continue
        person_folder_path = os.path.join(dataset_dir, person_name)
        true_label_id = name_to_label[person_name]

        if not os.path.isdir(person_folder_path):
            print(f"[WARN] No directory for '{person_name}'. Skipping.")
            continue

        for image_name in os.listdir(person_folder_path):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            img_path = os.path.join(person_folder_path, image_name)
            total_images_scanned += 1

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            # --- Apply SAME Preprocessing ---
            processed_img = preprocess_face(img)
            # -------------------------------

            try:
                detected_faces = detector(processed_img, 1)
            except Exception: continue

            # --- Filter: Only test images with exactly ONE face (like training) ---
            if len(detected_faces) != 1: continue
            # -------------------------------------------------------------------
            total_faces_detected += 1

            face = detected_faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(processed_img.shape[1], x + w), min(processed_img.shape[0], y + h)
            w_adj, h_adj = x2 - x1, y2 - y1

            face_roi = processed_img[y1:y2, x1:x2]

            if face_roi.size == 0 or w_adj < 50 or h_adj < 50: continue

            try:
                face_resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
            except cv2.error: continue

            # --- Perform prediction AND Store results ---
            predicted_label, score = dcva_model.predict(face_resized, method=method)
            total_faces_processed += 1

            list_true_labels.append(true_label_id)
            list_pred_labels.append(predicted_label if predicted_label is not None else -1) # Use -1 for None prediction
            list_scores.append(score)
            # ------------------------------------------

            # Check for correct prediction *above* threshold (for summary)
            # Note: Cosine score > threshold means confident match
            if method == "cosine":
                is_confident = score > threshold
            elif method == "euclidean":
                is_confident = score < threshold # Lower is better for distance
            else:
                 is_confident = False # Default for unknown method

            if predicted_label is not None and predicted_label == true_label_id and is_confident:
                correct_confident_predictions += 1
            # Optional: Log misclassifications or low-score correct matches here if needed

    # --- Final Summary ---
    print("\n" + "="*30)
    print("[INFO] DCVA Evaluation Summary")
    print("="*30)
    # Print counts... (similar to LBPH test summary)
    print(f"Total Images Scanned: {total_images_scanned}")
    print(f"Total Faces Detected (with 1 face): {total_faces_detected}")
    print(f"Total Faces Processed (Prediction Attempted): {total_faces_processed}")
    print(f"Correct & Confident Recognitions (Above/Below Threshold {threshold}): {correct_confident_predictions}")
    if total_faces_processed > 0:
        accuracy = (correct_confident_predictions / total_faces_processed) * 100
        print(f"Recognition Success Rate (Confident & Correct / Total Processed): {accuracy:.2f}%")
    else:
        print("Recognition Success Rate: N/A")
    print("="*30 + "\n")

    if not list_true_labels: print("[WARNING] No prediction results collected.")
    return list_true_labels, list_pred_labels, list_scores


# --- Main Execution ---
if __name__ == "__main__":
    print("="*40)
    print("Starting Fisherface + DCVA Evaluation Script")
    print("="*40)

    # Load the trained DCVA model and label map
    dcva_classifier, label_names_map = load_dcva_model(model_filename, label_map_filename)

    if dcva_classifier is None or label_names_map is None:
        print("[ERROR] Failed to load model or label map. Exiting.")
    else:
        # Run evaluation
        true_labels, pred_labels, scores = test_fisher_dcva(
            dataset_path,
            dcva_classifier,
            hog_detector,
            face_size,
            prediction_threshold,
            prediction_method
        )

        # --- Visualization Code (Adapted for DCVA Scores) ---
        if not true_labels:
            print("\nNo evaluation results collected. Skipping visualizations.")
        else:
            print("\nGenerating Performance Visualizations...")
            # Ensure visualization libraries are imported
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.metrics import confusion_matrix, classification_report
                sns.set_theme(style="whitegrid")
            except ImportError:
                print("[ERROR] Please install matplotlib, seaborn, and scikit-learn to generate visualizations.")
                exit()

            # --- Confusion Matrix ---
            known_labels_list = sorted(label_names_map.keys())
            known_names_list = [label_names_map.get(lbl, f"L{lbl}") for lbl in known_labels_list]
            # Handle potential -1 label for None predictions if needed
            unique_labels_in_results = set(true_labels) | set(pred_labels)
            valid_known_labels = [lbl for lbl in known_labels_list if lbl in unique_labels_in_results and lbl != -1]
            if not valid_known_labels:
                 print("Error: No known labels found in evaluation results.")
            else:
                cm = confusion_matrix(true_labels, pred_labels, labels=valid_known_labels)
                valid_known_names = [label_names_map.get(lbl, f"L{lbl}") for lbl in valid_known_labels]
                plt.figure(figsize=(max(8, len(valid_known_names)*0.8), max(6, len(valid_known_names)*0.6)))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=valid_known_names, yticklabels=valid_known_names, annot_kws={"size": 10})
                plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix (DCVA Raw Predictions)')
                plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout(); plt.savefig(f'{model_output_prefix}_confusion_matrix.png'); plt.show()

            # --- Per-Person Accuracy ---
            if 'cm' in locals() and cm is not None:
                 class_counts = cm.sum(axis=1)
                 per_class_accuracy = np.zeros(len(valid_known_labels))
                 for i, count in enumerate(class_counts): per_class_accuracy[i] = cm.diagonal()[i] / count if count > 0 else np.nan
                 plt.figure(figsize=(max(10, len(valid_known_names)*0.6), 6))
                 bars = plt.bar(valid_known_names, per_class_accuracy * 100, color='lightcoral')
                 plt.xlabel('Person'); plt.ylabel('Accuracy (%)'); plt.title('Per-Person Accuracy (DCVA Raw Predictions)')
                 plt.xticks(rotation=45, ha='right'); plt.ylim(0, 105); plt.grid(axis='y', linestyle='--', alpha=0.7)
                 for bar in bars:
                     yval = bar.get_height()
                     if not np.isnan(yval): plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', va='bottom', ha='center', fontsize=9)
                 plt.tight_layout(); plt.savefig(f'{model_output_prefix}_per_person_accuracy.png'); plt.show()

            # --- Score Histogram (Adapted for Cosine Similarity) ---
            if scores:
                correct_scores = [s for t, p, s in zip(true_labels, pred_labels, scores) if t == p and p != -1]
                incorrect_scores = [s for t, p, s in zip(true_labels, pred_labels, scores) if t != p and p != -1]
                none_scores = [s for p, s in zip(pred_labels, scores) if p == -1] # Scores when prediction was None

                plt.figure(figsize=(12, 7))
                max_score = max(scores) if scores else 1.0
                min_score = min(scores) if scores else 0.0
                 # Adjust bins based on score range (cosine is usually 0-1)
                bins = np.linspace(max(0, min_score - 0.05), min(1, max_score + 0.05), 50)

                plt.hist(correct_scores, bins=bins, alpha=0.6, label=f'Correct ({len(correct_scores)})', color='green')
                plt.hist(incorrect_scores, bins=bins, alpha=0.6, label=f'Incorrect ({len(incorrect_scores)})', color='red')
                if none_scores: # Only plot if there were None predictions
                    plt.hist(none_scores, bins=bins, alpha=0.4, label=f'No Match ({len(none_scores)})', color='gray')

                # Threshold line (Higher score is better for cosine)
                plt.axvline(prediction_threshold, color='blue', linestyle='dashed', linewidth=2, label=f'Threshold ({prediction_threshold})')

                plt.xlabel(f'Score ({prediction_method} - Higher is Better)', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.title('Prediction Score Distribution (DCVA)', fontsize=14)
                plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
                plt.savefig(f'{model_output_prefix}_score_histogram.png'); plt.show()

            # --- Classification Report ---
            if 'valid_known_labels' in locals(): # Check if valid labels were determined
                 print("\n" + "="*40 + "\nClassification Report (DCVA Raw Predictions)\n" + "="*40)
                 report = classification_report(true_labels, pred_labels, labels=valid_known_labels, target_names=valid_known_names, zero_division=0)
                 print(report)
                 # Add report based on confident predictions if needed...

    print("="*40)
    print("Evaluation Script Finished.")
    print("="*40)