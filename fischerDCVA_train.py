# train_fisher_dcva.py
import os
import cv2
import dlib
import numpy as np
import json
import time
import random
from collections import defaultdict
from sklearn.preprocessing import normalize # For normalize function used in DCVA
# from scipy.spatial.distance import euclidean # Or implement simple euclidean

print(f"Using OpenCV version: {cv2.__version__}")
print(f"Using dlib version: {dlib.__version__}")
print(f"Using Numpy version: {np.__version__}")
# Ensure scikit-learn is installed: pip install scikit-learn

# --- Configuration ---
dataset_path = "png_dataset/png_dataset" # Training dataset path
face_size = (200, 200)                   # Face size used by this model
target_images_per_person = 35            # Target number of images after augmentation
model_output_prefix = "fisher_dcva"      # Prefix for saved files
model_filename = f"{model_output_prefix}_model.npz"
label_map_filename = f"{model_output_prefix}_label_map.json"

# --- Load Face Detector ---
print("[INFO] Loading face detector...")
try:
    hog_detector = dlib.get_frontal_face_detector()
except Exception as e:
    print(f"[ERROR] Failed to load Dlib face detector: {e}")
    exit()

# --- Preprocessing Function (as provided) ---
def preprocess_face(img):
    """Applies equalization, blur, and CLAHE."""
    # Ensure input is grayscale (load_training_data already does this)
    if len(img.shape) > 2:
         print("[WARN] Preprocessing received non-grayscale image, converting.")
         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Histogram Equalization first
    gray = cv2.equalizeHist(img)
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(blurred)
    return processed

# --- Augmentation Function (as provided) ---
def augment_image(image):
    """Generates augmented versions of an image."""
    augmented = []
    rows, cols = image.shape[:2] # Handle potential color image input? No, assume gray

    # Rotation
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1.0)
        # Use BORDER_REFLECT or BORDER_REPLICATE to handle edges
        rotated = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)

    # Brightness/Contrast
    for alpha, beta in [(1.1, 5), (0.9, -5)]: # Contrast (alpha), Brightness (beta)
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        augmented.append(adjusted)

    # Optional: Add Horizontal Flip (like LBPH example)
    # if random.random() < 0.5:
    #    flipped = cv2.flip(image, 1)
    #    augmented.append(flipped)

    return augmented

# --- Data Loading Function (modified slightly for clarity) ---
def load_training_data(dataset_dir, target_size, target_count):
    """Loads, preprocesses, detects, augments, and balances training data."""
    print(f"[INFO] Loading training data from: {dataset_dir}")
    label_dict = {}            # label_id -> person_name
    current_label = 0
    person_images = {}         # label_id -> list of processed face images
    total_original_valid = 0
    total_augmented = 0

    try:
        persons = sorted(os.listdir(dataset_dir))
    except FileNotFoundError:
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return None, None, None

    for person in persons:
        person_path = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_path):
            continue

        print(f"  - Processing person: {person}")
        label_dict[current_label] = person
        valid_original_images = [] # Store valid original faces for this person

        for img_name in os.listdir(person_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            img_path = os.path.join(person_path, img_name)

            # Read directly as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            # --- Apply Preprocessing ---
            processed_img = preprocess_face(img)
            # -------------------------

            try: # Detect faces on the *preprocessed* image
                detected_faces = hog_detector(processed_img, 1) # Use upsampling=1
            except Exception as e:
                print(f"[WARN] dlib detection failed for {img_name}: {e}"); continue

            # --- Crucial Filter: Keep only images with exactly ONE face ---
            if len(detected_faces) != 1:
                # print(f"    [SKIP] {img_name} - Found {len(detected_faces)} faces.") # Verbose
                continue
            # ----------------------------------------------------------

            face = detected_faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(processed_img.shape[1], x + w), min(processed_img.shape[0], y + h)
            w_adj, h_adj = x2 - x1, y2 - y1

            # --- Crop from the *preprocessed* image ---
            face_roi = processed_img[y1:y2, x1:x2]
            # -----------------------------------------

            # --- Filter small or invalid ROIs ---
            if face_roi.size == 0 or w_adj < 50 or h_adj < 50: # Check adjusted size
                # print(f"    [SKIP] {img_name} - Face ROI too small or invalid.") # Verbose
                continue
            # ----------------------------------

            try: # Resize to the target size for the model
                face_resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
                valid_original_images.append(face_resized)
            except cv2.error: continue # Skip resize errors

        print(f"    - Found {len(valid_original_images)} valid original face images.")
        total_original_valid += len(valid_original_images)

        if not valid_original_images:
             print(f"    [WARN] No valid faces found for {person}. Skipping augmentation.")
             # Don't increment label if no images found for this person
             del label_dict[current_label]
             continue # Skip to next person

        # --- Augmentation and Balancing by Count ---
        augmented_image_list = valid_original_images.copy()
        i = 0
        while len(augmented_image_list) < target_count:
            # Cycle through original valid images for augmentation source
            source_img = valid_original_images[i % len(valid_original_images)]
            generated_augmentations = augment_image(source_img)
            random.shuffle(generated_augmentations) # Add randomness to which augmentations get added first

            for aug_img in generated_augmentations:
                if len(augmented_image_list) >= target_count: break
                augmented_image_list.append(aug_img)
            if len(augmented_image_list) >= target_count: break # Exit while loop
            i += 1 # Move to next original image base

        # Store the final list (capped at target_count)
        person_images[current_label] = augmented_image_list[:target_count]
        num_augmented_added = len(person_images[current_label]) - len(valid_original_images)
        total_augmented += num_augmented_added
        print(f"    - Augmented to {len(person_images[current_label])} images (added {num_augmented_added} augmentations).")
        # -----------------------------------------

        current_label += 1 # Increment label ID for the next valid person

    if not person_images:
         print("[ERROR] No valid data loaded for any person. Cannot train.")
         return None, None, None

    # --- Flatten data for training ---
    final_images = []
    final_labels = []
    for label_id, img_list in person_images.items():
        final_images.extend(img_list)
        final_labels.extend([label_id] * len(img_list))
    # -------------------------------

    print(f"\n[INFO] Data Loading Summary:")
    print(f"  - Total valid original faces found: {total_original_valid}")
    print(f"  - Total augmented faces added: {total_augmented}")
    print(f"  - Total faces prepared for training: {len(final_images)}")
    print(f"  - Number of persons (classes): {len(label_dict)}")

    return final_images, np.array(final_labels, dtype=np.int32), label_dict


# --- DCVAHybridClassifier Class (as provided, added imports/implementations) ---
# Helper for Euclidean distance if scipy isn't used
def simple_euclidean(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

class DCVAHybridClassifier:
    def __init__(self, eigenfaces, mean_vector):
        self.eigenfaces = eigenfaces # Should be shape (n_features, n_components)
        self.mean = mean_vector      # Should be shape (1, n_features)
        self.class_vectors = {}      # label_id -> normalized class vector
        # self.labels = []           # Not explicitly used in predict, maybe for internal state?
        self.label_names = {}        # label_id -> person_name

    def fit(self, faces, labels, label_names):
        """Calculates class-specific vectors based on Fisherface projections."""
        print("[INFO] Fitting DCVA Hybrid Classifier...")
        self.label_names = label_names
        features = []
        n_features = self.eigenfaces.shape[0] # Original feature dimension (e.g., 200*200)

        print(f"  - Projecting {len(faces)} faces onto Fisherface subspace...")
        for img in faces:
            if img.size != n_features:
                 print(f"[WARN] Image size mismatch ({img.size} != {n_features}). Skipping.")
                 continue
            flat = img.flatten().astype(np.float32).reshape(1, -1)
            # Project: (face - mean) * Eigenvectors
            proj = (flat - self.mean) @ self.eigenfaces
            features.append(proj.flatten())

        if not features:
             print("[ERROR] No features could be projected. Cannot fit DCVA.")
             return False

        features = np.array(features) # Shape: (n_samples, n_components)
        labels = np.array(labels)     # Ensure labels correspond to the successfully projected features

        if len(features) != len(labels):
             print("[ERROR] Mismatch between number of projected features and labels after filtering.")
             # This might indicate issues during projection loop, adjust label array if needed
             # For simplicity, we'll stop here. Robust handling would track skipped indices.
             return False

        print(f"  - Calculating class vectors for {len(np.unique(labels))} classes...")
        class_features = defaultdict(list)
        for feat, lbl in zip(features, labels):
            class_features[lbl].append(feat)

        n_components = features.shape[1]
        for lbl, feats in class_features.items():
            feats = np.array(feats) # Shape: (n_samples_class, n_components)
            if feats.shape[0] < 1: continue # Skip if no features for this class

            # Calculate mean feature vector for the class
            mean_vector = np.mean(feats, axis=0) # Shape: (n_components,)

            # --- DCVA Logic ---
            # Calculate residuals within the class
            residuals = feats - mean_vector
            # Perform SVD on residuals to find directions of variation *within* the class
            try:
                 # Note: SVD might be slow for many features/components
                 U, S, Vt = np.linalg.svd(residuals, full_matrices=False)
                 # Identify vectors corresponding to near-zero singular values (null space of covariance within class)
                 # These directions have minimal variance *within* this class.
                 null_space_basis = Vt[S < 1e-5] # Adjust threshold if needed

                 if null_space_basis.size == 0:
                     # If no significant null space, use the simple class mean
                     common_vec = mean_vector
                     # print(f"    - Label {lbl}: No significant null space found, using mean.") # Verbose
                 else:
                     # Project residuals onto the null space and average
                     # This aims to find a vector robust to within-class variations
                     proj_onto_null = residuals @ null_space_basis.T @ null_space_basis
                     common_vec = mean_vector + np.mean(proj_onto_null, axis=0)
                     # print(f"    - Label {lbl}: Found null space (dim={null_space_basis.shape[0]}), calculated common vec.") # Verbose

            except np.linalg.LinAlgError:
                 print(f"[WARN] SVD failed for label {lbl}. Using simple mean vector.")
                 common_vec = mean_vector
            # --- End DCVA ---

            # Normalize and store the characteristic vector for the class
            # Use sklearn's normalize for robust L2 normalization
            self.class_vectors[lbl] = normalize(common_vec.reshape(1, -1), norm='l2')[0]

        print("[INFO] DCVA fitting complete.")
        return True # Indicate success


    def predict(self, img, method="cosine"):
        """Predicts the label for a given face image."""
        n_features = self.eigenfaces.shape[0]
        if img.size != n_features:
             print(f"[WARN] Predict input size mismatch ({img.size} != {n_features}).")
             return None, 0.0 # Or raise error

        flat = img.flatten().astype(np.float32).reshape(1, -1)
        proj = (flat - self.mean) @ self.eigenfaces
        # Normalize the projected test vector
        test_vec = normalize(proj, norm='l2')[0] # Shape: (n_components,)

        best_label = None
        # Initialize best_score based on method (cosine: higher is better, euclidean: lower is better)
        best_score = -np.inf if method == "cosine" else np.inf

        if not self.class_vectors: # Check if class vectors exist
             print("[WARN] No class vectors loaded/fitted in DCVA model.")
             return None, 0.0

        for lbl, class_vec in self.class_vectors.items():
            if method == "cosine":
                # Cosine Similarity = dot product of normalized vectors
                score = np.dot(test_vec, class_vec)
                if score > best_score:
                    best_score = score
                    best_label = lbl
            elif method == "euclidean":
                score = simple_euclidean(test_vec, class_vec) # Lower is better
                # score = np.linalg.norm(test_vec - class_vec) # Alternative numpy way
                if score < best_score:
                    best_score = score
                    best_label = lbl
            else:
                 raise ValueError("Unsupported prediction method. Use 'cosine' or 'euclidean'.")

        # Handle cases where no prediction could be made
        if best_label is None:
            # Return score based on method convention (0 for cosine, inf for euclidean)
            final_score = 0.0 if method == "cosine" else np.inf
            return None, final_score
        else:
            # Return the best label and its associated score
            # Clip cosine score to [0, 1] range (though dot product of unit vectors is [-1, 1])
            # Maybe clip to [-1, 1] or rescale? Let's clip [0,1] as per original snippet.
            final_score = best_score
            if method == "cosine":
                 final_score = np.clip(best_score, 0.0, 1.0)
            return best_label, final_score
# --- End DCVAHybridClassifier Class ---


# --- Main Training Function ---
def train_fisher_dcva(dataset_dir, target_size, target_count):
    """Loads data, trains Fisherface, fits DCVA, returns model and map."""
    faces, labels, label_dict = load_training_data(dataset_dir, target_size, target_count)

    if faces is None or labels is None or label_dict is None:
        print("[ERROR] Data loading failed. Aborting training.")
        return None, None

    if len(np.unique(labels)) < 2:
         print("[ERROR] Need at least two distinct persons (classes) to train Fisherfaces/DCVA. Aborting.")
         return None, None

    print("\n[INFO] Training Fisherface Recognizer...")
    start_time = time.time()
    # Ensure enough components are requested (max is C-1 classes)
    num_components = min(len(label_dict) - 1, 100) # Example limit, adjust as needed
    print(f"  - Using {num_components} components for Fisherfaces.")
    fisher_recognizer = cv2.face.FisherFaceRecognizer_create(num_components=num_components)
    try:
        fisher_recognizer.train(faces, labels)
    except cv2.error as e:
        print(f"[ERROR] OpenCV error during Fisherface training: {e}")
        print("  - Ensure all images are the same size and grayscale.")
        print(f"  - Number of samples: {len(faces)}, Number of unique labels: {len(np.unique(labels))}")
        return None, None
    print(f"[INFO] Fisherface training finished in {time.time() - start_time:.2f} seconds.")

    # Extract components for DCVA
    try:
        eigenfaces = fisher_recognizer.getEigenVectors() # Shape: (n_features, n_components)
        mean = fisher_recognizer.getMean().reshape(1, -1) # Shape: (1, n_features)
        # Note: Fisherfaces might return eigenvectors differently than PCA/Eigenfaces.
        # Check documentation/shape if issues arise. Usually (features x components).
        print(f"  - Extracted Mean vector shape: {mean.shape}")
        print(f"  - Extracted Eigenvectors shape: {eigenfaces.shape}")
    except Exception as e:
         print(f"[ERROR] Failed to extract components from Fisherface model: {e}")
         return None, None


    # Initialize and fit DCVA
    dcva = DCVAHybridClassifier(eigenfaces, mean)
    fit_success = dcva.fit(faces, labels, label_dict)

    if not fit_success:
         print("[ERROR] DCVA fitting failed.")
         return None, None

    return dcva, label_dict


# --- Main Execution ---
if __name__ == "__main__":
    print("="*40)
    print("Starting Fisherface + DCVA Training Script")
    print("="*40)

    if not os.path.isdir(dataset_path):
        print(f"[ERROR] Dataset path not found: '{dataset_path}'")
    else:
        # Train the model
        dcva_model, label_map = train_fisher_dcva(dataset_path, face_size, target_images_per_person)

        # Save the results if training was successful
        if dcva_model is not None and label_map is not None:
            print("\n[INFO] Saving trained model components and label map...")

            # Save DCVA components using np.savez_compressed
            try:
                np.savez_compressed(
                    model_filename,
                    eigenfaces=dcva_model.eigenfaces,
                    mean=dcva_model.mean,
                    # Convert class_vectors dict for saving: keys and values separately
                    class_vector_labels=np.array(list(dcva_model.class_vectors.keys())),
                    class_vector_data=np.array(list(dcva_model.class_vectors.values()))
                )
                print(f"[SUCCESS] Model components saved to: {model_filename}")
            except Exception as e:
                print(f"[ERROR] Failed to save model components '{model_filename}': {e}")

            # Save the label map as JSON
            try:
                with open(label_map_filename, 'w') as f:
                    # Ensure keys are strings for JSON standard (though int keys often work)
                    str_key_label_map = {str(k): v for k, v in label_map.items()}
                    json.dump(str_key_label_map, f, indent=4)
                print(f"[SUCCESS] Label map saved to: {label_map_filename}")
            except Exception as e:
                 print(f"[ERROR] Failed to save label map '{label_map_filename}': {e}")

            print("\n[INFO] Training process completed.")
        else:
            print("\n[ERROR] Model training failed.")

    print("="*40)
    print("Training Script Finished.")
    print("="*40)