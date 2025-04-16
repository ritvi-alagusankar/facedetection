import os
import cv2
import dlib
import numpy as np
import json
from skimage.feature import local_binary_pattern # Using scikit-image for LBP calculation
import shutil

print(f"Using OpenCV version: {cv2.__version__}")
print(f"Using dlib version: {dlib.__version__}")
print(f"Using Numpy version: {np.__version__}")
try:
    import skimage
    print(f"Using scikit-image version: {skimage.__version__}")
except ImportError:
    print("[ERROR] scikit-image not found. Please install it: pip install scikit-image")
    exit()

# --- Configuration (Match train.py where necessary) ---
DATASET_PATH = "png_dataset/png_dataset" # Path to the ORIGINAL dataset
FACE_SIZE = (100, 100)                  # Face size used during training
OUTPUT_LBP_IMAGE_PATH = "lbp_visualizations" # Directory to save LBP images
LABEL_MAP_PATH = "label_map_final.json"     # Path to your label map

# LBPH parameters used in train.py (radius and neighbors affect LBP image)
LBP_RADIUS = 1
LBP_NEIGHBORS = 8
LBP_METHOD = 'uniform' # 'uniform' is common for LBPH robustness

# --- Load Face Detector ---
print("[INFO] Loading face detector...")
try:
    detector = dlib.get_frontal_face_detector()
except Exception as e:
    print(f"[ERROR] Failed to load dlib face detector: {e}")
    exit()

# --- Load Label Map ---
print(f"[INFO] Loading label map from: {LABEL_MAP_PATH}")
try:
    with open(LABEL_MAP_PATH, 'r') as f:
        label_to_name_str_keys = json.load(f)
        label_to_name = {int(k): v for k, v in label_to_name_str_keys.items()}
    print(f"[INFO] Loaded {len(label_to_name)} labels.")
except FileNotFoundError:
    print(f"[ERROR] Label map file not found: {LABEL_MAP_PATH}")
    exit()
except Exception as e:
    print(f"[ERROR] Failed to load label map: {e}")
    exit()

# --- Create Output Directory ---
if os.path.exists(OUTPUT_LBP_IMAGE_PATH):
    print(f"[INFO] Removing existing output directory: {OUTPUT_LBP_IMAGE_PATH}")
    shutil.rmtree(OUTPUT_LBP_IMAGE_PATH)
os.makedirs(OUTPUT_LBP_IMAGE_PATH, exist_ok=True)
print(f"[INFO] Created output directory: {OUTPUT_LBP_IMAGE_PATH}")

# --- Process Each Person ---
print("\n[INFO] Generating LBP images...")
processed_count = 0
error_count = 0

persons = sorted([label_to_name[label] for label in label_to_name])

for person_name in persons:
    person_folder_path = os.path.join(DATASET_PATH, person_name)
    print(f"  - Processing: {person_name}")

    representative_image_path = None
    # Find the first valid image file in the person's directory
    try:
        image_files = [f for f in os.listdir(person_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if not image_files:
            print(f"    [WARNING] No image files found for {person_name}. Skipping.")
            error_count += 1
            continue
        # Use the first image found as representative (or choose differently if needed)
        representative_image_path = os.path.join(person_folder_path, image_files[0])
    except FileNotFoundError:
         print(f"    [ERROR] Directory not found for {person_name}: {person_folder_path}. Skipping.")
         error_count += 1
         continue
    except Exception as e:
         print(f"    [ERROR] Error listing files for {person_name}: {e}. Skipping.")
         error_count += 1
         continue

    # --- Load, Detect, Crop, Resize (similar to train.py) ---
    try:
        image = cv2.imread(representative_image_path)
        if image is None:
            print(f"    [ERROR] Could not read image: {representative_image_path}. Skipping.")
            error_count += 1
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected_faces = detector(gray, 1) # Use upsampling=1 as in training

        if not detected_faces:
            print(f"    [WARNING] No face detected in representative image: {image_files[0]}. Skipping.")
            error_count += 1
            continue

        # Use the largest detected face if multiple, or just the first one
        face_rect = max(detected_faces, key=lambda rect: rect.width() * rect.height())
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        x, y = max(0, x), max(0, y)
        if w <= 0 or h <= 0: continue # Skip invalid rects
        y_end = min(y + h, gray.shape[0])
        x_end = min(x + w, gray.shape[1])
        face_roi = gray[y:y_end, x:x_end]

        if face_roi.size == 0:
            print(f"    [WARNING] Empty face ROI for {image_files[0]}. Skipping.")
            error_count += 1
            continue

        resized_face = cv2.resize(face_roi, FACE_SIZE, interpolation=cv2.INTER_AREA)

        # --- Calculate LBP Image ---
        # Use skimage's local_binary_pattern function
        lbp_image = local_binary_pattern(resized_face, LBP_NEIGHBORS, LBP_RADIUS, method=LBP_METHOD)

        # Normalize LBP image to 0-255 range for visualization
        # LBP values can range depending on neighbors, normalize to uint8
        lbp_image_normalized = cv2.normalize(lbp_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # --- Save the LBP Image ---
        output_filename = f"{person_name}_lbp_mask.png"
        output_path = os.path.join(OUTPUT_LBP_IMAGE_PATH, output_filename)
        cv2.imwrite(output_path, lbp_image_normalized)
        # print(f"    [SUCCESS] Saved LBP image to: {output_path}") # Verbose success
        processed_count += 1

    except dlib.error as de:
        print(f"    [ERROR] dlib error processing {image_files[0]}: {de}. Skipping.")
        error_count += 1
    except cv2.error as ce:
         print(f"    [ERROR] OpenCV error processing {image_files[0]}: {ce}. Skipping.")
         error_count += 1
    except Exception as e:
        print(f"    [ERROR] Unexpected error processing {image_files[0]} for {person_name}: {e}. Skipping.")
        error_count += 1


print("\n[INFO] LBP Image Generation Finished.")
print(f"  - Successfully generated LBP images: {processed_count}")
print(f"  - Errors/Skipped: {error_count}")
print(f"  - Output saved to: {OUTPUT_LBP_IMAGE_PATH}")