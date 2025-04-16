import os
import cv2
import dlib
import numpy as np
import json
from skimage.feature import local_binary_pattern # Using scikit-image for LBP calculation
import math # Added for grid layout calculations

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
OUTPUT_COMPOSITE_IMAGE_PATH = "lbp_composite_visualization.png" # Single output file
LABEL_MAP_PATH = "label_map_final.json"     # Path to your label map

# LBPH parameters used in train.py (radius and neighbors affect LBP image)
LBP_RADIUS = 1
LBP_NEIGHBORS = 8
LBP_METHOD = 'uniform' # 'uniform' is common for LBPH robustness

# --- Layout Configuration ---
GRID_COLS = 4 # Adjust as needed (e.g., if you have more/fewer people)
PADDING = 20      # Pixels around each image
LABEL_HEIGHT = 30 # Space above each image for the label
FONT_SCALE = 0.6
FONT_THICKNESS = 1
BACKGROUND_COLOR = (255, 255, 255) # White background (BGR)
TEXT_COLOR = (0, 0, 0)         # Black text (BGR)


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

# --- List to store results ---
lbp_results = [] # Will store tuples: (person_name, lbp_image_normalized)

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
        # Use the first image found as representative
        representative_image_path = os.path.join(person_folder_path, image_files[0])
    except FileNotFoundError:
         print(f"    [ERROR] Directory not found for {person_name}: {person_folder_path}. Skipping.")
         error_count += 1
         continue
    except Exception as e:
         print(f"    [ERROR] Error listing files for {person_name}: {e}. Skipping.")
         error_count += 1
         continue

    # --- Load, Detect, Crop, Resize ---
    try:
        image = cv2.imread(representative_image_path)
        if image is None:
            print(f"    [ERROR] Could not read image: {representative_image_path}. Skipping.")
            error_count += 1
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = detector(gray, 1)

        if not detected_faces:
            print(f"    [WARNING] No face detected in representative image: {image_files[0]}. Skipping.")
            error_count += 1
            continue

        face_rect = max(detected_faces, key=lambda rect: rect.width() * rect.height())
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        x, y = max(0, x), max(0, y)
        if w <= 0 or h <= 0: continue
        y_end = min(y + h, gray.shape[0])
        x_end = min(x + w, gray.shape[1])
        face_roi = gray[y:y_end, x:x_end]

        if face_roi.size == 0:
            print(f"    [WARNING] Empty face ROI for {image_files[0]}. Skipping.")
            error_count += 1
            continue

        resized_face = cv2.resize(face_roi, FACE_SIZE, interpolation=cv2.INTER_AREA)

        # --- Calculate LBP Image ---
        lbp_image = local_binary_pattern(resized_face, LBP_NEIGHBORS, LBP_RADIUS, method=LBP_METHOD)
        lbp_image_normalized = cv2.normalize(lbp_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # --- Store Result ---
        lbp_results.append((person_name, lbp_image_normalized))
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

# --- Create Composite Image ---
if not lbp_results:
    print("[ERROR] No LBP images were generated. Cannot create composite image.")
else:
    print("\n[INFO] Creating composite visualization...")
    num_images = len(lbp_results)
    grid_cols = GRID_COLS
    grid_rows = math.ceil(num_images / grid_cols)

    # Calculate canvas size
    img_h, img_w = FACE_SIZE[1], FACE_SIZE[0]
    canvas_h = PADDING + grid_rows * (img_h + PADDING + LABEL_HEIGHT)
    canvas_w = PADDING + grid_cols * (img_w + PADDING)

    # Create blank canvas (BGR for color text)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * BACKGROUND_COLOR[0] # Assume BGR values are same for white

    current_row, current_col = 0, 0
    for i, (name, lbp_img) in enumerate(lbp_results):
        # Convert grayscale LBP to BGR to place on canvas
        lbp_img_bgr = cv2.cvtColor(lbp_img, cv2.COLOR_GRAY2BGR)

        # Calculate positions
        x_start = PADDING + current_col * (img_w + PADDING)
        y_start_label = PADDING + current_row * (img_h + PADDING + LABEL_HEIGHT)
        y_start_image = y_start_label + LABEL_HEIGHT

        x_end = x_start + img_w
        y_end = y_start_image + img_h

        # Place image
        if y_end <= canvas_h and x_end <= canvas_w: # Ensure it fits
            canvas[y_start_image:y_end, x_start:x_end] = lbp_img_bgr
        else:
            print(f"[WARN] Image for {name} doesn't fit on canvas. Skipping placement.")
            continue


        # Add text label above the image
        text_x = x_start + 5 # Small offset from left edge
        text_y = y_start_label + int(LABEL_HEIGHT * 0.7) # Position within the label space
        cv2.putText(canvas, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # Move to next grid position
        current_col += 1
        if current_col >= grid_cols:
            current_col = 0
            current_row += 1

    # Save the final composite image
    try:
        cv2.imwrite(OUTPUT_COMPOSITE_IMAGE_PATH, canvas)
        print(f"[SUCCESS] Composite LBP visualization saved to: {OUTPUT_COMPOSITE_IMAGE_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to save composite image: {e}")

print("\n[INFO] Script Finished.")