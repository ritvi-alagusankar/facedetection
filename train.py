import os
import cv2
import dlib
import numpy as np
import json
import time
import random 

print(f"Using OpenCV version: {cv2.__version__}")
print(f"Using dlib version: {dlib.__version__}")
print(f"Using Numpy version: {np.__version__}")


dataset_path = "png_dataset/png_dataset"
face_size = (100, 100)
model_filename = "lbph_model_augmented.yml" # Changed filename
label_map_filename = "label_map_augmented.json" # Changed filename

AUGMENT_DATA = True 
AUG_FLIP_PROB = 0.5 
AUG_BRIGHTNESS_CONTRAST_PROB = 0.4 
AUG_CONTRAST_RANGE = (0.85, 1.15) 
AUG_BRIGHTNESS_RANGE = (-15, 15) 

print("[INFO] Loading face detector...")
detector = dlib.get_frontal_face_detector()

def adjust_brightness_contrast(image, alpha_range, beta_range):
    alpha = random.uniform(alpha_range[0], alpha_range[1]) 
    beta = random.uniform(beta_range[0], beta_range[1])   
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def prepare_augment_balance_train(data_folder_path):
    print(f"[INFO] Preparing, Augmenting & Balancing training data from: {data_folder_path}")
    start_time = time.time()

    temp_person_data = {}
    label_to_name_map = {}
    current_label_id = 0
    processed_image_count = 0
    original_faces_count = 0
    augmented_faces_count = 0
    face_detection_failures = 0

    try:
        person_names = sorted(os.listdir(data_folder_path))
    except FileNotFoundError:
        print(f"[ERROR] Dataset directory not found: {data_folder_path}")
        return None, None

    print("[INFO] Phase 1: Initial face detection, processing, and augmentation...")
    for person_name in person_names:
        person_folder_path = os.path.join(data_folder_path, person_name)
        if not os.path.isdir(person_folder_path): continue

        print(f"  - Processing images for: {person_name}")

        # Assign label ID
        person_label = -1
        found_existing = False
        for id, name in label_to_name_map.items():
             if name == person_name:
                  person_label = id; found_existing = True; break
        if not found_existing:
             label_to_name_map[current_label_id] = person_name
             person_label = current_label_id
             temp_person_data[person_label] = []
             current_label_id += 1
        elif person_label not in temp_person_data:
             temp_person_data[person_label] = []

        faces_added_this_person = 0
        for image_name in os.listdir(person_folder_path):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')): continue

            image_path = os.path.join(person_folder_path, image_name)
            image = cv2.imread(image_path)
            if image is None: continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            try:
                detected_faces = detector(gray, 1)
            except Exception:
                 face_detection_failures += 1; continue

            if not detected_faces:
                face_detection_failures += 1; continue

            face_rect = detected_faces[0] 
            x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
            x, y = max(0, x), max(0, y)
            if w <= 0 or h <= 0: continue

            face_roi = gray[y:y+h, x:x+w]
            if face_roi.size == 0: continue

            rois_to_process = [face_roi] 
            original_faces_count += 1

            if AUGMENT_DATA:
                if random.random() < AUG_FLIP_PROB:
                    flipped_roi = cv2.flip(face_roi, 1)
                    rois_to_process.append(flipped_roi)

                temp_rois = []
                for roi in rois_to_process:
                    temp_rois.append(roi) 
                    if random.random() < AUG_BRIGHTNESS_CONTRAST_PROB:
                        adjusted_roi = adjust_brightness_contrast(roi, AUG_CONTRAST_RANGE, AUG_BRIGHTNESS_RANGE)
                        temp_rois.append(adjusted_roi) 
                rois_to_process = temp_rois 

            
            for roi in rois_to_process:
                try:
                    resized_face = cv2.resize(roi, face_size, interpolation=cv2.INTER_AREA)
                    temp_person_data[person_label].append(resized_face)
                    faces_added_this_person += 1
                    processed_image_count += 1
                    if roi is not face_roi: 
                         augmented_faces_count +=1
                except cv2.error: continue  

        print(f"    - Added {faces_added_this_person} faces (including augmentations) for this person.")

    print("[INFO] Phase 1 finished.")
    print(f"[INFO] Original faces processed: {original_faces_count}")
    if AUGMENT_DATA:
        print(f"[INFO] Augmented faces generated: {augmented_faces_count}")
    print(f"[INFO] Total face samples collected (pre-balancing): {processed_image_count}")
    if face_detection_failures > 0:
        print(f"[INFO] Face detection failed/skipped: {face_detection_failures} times.")

    print("\n[INFO] Phase 2: Calculating minimum faces for balancing...")
    min_faces_per_person = float('inf')
    valid_counts = []
    persons_with_faces = 0
    for label, faces in temp_person_data.items():
        count = len(faces)
        person_name = label_to_name_map.get(label, f"Unknown Label {label}")
        print(f"  - Person '{person_name}' (Label {label}): Has {count} samples (orig + aug).")
        if count > 0:
            valid_counts.append(count); persons_with_faces += 1

    if not valid_counts:
        print("\n[ERROR] No valid face samples collected for *any* person.")
        return None, None

    min_faces_per_person = min(valid_counts)
    print(f"[INFO] Minimum number of samples found per person: {min_faces_per_person}")

    if persons_with_faces < 2:
        print("\n[WARNING] Training data contains usable samples for only {persons_with_faces} person(s).")
    if min_faces_per_person < 10: 
        print(f"[WARNING] Minimum sample count ({min_faces_per_person}) is low. Augmentation might help, but consider more base images.")

    print("\n[INFO] Phase 3: Balancing dataset...")
    final_faces_data = []
    final_labels_data = []
    total_faces_after_balancing = 0
    sorted_labels = sorted(temp_person_data.keys())

    for label in sorted_labels:
        faces = temp_person_data[label]
        person_name = label_to_name_map.get(label, f"Unknown Label {label}")
        num_available = len(faces)
        if num_available == 0: continue

        num_to_use = min(num_available, min_faces_per_person)
        selected_faces = random.sample(faces, num_to_use) 

        print(f"  - Person '{person_name}' (Label {label}): {num_available} available, selecting {num_to_use} samples.")
        final_faces_data.extend(selected_faces)
        final_labels_data.extend([label] * num_to_use)
        total_faces_after_balancing += num_to_use

    print(f"[INFO] Total faces included in training after balancing: {total_faces_after_balancing}")

    if not final_faces_data:
        print("\n[ERROR] No faces remaining after balancing. Cannot train.")
        return None, None
    unique_labels_after_balancing = len(set(final_labels_data))
    if unique_labels_after_balancing < 2:
         print(f"\n[WARNING] Only {unique_labels_after_balancing} unique person(s) remain after balancing.")

    end_preparation_time = time.time()
    print(f"\n[INFO] Data preparation, augmentation & balancing finished in {end_preparation_time - start_time:.2f} seconds.")
    print(f"[INFO] Final Label map for included persons: {label_to_name_map}")

    print("\n[INFO] Phase 4: Training LBPH model...")
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

    start_training_time = time.time()
    try:
        recognizer.train(final_faces_data, np.array(final_labels_data, dtype=np.int32))
    except cv2.error as e:
         print(f"\n[ERROR] OpenCV error during training: {e}"); return None, None
    except Exception as e:
         print(f"\n[ERROR] An unexpected error occurred during training: {e}"); return None, None

    end_training_time = time.time()
    print(f"[INFO] LBPH model trained successfully in {end_training_time - start_training_time:.2f} seconds.")

    return recognizer, label_to_name_map

if __name__ == "__main__":
    print("="*40)
    print("Starting LBPH Face Recognizer Training Script")
    print(" (with Data Augmentation & Balancing)")
    print("="*40)

    if not os.path.isdir(dataset_path):
        print(f"[ERROR] Dataset path not found: '{dataset_path}'")
    else:
        trained_recognizer, label_map = prepare_augment_balance_train(dataset_path)

        if trained_recognizer is not None and label_map is not None:
            print("\n[INFO] Saving trained model and label map...")
            try:
                trained_recognizer.save(model_filename)
                print(f"[SUCCESS] Trained model saved to: {model_filename}")
            except Exception as e:
                print(f"[ERROR] Failed to save model '{model_filename}': {e}")
            try:
                with open(label_map_filename, 'w') as f:
                    json.dump(label_map, f, indent=4)
                print(f"[SUCCESS] Label map saved to: {label_map_filename}")
            except Exception as e:
                 print(f"[ERROR] Failed to save label map '{label_map_filename}': {e}")
            print("\n[INFO] Training process completed.")
        else:
            print("\n[ERROR] Model training failed or was aborted.")

    print("="*40)
    print("Training Script Finished.")
    print("="*40)