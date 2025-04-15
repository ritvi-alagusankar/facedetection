import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns
import random
import joblib
from tqdm import tqdm

# ====================== CONFIGURATION ======================
DATASET_DIR = "../dataset"  # Path to your dataset directory
OUTPUT_DIR = "./output"     # Directory to save models and visualizations
FACE_SIZE = (100, 100)      # Standard size for all face images
RANDOM_SEED = 42            # For reproducibility

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== FACE DETECTION ======================
# Initialize face detectors
haar_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hog_detector = dlib.get_frontal_face_detector()

# ====================== DATA AUGMENTATION FUNCTIONS ======================
def augment_image(image):
    """Generate augmented versions of the input image"""
    augmented = []
    
    # Original image
    augmented.append(image)
    
    # Horizontal flip
    augmented.append(cv2.flip(image, 1))
    
    # Brightness variations
    for alpha in [0.8, 1.2]:  # Darker and brighter
        augmented.append(cv2.convertScaleAbs(image, alpha=alpha, beta=0))
    
    # Small rotations
    for angle in [-10, 10]:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotated)
    
    # Small translations
    for tx, ty in [(5, 0), (-5, 0), (0, 5), (0, -5)]:
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]),
                                  borderMode=cv2.BORDER_REPLICATE)
        augmented.append(translated)
    
    # Slight noise addition
    for intensity in [5, 10]:
        noise = np.random.normal(0, intensity, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        augmented.append(noisy)
    
    return augmented

# Helper function for gamma correction
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# ====================== FACE DETECTION & PREPROCESSING ======================
def detect_face(image, detector_type="haar"):
    """Detect face using specified detector and return the face region"""
    if image is None:
        return None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    if detector_type == "haar":
        faces = haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            # Get the largest face
            face = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]
            x, y, w, h = face
            return gray[y:y+h, x:x+w]
        
    elif detector_type == "hog":
        faces = hog_detector(gray, 1)  # 1 = upsample once for better detection of small faces
        if len(faces) > 0:
            # Get the largest face
            face = sorted(faces, key=lambda x: x.width() * x.height(), reverse=True)[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            # Check for valid coordinates (might be out of bounds)
            if x >= 0 and y >= 0 and w > 0 and h > 0 and x+w <= gray.shape[1] and y+h <= gray.shape[0]:
                return gray[y:y+h, x:x+w]
    
    return None

def preprocess_face(face):
    """Apply preprocessing to detected face"""
    if face is None:
        return None
    
    try:
        # Resize to standard size
        face_resized = cv2.resize(face, FACE_SIZE)
        
        # Apply additional preprocessing
        face_processed = cv2.equalizeHist(face_resized)
        
        return face_processed
    except Exception as e:
        # If there's an error, return None
        print(f"Error in preprocessing: {e}")
        return None

# ====================== LOAD DATASET ======================
def load_dataset(dataset_dir, detector_type="haar", apply_augmentation=True):
    """Load face dataset with optional augmentation"""
    faces = []
    labels = []
    class_counts = {}
    error_count = 0
    
    print(f"[INFO] Loading dataset using {detector_type.upper()} detector...")
    
    for person in sorted(os.listdir(dataset_dir)):
        person_path = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_path):
            continue
        
        print(f"[INFO] Processing class: {person}")
        class_count = 0
        
        # Process each image in the person's directory
        for img_name in tqdm(os.listdir(person_path)):
            img_path = os.path.join(person_path, img_name)
            try:
                image = cv2.imread(img_path)
                
                if image is None:
                    continue
                    
                # Detect and preprocess face
                face = detect_face(image, detector_type)
                if face is not None:
                    face_processed = preprocess_face(face)
                    
                    if face_processed is not None:
                        # Original face
                        faces.append(face_processed)
                        labels.append(person)
                        class_count += 1
                        
                        # Apply augmentation if enabled
                        if apply_augmentation:
                            # Only augment half of the detected faces to maintain balance
                            if random.random() < 0.5:
                                augmented_faces = augment_image(face_processed)
                                # Add only 2 random augmentations to avoid class imbalance
                                selected_augmentations = random.sample(augmented_faces[1:], 2)
                                for aug_face in selected_augmentations:
                                    faces.append(aug_face)
                                    labels.append(person)
                                    class_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                error_count += 1
        
        class_counts[person] = class_count
        print(f"[INFO] Class {person}: {class_count} images (including augmentations)")
    
    print(f"[INFO] Total faces processed: {len(faces)}")
    print(f"[INFO] Total errors/skipped images: {error_count}")
    
    if len(faces) == 0:
        raise ValueError(f"No faces were detected with {detector_type} detector!")
    
    return np.array(faces), np.array(labels), class_counts

# ====================== MAIN EXECUTION ======================
def main():
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # Try loading dataset with Haar first
    try:
        print("\n=== LOADING DATASET WITH HAAR CASCADE DETECTOR ===")
        haar_faces, haar_labels, haar_counts = load_dataset(DATASET_DIR, detector_type="haar")
        haar_success = True
    except Exception as e:
        print(f"[ERROR] Failed to load dataset with Haar detector: {e}")
        haar_success = False
        haar_faces, haar_labels, haar_counts = np.array([]), np.array([]), {}
    
    # Try loading dataset with HOG
    try:
        print("\n=== LOADING DATASET WITH HOG DETECTOR ===")
        hog_faces, hog_labels, hog_counts = load_dataset(DATASET_DIR, detector_type="hog")
        hog_success = True
    except Exception as e:
        print(f"[ERROR] Failed to load dataset with HOG detector: {e}")
        hog_success = False
        hog_faces, hog_labels, hog_counts = np.array([]), np.array([]), {}
    
    # Choose the detector that succeeded or found more faces
    if haar_success and (not hog_success or len(haar_faces) >= len(hog_faces)):
        print("\n[INFO] Using Haar Cascade results")
        faces, labels, class_counts = haar_faces, haar_labels, haar_counts
        detector_type = "haar"
    elif hog_success:
        print("\n[INFO] Using HOG results")
        faces, labels, class_counts = hog_faces, hog_labels, hog_counts
        detector_type = "hog"
    else:
        raise ValueError("Both detectors failed to process the dataset!")
    
    # Visualize class distribution
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Number of Images per Class (Including Augmentations)")
    plt.xlabel("Person")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
    
    # Visualize sample faces
    plt.figure(figsize=(15, 8))
    unique_labels = sorted(set(labels))
    for i, label in enumerate(unique_labels):
        # Find images with this label
        indices = np.where(labels == label)[0]
        if len(indices) > 0:
            # Show one sample from each class
            idx = indices[0]
            plt.subplot(2, 4, i+1)
            plt.imshow(faces[idx], cmap='gray')
            plt.title(f"Class: {label}")
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_faces.png"))
    
    # Convert faces to feature vectors
    face_vectors = [face.flatten() for face in faces]
    X = np.array(face_vectors)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Split into training and testing sets (75:25)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    
    # Reshape back to images for visualization
    n_samples, h, w = len(faces), FACE_SIZE[0], FACE_SIZE[1]
    X_train_img = X_train.reshape(-1, h, w)
    X_test_img = X_test.reshape(-1, h, w)
    
    print(f"[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Testing set: {X_test.shape[0]} samples")
    
    # ====================== TRAIN EIGENFACES MODEL ======================
    print("\n=== TRAINING EIGENFACES (PCA) MODEL ===")
    # Determine optimal number of components to retain 95% variance
    temp_pca = PCA().fit(X_train)
    cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= 0.95) + 1
    n_components = min(n_components, X_train.shape[0] - 1)  # Ensure valid number of components
    print(f"[INFO] Using {n_components} eigenfaces to retain 95% variance")
    
    # Define the eigenfaces pipeline
    eigenface_model = Pipeline([
        ('pca', PCA(n_components=n_components, whiten=True)),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
    ])
    
    # Train the model
    eigenface_model.fit(X_train, y_train)
    print("[INFO] Eigenfaces model trained")
    
    # Visualize eigenfaces
    pca = eigenface_model.named_steps['pca']
    plt.figure(figsize=(15, 8))
    for i in range(min(16, n_components)):
        plt.subplot(4, 4, i+1)
        eigenface = pca.components_[i].reshape(FACE_SIZE)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f"Eigenface {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eigenfaces.png"))
    
    # ====================== TRAIN FISHERFACES MODEL ======================
    print("\n=== TRAINING FISHERFACES (LDA) MODEL ===")
    # Maximum number of components for LDA is (n_classes - 1)
    n_classes = len(np.unique(y_train))
    n_fisherfaces = min(n_classes - 1, 50)  # Limit to 50 max
    
    try:
        # First reduce dimensionality with PCA to avoid singular matrix issues
        fisherface_model = Pipeline([
            ('pca', PCA(n_components=min(X_train.shape[0], X_train.shape[1]), whiten=True)),
            ('lda', LDA(n_components=n_fisherfaces)),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
        ])
        
        # Train the model
        fisherface_model.fit(X_train, y_train)
        print("[INFO] Fisherfaces model trained")
        
        # Visualize fisherfaces
        lda = fisherface_model.named_steps['lda']
        plt.figure(figsize=(15, 8))
        for i in range(min(8, n_fisherfaces)):
            plt.subplot(2, 4, i+1)
            fisherface = lda.scalings_[:, i].reshape(FACE_SIZE)
            plt.imshow(fisherface, cmap='gray')
            plt.title(f"Fisherface {i+1}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "fisherfaces.png"))
        
        fisherfaces_trained = True
    except Exception as e:
        print(f"[WARNING] Fisherfaces training failed: {e}")
        fisherfaces_trained = False
    
    # ====================== EVALUATE MODELS ======================
    # Evaluate Eigenfaces
    print("\n=== EVALUATING EIGENFACES MODEL ===")
    eigen_predictions = eigenface_model.predict(X_test)
    eigen_accuracy = np.mean(eigen_predictions == y_test) * 100
    print(f"[INFO] Eigenfaces Accuracy: {eigen_accuracy:.2f}%")
    
    # Display confusion matrix for Eigenfaces
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, eigen_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Eigenfaces Confusion Matrix (Accuracy: {eigen_accuracy:.2f}%)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eigenfaces_confusion_matrix.png"))
    
    # Display classification report for Eigenfaces
    eigen_report = classification_report(y_test, eigen_predictions, target_names=le.classes_)
    print("\nEigenfaces Classification Report:")
    print(eigen_report)
    
    # Evaluate Fisherfaces if training succeeded
    if fisherfaces_trained:
        print("\n=== EVALUATING FISHERFACES MODEL ===")
        fisher_predictions = fisherface_model.predict(X_test)
        fisher_accuracy = np.mean(fisher_predictions == y_test) * 100
        print(f"[INFO] Fisherfaces Accuracy: {fisher_accuracy:.2f}%")
        
        # Display confusion matrix for Fisherfaces
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, fisher_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Fisherfaces Confusion Matrix (Accuracy: {fisher_accuracy:.2f}%)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "fisherfaces_confusion_matrix.png"))
        
        # Display classification report for Fisherfaces
        fisher_report = classification_report(y_test, fisher_predictions, target_names=le.classes_)
        print("\nFisherfaces Classification Report:")
        print(fisher_report)
        
        # Select the best model
        best_model = eigenface_model if eigen_accuracy >= fisher_accuracy else fisherface_model
        best_model_name = "Eigenfaces" if eigen_accuracy >= fisher_accuracy else "Fisherfaces"
    else:
        # If Fisherfaces failed, use Eigenfaces
        best_model = eigenface_model
        best_model_name = "Eigenfaces"
    
    print(f"\n[INFO] Best model: {best_model_name} with accuracy {eigen_accuracy:.2f}%")
    
    # Save the best model and necessary components
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "face_recognition_model.pkl"))
    joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
    joblib.dump(detector_type, os.path.join(OUTPUT_DIR, "detector_type.pkl"))
    
    print(f"\n[INFO] Model and components saved to {OUTPUT_DIR}")
    print("[INFO] Ready for real-time face recognition")
