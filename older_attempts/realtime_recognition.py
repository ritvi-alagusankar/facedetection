import os
import cv2
import dlib
import numpy as np
import joblib
import time

# ====================== CONFIGURATION ======================
OUTPUT_DIR = "./output"  # Directory where models are saved
FACE_SIZE = (100, 100)   # Must match the size used during training
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence threshold for recognition

# ====================== LOAD MODELS ======================
print("[INFO] Loading face recognition model...")
model = joblib.load(os.path.join(OUTPUT_DIR, "face_recognition_model.pkl"))
le = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
detector_type = joblib.load(os.path.join(OUTPUT_DIR, "detector_type.pkl"))

# Load appropriate face detector
if detector_type == "haar":
    print("[INFO] Using Haar Cascade detector")
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
else:
    print("[INFO] Using HOG detector")
    face_detector = dlib.get_frontal_face_detector()

# ====================== HELPER FUNCTIONS ======================
def detect_faces(frame, detector_type="haar"):
    """Detect faces in the input frame using the specified detector"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    if detector_type == "haar":
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        # Convert to list of (x, y, w, h) tuples
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    elif detector_type == "hog":
        dlib_faces = face_detector(gray, 1)  # 1 = upsample once
        # Convert dlib rectangle to (x, y, w, h) format
        return [(f.left(), f.top(), f.width(), f.height()) for f in dlib_faces]
    
    return []

def preprocess_face(face):
    """Apply same preprocessing as in training"""
    face_resized = cv2.resize(face, FACE_SIZE)
    face_processed = cv2.equalizeHist(face_resized)
    return face_processed

def predict_face(face, model, le):
    """Recognize face using the trained model"""
    # Flatten the face image to a 1D vector
    face_vector = face.flatten().reshape(1, -1)
    
    # Get prediction and probability
    prediction = model.predict(face_vector)[0]
    proba = model.predict_proba(face_vector)[0]
    
    # Get the predicted label and confidence
    max_proba_idx = np.argmax(proba)
    confidence = proba[max_proba_idx]
    
    if confidence >= CONFIDENCE_THRESHOLD:
        name = le.inverse_transform([prediction])[0]
    else:
        name = "Unknown"
        
    return name, confidence

# ====================== MAIN WEBCAM FUNCTION ======================
def run_webcam_recognition():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return
    
    print("[INFO] Starting webcam face recognition. Press 'q' to quit.")
    
    # FPS calculation variables
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break
        
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Detect faces
        face_locations = detect_faces(frame, detector_type)
        
        # Process each detected face
        for (x, y, w, h) in face_locations:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Convert to grayscale
            if len(face_roi.shape) == 3:
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Preprocess face
            processed_face = preprocess_face(face_roi)
            
            # Predict identity
            name, confidence = predict_face(processed_face, model, le)
            
            # Draw bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            # Prepare label with name and confidence
            label = f"{name}: {confidence:.2f}"
            
            # Calculate text position and size for background rectangle
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw background rectangle for text
            cv2.rectangle(display_frame, 
                        (x, y - label_size[1] - 10), 
                        (x + label_size[0], y), 
                        color, cv2.FILLED)
            
            # Display name and confidence
            cv2.putText(display_frame, label, (x, y - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Calculate FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Draw FPS info
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                  
        # Show the detector type
        cv2.putText(display_frame, f"Detector: {detector_type.upper()}", (10, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow("Face Recognition", display_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_recognition()