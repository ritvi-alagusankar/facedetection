import os
import cv2
import dlib
import numpy as np
import joblib
import time

# ====================== CONFIGURATION ======================
OUTPUT_DIR = "./output"  # directory where models are saved
FACE_SIZE = (100, 100)   # match the size used during training
CONFIDENCE_THRESHOLD = 0.55  # minimum confidence threshold for recognition

# ====================== LOAD MODELS ======================
print("[INFO] Loading face recognition model...")
model = joblib.load(os.path.join(OUTPUT_DIR, "face_recognition_model.pkl"))
le = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
detector_type = joblib.load(os.path.join(OUTPUT_DIR, "detector_type.pkl"))

# load the appropriate face detector
if detector_type == "haar":
    print("[INFO] Using Haar Cascade detector")
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
else:
    print("[INFO] Using HOG detector")
    face_detector = dlib.get_frontal_face_detector()

# ====================== HELPER FUNCTIONS ======================
def detect_faces(frame, detector_type="haar"):
    """detect faces in the input frame using specified detector"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    if detector_type == "haar":
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        # convert to list of (x, y, w, h) tuples
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    elif detector_type == "hog":
        dlib_faces = face_detector(gray, 1)  # 1 = upsample once
        # convert dlib rectangle to (x, y, w, h) format
        return [(f.left(), f.top(), f.width(), f.height()) for f in dlib_faces]
    
    return []

def preprocess_face(face):
    """apply same preprocessing as training"""
    face_resized = cv2.resize(face, FACE_SIZE)
    face_processed = cv2.equalizeHist(face_resized)
    return face_processed

def predict_face(face, model, le):
    """recognize face using the trained model"""
    # flatten face image to a 1D vector
    face_vector = face.flatten().reshape(1, -1)
    
    # get prediction and probability
    prediction = model.predict(face_vector)[0]
    proba = model.predict_proba(face_vector)[0]
    
    # get predicted label and confidence
    max_proba_idx = np.argmax(proba)
    confidence = proba[max_proba_idx]
    
    if confidence >= CONFIDENCE_THRESHOLD:
        name = le.inverse_transform([prediction])[0]
    else:
        name = "Unknown"
        
    return name, confidence

# ====================== MAIN WEBCAM FUNCTION ======================
def run_webcam_recognition():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return
    
    print("[INFO] Starting webcam face recognition. Press 'q' to quit.")
    
    # FPS calculation variables
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break
        
        display_frame = frame.copy()
        
        # detect faces
        face_locations = detect_faces(frame, detector_type)
        
        # process each detected face
        for (x, y, w, h) in face_locations:
            # extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # convert to grayscale
            if len(face_roi.shape) == 3:
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # preprocess face
            processed_face = preprocess_face(face_roi)
            
            # predict identity
            name, confidence = predict_face(processed_face, model, le)
            
            # draw bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            # prepare label with name and confidence
            label = f"{name}: {confidence:.2f}"
            
            # calculate text position and size for background rectangle
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # draw background rectangle for text
            cv2.rectangle(display_frame, 
                        (x, y - label_size[1] - 10), 
                        (x + label_size[0], y), 
                        color, cv2.FILLED)
            
            # display name and confidence
            cv2.putText(display_frame, label, (x, y - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                  
        # show detector type
        cv2.putText(display_frame, f"Detector: {detector_type.upper()}", (10, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # display resulting frame
        cv2.imshow("Face Recognition", display_frame)
        
        # exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

'''
if __name__ == "__main__":
    run_webcam_recognition()'''