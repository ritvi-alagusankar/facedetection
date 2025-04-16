import os
import pickle
import io
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import cv2
from deepface import DeepFace
from tqdm import tqdm
import argparse
import dlib
import joblib
app = FastAPI()

FACENET_MODEL_PATH = "model/best_face_model.pkl"

# Configuration parameters
DATASET_PATH = './dataset/'
DB_PATH = 'known_faces_db.npz'
MODEL_NAME = 'ArcFace'
ENROLLMENT_DETECTOR = 'retinaface'
REALTIME_DETECTOR = 'yunet'
SIMILARITY_THRESHOLD = 0.60  # Can adjust based on testing
FPS_UPDATE_INTERVAL = 1  # Update FPS calculation every second

facenet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=40, keep_all=True, thresholds=[0.4, 0.5, 0.6], min_face_size=15)

# Image transformations
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

try:
    with open(FACENET_MODEL_PATH, 'rb') as f:
        facenet_model_data = pickle.load(f)
        facenet_class_names = facenet_model_data['class_names']
        print(f"Loaded facenet model with classes: {facenet_class_names}")
        OUTPUT_DIR = "./model"  # directory where models are saved
        FACE_SIZE = (100, 100)   # match the size used during training
        CONFIDENCE_THRESHOLD = 0.55  # minimum confidence threshold for recognition

        print("[INFO] Loading traditional face recognition model...")
        model = joblib.load(os.path.join(OUTPUT_DIR, "face_recognition_model.pkl"))
        le = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
        detector_type = joblib.load(os.path.join(OUTPUT_DIR, "detector_type.pkl"))
        face_detector_hog = dlib.get_frontal_face_detector()
        face_detector_viola = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("[INFO] Traditional face recognition model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    facenet_class_names = []
# Add model caching to avoid reloading the classifier on each request
face_classifier = None
known_faces_db = None
try:
    # Attempt to load the DeepFace model for face recognition
    known_faces_db = np.load(DB_PATH)
    model_data = DeepFace.build_model(MODEL_NAME)
    print(f"Loaded {MODEL_NAME} model successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model_data = None
# Add model caching to avoid reloading the model on each request
deepface_model = model_data

def get_face_classifier():
    global face_classifier
    if face_classifier is None:
        face_classifier = FaceClassifier(embedding_dim=512, num_classes=len(facenet_class_names))
        face_classifier.load_state_dict(facenet_model_data['model'])
        face_classifier.eval()
    return face_classifier

def get_deepface_model(model_name='ArcFace'):
    global deepface_model
    if deepface_model is None:
        try:
            # Load the DeepFace model for the specified model name
            deepface_model = DeepFace.build_model(model_name)
            print(f"Successfully loaded {model_name} model.")
        except Exception as e:
            print(f"Error loading {model_name} model: {e}")
            deepface_model = None
    return deepface_model

class FaceClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout_rate=0.5):
        super(FaceClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


       

def calculate_cosine_similarity(embedding1, embedding2):
    """
    Calculate the cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity (1 = identical, 0 = completely different)
    """
    # Normalize embeddings to unit vectors
    embedding1_normalized = embedding1 / np.linalg.norm(embedding1)
    embedding2_normalized = embedding2 / np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1_normalized, embedding2_normalized)
    
    return similarity

def detect_faces(frame, facedetector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    gray = cv2.equalizeHist(gray)
    if facedetector == "hog":
        dlib_faces = face_detector_hog(gray, 1)  # 1 = upsample once
        return [(f.left(), f.top(), f.width(), f.height()) for f in dlib_faces]
    elif facedetector == "viola-jones":
        faces = face_detector_viola.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        # convert to list of (x, y, w, h) tuples
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    else:
        raise ValueError(f"Unsupported face detector: {facedetector}")

def preprocess_face(face):
    face_resized = cv2.resize(face, FACE_SIZE)
    face_processed = cv2.equalizeHist(face_resized)
    return face_processed

def predict_face(face, model, le):
    """recognize face using the trained model"""
    face_vector = face.flatten().reshape(1, -1)
    
    prediction = model.predict(face_vector)[0]
    proba = model.predict_proba(face_vector)[0]
    
    max_proba_idx = np.argmax(proba)
    confidence = proba[max_proba_idx]
    
    if confidence >= CONFIDENCE_THRESHOLD:
        name = le.inverse_transform([prediction])[0]
    else:
        name = "Unknown"
        
    return name, confidence

def eigenfaces(img, facedetector):
    boxes, names = [], []  # Placeholder for traditional model
    img = np.array(img)  # Convert PIL Image to numpy array
    if facedetector is None:
        print("[ERROR] Face detector not initialized")
        return names, boxes
    else:
        face_locations = detect_faces(img, facedetector)
    print(face_locations)  # Debugging line to check detected face locations
    # process each detected face
    for (x, y, w, h) in face_locations:
        # extract face ROI
        face_roi = img[y:y+h, x:x+w]
    
        # convert to grayscale
        if len(face_roi.shape) == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # preprocess face
        processed_face = preprocess_face(face_roi)
        
        # predict identity
        name, confidence = predict_face(processed_face, model, le)
        if confidence >= CONFIDENCE_THRESHOLD:
            names.append(f"{name} ({confidence:.2f})")
            boxes.append([x, y, w, h])
        else:
            names.append("Unknown")
            boxes.append([x, y, w, h])
    print(names, boxes)  # Debugging line to check names and boxes
    return names, boxes  # Return empty lists for names and boxes 


def facenet_model(img):
    def recognize_face(face_img, model, facenet_class_names):
        if face_img is None or face_img.shape[0] == 0 or face_img.shape[1] == 0:
            return "Unknown"

        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (160, 160))
        face_tensor = test_transform(face_img).unsqueeze(0)
        
        with torch.no_grad():
            embedding = facenet(face_tensor)
            output = model(embedding)
            
            if isinstance(model, FaceClassifier):
                output = F.softmax(output, dim=1)
                
            pred_idx = torch.argmax(output, dim=1).item()
            confidence = torch.max(output).item()
            
            label = f"{facenet_class_names[pred_idx]} ({confidence:.2f})"
            return label if confidence > 0.6 else "Unknown"
    try:      
        # Detect faces using MTCNN
        boxes, _ = mtcnn.detect(img)
        
        if boxes is None:
            result = [[], []]
            return result
            
        # Convert boxes to list format
        boxes_list = []
        names = []
        img = np.array(img)

        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            boxes_list.append([int(x1), int(y1), int(width), int(height)])
            
            # Extract face and get embedding
            face = img[int(y1):int(y2), int(x1):int(x2)]
            if face is not None:
                label = recognize_face(face, get_face_classifier(), facenet_class_names)
                names.append(label)
            else:
                names.append("Unknown")
        
        result = [names, boxes_list]
        return result
    except Exception as e:
        print(f"Error in facenet_model: {str(e)}")
        return [[], []]

def run_recognition(img, db_path):
    
        try:
            # Convert image to OpenCV format
            # Convert image to OpenCV format
            img_cv2 = np.array(img)  # Convert PIL Image to numpy array
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
            
            
            faces = DeepFace.extract_faces(
                img_path=img_cv2,
                detector_backend="yunet",
                align=True,
                enforce_detection=False  # Prevent crashing if no face detected
            )

            if not faces:
                return [[], []]  # No faces detected

            names = []
            boxes_list = []

            for face_obj in faces:
                confidence = face_obj.get("confidence", 0)
                if confidence < 0.8:  # Skip low-confidence detections
                    continue

                face_img = face_obj["face"]
                facial_area = face_obj["facial_area"]

                # Extract embedding using DeepFace
                embedding_data = DeepFace.represent(
                    img_path=face_img,
                    model_name="ArcFace",
                    detector_backend="skip",  # No need to re-detect
                    enforce_detection=False
                )

                if not embedding_data:
                    names.append("Unknown")
                    continue

                embedding_vector = embedding_data[0]["embedding"]


                best_match = "Unknown"
                best_similarity = 0

                for person in known_faces_db.files:
                    known_embedding = known_faces_db[person]
                    similarity = calculate_cosine_similarity(embedding_vector, known_embedding)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = person

                # Apply similarity threshold
                label = best_match if best_similarity >= 0.5 else "Unknown"
                if label != "Unknown":
                    names.append(f"{label} ({best_similarity:.2f})")
                else:
                    names.append(label)

                # Get bounding box
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                boxes_list.append([x, y, w, h])

            return [names, boxes_list]

        except Exception as e:
            print(f"Error in deepface_model: {str(e)}")
            return [[], []]


def deepface_model(img):

    try:
        result = run_recognition(img, DB_PATH)
        if result is None:
            return [[], []]  # Ensure it returns an empty list instead of None
        return result
    except Exception as e:
        print(f"Error in deepface_model: {str(e)}")
        return [[], []]  # Return valid format even if an error occurs

    

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), model_type: str = Form(...), deep_learning_model: str = Form(None), traditional_detection_model: str = Form(None), traditional_recognition_model: str = Form(None)):
    try:
        if not file.content_type.startswith('image/'):
            return JSONResponse(content={"message": "The uploaded file is not an image."}, status_code=400)

        print(f"Received file: {file.filename} with model type: {model_type} and deep learning model: {deep_learning_model}")

        # Read image data directly into memory instead of saving to disk
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Process the image with the appropriate model based on model_type
        if model_type == "traditional":
            if traditional_recognition_model == "eigenfaces":
                name_list, boxes_list = eigenfaces(img, traditional_detection_model)
            elif traditional_recognition_model == "fisherfaces":
                name_list, boxes_list = eigenfaces(img, traditional_detection_model)
            elif traditional_recognition_model == "lbph":
                name_list, boxes_list = eigenfaces(img, traditional_detection_model)
            else:
                return JSONResponse(content={"message": "Invalid traditional recognition model specified"}, status_code=400)
        elif model_type == "deep-learning":
            if deep_learning_model == "facenet":
                name_list, boxes_list = facenet_model(img)
            elif deep_learning_model == "deepface":
                name_list, boxes_list = deepface_model(img)
            else:
                return JSONResponse(content={"message": "Invalid deep learning model specified"}, status_code=400)
        else:
            return JSONResponse(content={"message": "Invalid model type specified"}, status_code=400)

        return JSONResponse(content={"message": "Image processed successfully", "names": name_list, "bounding": boxes_list}, status_code=200)

    except Exception as e:
        print(f"Error occurred: {e}")
        return JSONResponse(content={"message": str(e)}, status_code=500)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend requests from this address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

