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


def enroll_faces(dataset_path, db_output_path):
    """
    Process images of known individuals from a dataset folder structure,
    calculate average face embeddings, and save them to a database file.
    
    Args:
        dataset_path: Path to the dataset folder containing subfolders with person images
        db_output_path: Path where the database file should be saved
    """
    print("Starting enrollment process...")
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
        return False
    
    # Dictionary to store embeddings for each person
    known_face_embeddings = {}
    
    # Get person folders (each subfolder is a person)


    person_folders = [f for f in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, f))]
    
    if len(person_folders) == 0:
        print(f"No person folders found in '{dataset_path}'.")
        return False
    
    # Process each person
    for person in person_folders:
        print(f"Enrolling {person}...")
        person_path = os.path.join(dataset_path, person)
        image_files = [f for f in os.listdir(person_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            print(f"No images found for {person}. Skipping.")
            continue
        
        # Store embeddings for this person
        person_embeddings = []
        
        # Process each image for this person
        for img_file in tqdm(image_files, desc=f"Processing {person}'s images"):
            img_path = os.path.join(person_path, img_file)

        try:
                # Extract face embedding
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    detector_backend=ENROLLMENT_DETECTOR,
                    enforce_detection=True
                )
                
                if embedding and len(embedding) > 0:
                    # DeepFace.represent returns a list of embeddings, we typically get the first one
                    person_embeddings.append(embedding[0]["embedding"])
            
        except Exception as e:
            # This could happen if no face is detected or other errors
            print(f"  Warning: Could not process {img_file} - {str(e)}")
        
        # Calculate average embedding if we have at least one valid embedding
        if len(person_embeddings) > 0:
            avg_embedding = np.mean(person_embeddings, axis=0)
            known_face_embeddings[person] = avg_embedding
            print(f"  Successfully enrolled {person} with {len(person_embeddings)} images.")
        else:
            print(f"  Failed to enroll {person}. No valid face embeddings found.")
    
    # Save embeddings to database file
    if len(known_face_embeddings) > 0:
        np.savez_compressed(db_output_path, **known_face_embeddings)
        print(f"Saved database with {len(known_face_embeddings)} people to {db_output_path}")
        return True
    else:
        print("No valid embeddings found. Database not created.")
        return False




        

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

def traditional_model(img):
    return [["Jose", "Ritvi"], [[100, 200, 100, 100], [200, 300, 100, 100]]]


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
                label = best_match if best_similarity >= SIMILARITY_THRESHOLD else "Unknown"
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
async def upload_file(file: UploadFile = File(...), model_type: str = Form(...), deep_learning_model: str = Form(None)):
    try:
        if not file.content_type.startswith('image/'):
            return JSONResponse(content={"message": "The uploaded file is not an image."}, status_code=400)

        print(f"Received file: {file.filename} with model type: {model_type} and deep learning model: {deep_learning_model}")

        # Read image data directly into memory instead of saving to disk
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Process the image with the appropriate model based on model_type
        if model_type == "traditional":
            name_list, boxes_list = traditional_model(img)
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

