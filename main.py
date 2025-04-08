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
app = FastAPI()

FACENET_MODEL_PATH = "model/best_face_model.pkl"

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
        print(f"Loaded model with classes: {facenet_class_names}")
except Exception as e:
    print(f"Error loading model: {e}")
    facenet_class_names = []
# Add model caching to avoid reloading the classifier on each request
face_classifier = None

def get_face_classifier():
    global face_classifier
    if face_classifier is None:
        face_classifier = FaceClassifier(embedding_dim=512, num_classes=len(facenet_class_names))
        face_classifier.load_state_dict(facenet_model_data['model'])
        face_classifier.eval()
    return face_classifier

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

def deepface_model(img):
    return [["Ritvi", "Amisha"], [[100, 200, 100, 100], [200, 300, 100, 100]]]

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

