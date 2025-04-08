import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

UPLOAD_DIRECTORY = "uploaded_images"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def traditional_model(image_path):
    return [["Jose", "Ritvi"], [[100, 200, 100, 100], [200, 300, 100, 100]]]

def facenet_model(image_path):
    return [["Jose", "Amisha"], [[100, 200, 100, 100], [200, 300, 100, 100]]]

def deepface_model(image_path):
    return [["Ritvi", "Amisha"], [[100, 200, 100, 100], [200, 300, 100, 100]]]

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), model_type: str = Form(...), deep_learning_model: str = Form(None)):
    try:
        if not file.content_type.startswith('image/'):
            return JSONResponse(content={"message": "The uploaded file is not an image."}, status_code=400)

        print(f"Received file: {file.filename} with model type: {model_type} and deep learning model: {deep_learning_model}")

        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)

        if not file.filename.endswith(".jpg"):
            file_location = os.path.splitext(file_location)[0] + ".jpg"
        
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the file with the appropriate model based on model_type
        if model_type == "traditional":
            name_list, boxes_list = traditional_model(file_location)
        elif model_type == "deep-learning":
            if deep_learning_model == "facenet":
                name_list, boxes_list = facenet_model(file_location)
            elif deep_learning_model == "deepface":
                name_list, boxes_list = deepface_model(file_location)
            else:
                return JSONResponse(content={"message": "Invalid deep learning model specified"}, status_code=400)
        else:
            return JSONResponse(content={"message": "Invalid model type specified"}, status_code=400)

        return JSONResponse(content={"message": "File uploaded successfully", "names": name_list, "bounding": boxes_list}, status_code=200)

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