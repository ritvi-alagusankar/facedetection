import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

UPLOAD_DIRECTORY = "uploaded_images"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def traditional_model(image_path):
    # Assuming this is the placeholder model function for now
    return [["Jose", "Ritvi"], [[100, 200, 100, 100], [200, 300, 100, 100]]]

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Ensure the uploaded file is an image
        if not file.content_type.startswith('image/'):
            return JSONResponse(content={"message": "The uploaded file is not an image."}, status_code=400)

        print(f"Received file: {file.filename}")

        # Save the file with a .jpg extension if necessary
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)

        # If it's not a .jpg file, change the extension
        if not file.filename.endswith(".jpg"):
            file_location = os.path.splitext(file_location)[0] + ".jpg"
        
        # Save the file before calling the model
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the file with your model after saving it
        name_list, boxes_list = traditional_model(file_location)

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