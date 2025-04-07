import os
import time
import cv2
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

# Configuration parameters
DATASET_PATH = './dataset/'
DB_PATH = 'known_faces_db.npz'
MODEL_NAME = 'ArcFace'
ENROLLMENT_DETECTOR = 'retinaface'
REALTIME_DETECTOR = 'yunet'
SIMILARITY_THRESHOLD = 0.60  # Can adjust based on testing
FPS_UPDATE_INTERVAL = 1  # Update FPS calculation every second

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

def run_recognition(db_path):
    """
    Real-time face recognition using webcam.
    
    Args:
        db_path: Path to the database file containing known face embeddings
    """
    # Load the known faces database
    try:
        known_faces_db = np.load(db_path)
        if len(known_faces_db.files) == 0:
            print("Error: Empty database loaded.")
            return
        print(f"Loaded database with {len(known_faces_db.files)} people.")
    except Exception as e:
        print(f"Error loading database: {str(e)}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get webcam resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {width}x{height}")
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    print("Starting real-time face recognition...")
    print("Press 'q' to quit.")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break
        
        # Frame counter for FPS calculation
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Update FPS calculation every second
        if elapsed_time >= FPS_UPDATE_INTERVAL:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = current_time
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        try:
            # Detect faces in the frame (using a fast detector for real-time)
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=REALTIME_DETECTOR,
                align=True,
                enforce_detection=False
            )
            
            # Process each detected face
            for face_obj in faces:
                # Check confidence (only process if confident enough)
                confidence = face_obj.get("confidence", 0)
                if confidence < 0.8:  # Skip low confidence detections
                    continue
                
                # Get face image and coordinates
                face_img = face_obj["face"]
                facial_area = face_obj["facial_area"]
                
                # Extract embedding from the already detected and cropped face
                face_embedding = DeepFace.represent(
                    img_path=face_img,
                    model_name=MODEL_NAME,
                    detector_backend="skip",  # Skip detection as we already have the face
                    enforce_detection=False
                )
                
                if face_embedding and len(face_embedding) > 0:
                    # Get the embedding vector
                    embedding_vector = face_embedding[0]["embedding"]
                    
                    # Find the best matching person
                    best_match = "Unknown"
                    best_similarity = 0
                    
                    for person in known_faces_db.files:
                        # Get the stored average embedding for this person
                        known_embedding = known_faces_db[person]
                        
                        # Calculate similarity
                        similarity = calculate_cosine_similarity(embedding_vector, known_embedding)
                        
                        # Update best match if this is better
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = person
                    
                    # Determine if it's a known person or unknown based on threshold
                    label = best_match if best_similarity >= SIMILARITY_THRESHOLD else "Unknown"
                    color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
                    
                    # Draw bounding box
                    x = facial_area["x"]
                    y = facial_area["y"]
                    w = facial_area["w"]
                    h = facial_area["h"]
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Create label text with name and similarity score
                    label_text = f"{label}: {best_similarity:.2f}"
                    
                    # Put text for the label
                    cv2.putText(
                        display_frame, 
                        label_text, 
                        (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2
                    )
        
        except Exception as e:
            # Just log errors and continue with the next frame
            print(f"Error processing frame: {str(e)}")
        
        # Display FPS
        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        
        # Display the frame
        cv2.imshow("Real-time Face Recognition", display_frame)
        
        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Face recognition stopped.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepFace Real-time Face Recognition")
    parser.add_argument("--enroll", action="store_true", help="Run the enrollment process")
    parser.add_argument("--recognize", action="store_true", help="Run real-time recognition")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Dataset path for enrollment")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Database path")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not (args.enroll or args.recognize):
        parser.print_help()
        print("\nError: You must specify either --enroll or --recognize (or both).")
        exit(1)
    
    # Run the enrollment process if requested
    if args.enroll:
        enroll_faces(args.dataset, args.db)
        
    # Run real-time recognition if requested
    if args.recognize:
        if not os.path.exists(args.db):
            print(f"Error: Database file '{args.db}' not found. Please run enrollment first.")
            exit(1)
        run_recognition(args.db) 