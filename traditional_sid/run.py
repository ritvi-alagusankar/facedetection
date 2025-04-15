import os
import sys
import argparse

# this script is just to simplify training or live real-time detection process into a single script

def print_banner():
    """Print a nice banner for the application"""
    print("\n" + "="*50)
    print("FACE RECOGNITION SYSTEM - TRADITIONAL METHODS")
    print("Using: Viola-Jones/HOG + Eigenfaces + SVM")
    print("="*50 + "\n")

def main():
    """Main function to parse arguments and run the appropriate script"""
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'webcam'],
                        help='Operation mode: train or webcam')
    
    args = parser.parse_args()
    print_banner()

    if args.mode == 'train':
        print("[INFO] Running in TRAINING mode")
        import face_recognition_system
        face_recognition_system.main()

    elif args.mode == 'webcam':
        print("[INFO] Running in WEBCAM RECOGNITION mode")
        if not os.path.exists("./output/face_recognition_model.pkl"):
            print("[ERROR] Face recognition model not found. Run training first.")
            return
        
        import realtime_recognition
        realtime_recognition.run_webcam_recognition()

if __name__ == "__main__":
    main()
