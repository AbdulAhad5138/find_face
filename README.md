# FaceMatch AI

A premium web application for face detection and recognition.

## Features
- **Custom Model Support**: Uses your trained `best1.pt` YOLO model for detection (if available).
- **Advanced Recognition**: Uses FaceNet (InceptionResnetV1) for accurate face matching.
- **Batch Processing**: Upload a folder of images to find a specific person.
- **Premium UI**: Modern, dark-themed interface with drag-and-drop support.

## Setup
1. Ensure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: `facenet-pytorch` will install `torch` and `torchvision`)

## Usage
1. Run the application:
   ```bash
   python app.py
   ```
2. Open your browser to `http://localhost:5000`.
3. Upload a **Target Image** (the face you want to find).
4. Upload a **Search Directory** (the folder containing images to search).
5. Click **Start Scanning**.

## Notes
- The app uses `best1.pt` for detection if present. If not, it falls back to MTCNN.
- Matching threshold is set to 0.8. Lower this value in `app.py` for stricter matching.
