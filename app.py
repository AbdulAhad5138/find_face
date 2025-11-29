import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import shutil
import uuid

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MATCHES_FOLDER = 'static/matches'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MATCHES_FOLDER, exist_ok=True)

# Load Models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# YOLO Model (User's model)
try:
    print("Loading YOLO model...")
    yolo_model = YOLO('best1.pt')
    print("YOLO model loaded.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

# Face Recognition Models
print("Loading FaceNet models...")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device) # Fallback detector
print("FaceNet models loaded.")

def get_embedding(img_rgb, box=None):
    """
    Get embedding for a face.
    img_rgb: PIL Image or numpy array (RGB)
    box: (x1, y1, x2, y2) optional
    """
    if isinstance(img_rgb, np.ndarray):
        img_pil = Image.fromarray(img_rgb)
    else:
        img_pil = img_rgb

    # If box is provided, crop. Else, detect.
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        # Ensure bounds
        w, h = img_pil.size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        face = img_pil.crop((x1, y1, x2, y2))
    else:
        # If no box, try to find one with MTCNN
        boxes, _ = mtcnn.detect(img_pil)
        if boxes is None or len(boxes) == 0:
            return None
        # Take largest face
        box = boxes[0]
        return get_embedding(img_rgb, box)

    # Resize to 160x160 for InceptionResnetV1
    face = face.resize((160, 160))
    
    # Convert to tensor and normalize
    face_tensor = np.float32(face)
    face_tensor = (face_tensor - 127.5) / 128.0
    face_tensor = torch.tensor(face_tensor).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = resnet(face_tensor).detach().cpu()
        
    return embedding

def detect_faces(img_rgb):
    """
    Detect faces using YOLO, fallback to MTCNN.
    Returns list of boxes (x1, y1, x2, y2).
    """
    boxes = []
    
    # Try YOLO
    if yolo_model:
        results = yolo_model(img_rgb, verbose=False)
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                boxes.append(box.xyxy[0].tolist())
    
    # If YOLO found nothing, try MTCNN
    if not boxes:
        mtcnn_boxes, _ = mtcnn.detect(Image.fromarray(img_rgb))
        if mtcnn_boxes is not None:
            boxes = mtcnn_boxes.tolist()
            
    return boxes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    if 'target' not in request.files or 'folder_images' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    target_file = request.files['target']
    folder_files = request.files.getlist('folder_images')

    # 1. Process Target Image
    target_bytes = target_file.read()
    nparr = np.frombuffer(target_bytes, np.uint8)
    target_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if target_img is None:
        return jsonify({'error': 'Invalid target image'}), 400
    
    target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    
    # Detect face in target (expecting 1 main face)
    target_boxes = detect_faces(target_rgb)
    if not target_boxes:
        return jsonify({'error': 'No face detected in target image'}), 400
    
    # Use the largest face if multiple
    # Sort by area
    target_boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    target_box = target_boxes[0]
    
    target_embedding = get_embedding(target_rgb, target_box)
    if target_embedding is None:
        return jsonify({'error': 'Could not process target face'}), 400

    matches = []
    
    # 2. Process Folder Images
    for file in folder_files:
        if not file.filename:
            continue
            
        try:
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect all faces
            faces_boxes = detect_faces(img_rgb)
            
            match_found = False
            for box in faces_boxes:
                embedding = get_embedding(img_rgb, box)
                if embedding is None:
                    continue
                
                # Calculate distance
                dist = (target_embedding - embedding).norm().item()
                
                # Threshold: usually 0.6 to 1.0 depending on normalization
                # InceptionResnetV1 pretrained on vggface2 usually works well with ~0.6-0.8
                if dist < 0.8: 
                    match_found = True
                    # Draw box
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    break # Found the person, move to next image
            
            if match_found:
                filename = f"match_{uuid.uuid4().hex}.jpg"
                save_path = os.path.join(MATCHES_FOLDER, filename)
                cv2.imwrite(save_path, img)
                
                matches.append({
                    'filename': file.filename,
                    'url': f"/{MATCHES_FOLDER}/{filename}"
                })
                
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            continue

    return jsonify({'matches': matches})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
