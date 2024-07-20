from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import sys
import pickle

# dlib 경로 추가
sys.path.append('C:\Users\82105\Desktop\face_detection')
import dlib

from facenet_pytorch import InceptionResnetV1
import torch
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 설정 변수
UPLOAD_FOLDER = './static/uploads'
FEATURE_DATASET_PATH = 'feature_dataset.pkl'
MODEL_PATH = './dlib/shape_predictor_68_face_landmarks.dat'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dlib 얼굴 탐지기와 랜드마크 모델 로드
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(MODEL_PATH)
model = InceptionResnetV1(pretrained='vggface2').eval()

# 저장된 특징 벡터 파일 로드
with open(FEATURE_DATASET_PATH, 'rb') as f:
    feature_dataset = pickle.load(f)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 누적합 히스토그램 평활화 적용
    channels = cv2.split(image)
    eq_channels = []
    for ch in channels:
        eq_ch = cv2.equalizeHist(ch)
        eq_channels.append(eq_ch)
    image = cv2.merge(eq_channels)
    
    return image

def extract_features(image):
    faces = detector(image)
    if len(faces) == 0:
        return None
    face = faces[0]
    shape = sp(image, face)
    aligned_face = dlib.get_face_chip(image, shape)
    aligned_face = torch.tensor(aligned_face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        feature_vector = model(aligned_face)
    return feature_vector.squeeze().numpy()

def find_most_similar_image(target_features, feature_dataset):
    max_similarity = -1
    most_similar_image_path = None
    for image_path, features in feature_dataset.items():
        similarity = cosine_similarity([target_features], [features])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_image_path = image_path
    return most_similar_image_path, max_similarity

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 전송되지 않았습니다.'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'})
    if file and file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        image = preprocess_image(filename)
        target_features = extract_features(image)
        if target_features is None:
            return jsonify({'error': '얼굴을 찾을 수 없습니다.'})
        most_similar_image_path, similarity = find_most_similar_image(target_features, feature_dataset)
        tag_name = os.path.basename(os.path.dirname(most_similar_image_path))
        return jsonify({'tag': tag_name, 'image': most_similar_image_path, 'similarity': float(similarity)})
    else:
        return jsonify({'error': '허용되지 않는 파일 형식입니다.'})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)