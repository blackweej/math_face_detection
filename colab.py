import cv2
import numpy as np
import os
import sys
import pickle

# dlib 경로 추가
sys.path.append(r'C:\Users\82105\Desktop\face_detection\dlib')
import dlib

from facenet_pytorch import InceptionResnetV1
import torch

# 설정 변수
IMAGE_DIR = r'C:\Users\82105\Desktop\face_detection\images'
FEATURE_DATASET_NAME = r'C:\Users\82105\Desktop\face_detection\feature_dataset.pkl'
MODEL_PATH = r'./dlib/shape_predictor_68_face_landmarks.dat'

# Dlib 얼굴 탐지기와 랜드마크 모델 로드
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(MODEL_PATH)
model = InceptionResnetV1(pretrained='vggface2').eval()

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

def build_feature_dataset(image_dir, save_path):
    feature_dataset = {}
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root, file)
                image = preprocess_image(image_path)
                features = extract_features(image)
                if features is not None:
                    feature_dataset[image_path] = features
    with open(save_path, 'wb') as f:
        pickle.dump(feature_dataset, f)

# 특징 데이터셋 생성
build_feature_dataset(IMAGE_DIR, FEATURE_DATASET_NAME)