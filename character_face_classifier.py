from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

import numpy as np
import cv2 as cv
import os
import glob
import pickle

class CharacterDataset:
    def __init__(self, params):
        self.params = params
        self.images = []
        self.labels = []
        self.char_to_idx = {
            'dexter': 0,
            'mom': 1, 
            'dad': 2,
            'deedee': 3,
            'unknown': 4,
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self._load_data()
    
    def _load_data(self):
        for character in self.params.characters:
            char_dir = os.path.join(self.params.positive_examples_output_dir, character)
            image_files = glob.glob(os.path.join(char_dir, '*.jpg'))
            
            for file_path in image_files:
                img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                img_flattened = img.flatten()
                self.images.append(img_flattened)
                self.labels.append(self.char_to_idx[character])
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

class CharacterFaceClassifier:
    def __init__(self, params):
        self.params = params
        self.model = None
        self.dataset = CharacterDataset(params)
        self.train_classifier()
    
    def train_classifier(self):
        model_path = os.path.join(self.params.base_classifier_path, 'svm_model.pkl')
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            return
        
        model = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=False,
            class_weight='balanced',
            random_state=42
        )
        
        model.fit(self.dataset.images, self.dataset.labels)
        self.model = model
        
        os.makedirs(self.params.base_classifier_path, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def run(self, img_resized):
        if len(img_resized.shape) == 3:
            img_resized = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
        
        img_flattened = img_resized.flatten()
        
        scores = self.model.decision_function([img_flattened])
        
        predicted_idx = np.argmax(scores)
        predicted_character = self.dataset.idx_to_char[predicted_idx]
        
        return predicted_character, scores[0][predicted_idx]
