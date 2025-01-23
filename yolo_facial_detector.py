from ultralytics import YOLO
from params import Params
from pathlib import Path

import cv2 as cv
import shutil
import yaml
import glob
import os

class YoloFacialDetector:
    def __init__(self, params: Params):
        self.params = params
        self.characters = self.params.characters
        self.data_dir = os.path.join(self.params.base_output_dir, "yolo")
        self.model_path = os.path.join("runs/detect/dexter_laboratory_face_detection/weights/best.pt")
        
        self.label_to_class = {character: i for i, character in enumerate(self.params.characters)}
        
        os.makedirs(self.data_dir, exist_ok=True)
        self.setup_directories()
        
    def setup_directories(self):
        for split in ['train', 'val']:
            Path(f"{self.data_dir}/{split}/images").mkdir(parents=True, exist_ok=True)
            Path(f"{self.data_dir}/{split}/labels").mkdir(parents=True, exist_ok=True)
    
    def create_data_yaml(self):
        data_yaml = {
            'path': os.path.abspath(self.data_dir),
            'train': 'train/images',
            'val': 'val/images',
            'names': {i: char for i, char in enumerate(self.characters)},
            'nc': len(self.characters)
        }
        with open(f"{self.data_dir}/data.yaml", 'w') as f:
            yaml.dump(data_yaml, f)
    
    def prepare_dataset(self):
        for idx, character in enumerate(self.characters):
            if character == "unknown":
                continue
            
            pos_dir = os.path.join(self.params.base_training_dir, character)
            images = glob.glob(os.path.join(pos_dir, "*.jpg"))
            
            annotation_path = os.path.join(self.params.base_training_dir, f"{character}_annotations.txt")
            annotations = {}
            
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        img_name = parts[0]
                        bbox = [float(x) for x in parts[1:5]]
                        label = parts[5]
                        
                        if img_name not in annotations:
                            annotations[img_name] = [(bbox, label)]
                        else: 
                            annotations[img_name].append((bbox, label))

            split_idx = int(len(images) * 0.8)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            for img_path in train_images:
                img_name = os.path.basename(img_path)
                shutil.copy(img_path, f"{self.data_dir}/train/images/{character}_{img_name}")
                
                for bbox, label in annotations[img_name]:
                    self._create_yolo_label(
                        f"{self.data_dir}/train/labels/{character}_{img_name.replace('.jpg', '.txt')}",
                        self.label_to_class[label],
                        bbox, 
                        img_path
                    )
            
            for img_path in val_images:
                img_name = os.path.basename(img_path)
                shutil.copy(img_path, f"{self.data_dir}/val/images/{character}_{img_name}")
                
                for bbox, label in annotations[img_name]:
                    self._create_yolo_label(
                        f"{self.data_dir}/val/labels/{character}_{img_name.replace('.jpg', '.txt')}",
                        self.label_to_class[label],
                        bbox, 
                        img_path
                    )

    def _create_yolo_label(self, label_path, class_idx, bbox, img_path):
        with open(label_path, 'a') as f:
            img = cv.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                return
            
            height, width = img.shape[:2]
            
            x1, y1, x2, y2 = bbox
            
            x1 = x1 / width
            x2 = x2 / width
            y1 = y1 / height
            y2 = y2 / height
            
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # <class> <x_center> <y_center> <width> <height>
            f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
    
    def train(self, epochs=200):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            model = YOLO(self.model_path)
            return model
        
        print("Training new YOLO model...")
        self.create_data_yaml()
        self.prepare_dataset()
        
        model = YOLO('yolov8n.pt')
        
        results = model.train(
            data=f"{self.data_dir}/data.yaml",
            epochs=epochs,
            imgsz=64,
            batch=16,
            name='dexter_laboratory_face_detection',
        )
        
        print(f"Model saved to {self.model_path}")
        return model