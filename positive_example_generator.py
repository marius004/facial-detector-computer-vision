from collections import defaultdict
from params import Params

import matplotlib.pyplot as plt
import cv2 as cv
import random
import os

class PositiveExampleGenerator:
    def __init__(self, params: Params):
        self.params = params
        self.unknown_image_counter = 0

    def process_annotation_file(self, annotation_file):
        character = annotation_file.split('_')[0]
        annotation_path = os.path.join(self.params.base_training_dir, annotation_file)
        
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
            
        img_counter = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            
            image_name, x1, y1, x2, y2, label = parts
            image_path = os.path.join(self.params.base_training_dir, character, image_name)
            
            if not os.path.exists(image_path):
                print(f"Image {image_path} does not exist!")
                continue
            
            image = cv.imread(image_path)
            
            height, width = image.shape[:2]
            max_pad_w = int(width * 0.10)
            max_pad_h = int(height * 0.10)
                
            pad_left = random.randint(0, max_pad_w)
            pad_right = random.randint(0, max_pad_w)
            pad_top = random.randint(0, max_pad_h)
            pad_bottom = random.randint(0, max_pad_h)
            
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            x1 = max(0, x1 - pad_left)
            x2 = min(width, x2 + pad_right)
            y1 = max(0, y1 - pad_top)
            y2 = min(height, y2 + pad_bottom)

            cropped = image[y1:y2, x1:x2]
            resized = cv.resize(cropped, (self.params.dim_window, self.params.dim_window))
            
            grayscale = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
            
            transformations = [
                grayscale,
            ]
            
            for transformed_image in transformations:
                if label == "unknown":
                    output_path = os.path.join(self.params.positive_examples_output_dir, "unknown", f"{self.unknown_image_counter}.jpg")
                    self.unknown_image_counter += 1
                else:
                    output_path = os.path.join(self.params.positive_examples_output_dir, label, f"{img_counter}.jpg")
                    img_counter += 1
                    
                cv.imwrite(output_path, transformed_image)

    def generate_positive_examples(self):
        if os.listdir(self.params.positive_examples_output_dir):
            print("Positive examples already exist. Skipping generation.")
            return
            
        for character in self.params.characters:
            os.makedirs(os.path.join(self.params.positive_examples_output_dir, character), exist_ok=True)
            
        annotation_files = [f for f in os.listdir(self.params.base_training_dir) if f.endswith('_annotations.txt')]
        for annotation_file in annotation_files:
            self.process_annotation_file(annotation_file)