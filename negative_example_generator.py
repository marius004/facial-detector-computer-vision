from utils import calculate_iou, calculate_sliding_window_sizes
from typing import Callable, List, Tuple
from params import Params

import numpy as np
import cv2 as cv
import random
import os

class NegativeExampleGenerator:
    def __init__(self, params: Params, stride: int = 32, retries: int = 10):
        self.retries = retries
        self.stride = stride
        self.params = params
        
    def get_negative_region_with_iou(self, image, bboxes, window_size):
        height, width = image.shape[:2]
        win_h, win_w = window_size

        x_positions = list(range(0, width - win_w + 1, self.stride))
        y_positions = list(range(0, height - win_h + 1, self.stride))
        
        random.shuffle(x_positions)
        random.shuffle(y_positions)

        for y in y_positions:
            for x in x_positions:
                current_box = (x, y, x + win_w, y + win_h)
                max_iou = 0

                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    iou = calculate_iou(current_box, (x1, y1, x2, y2))
                    max_iou = max(max_iou, iou)

                if max_iou < 0.3:
                    return x, y, x + win_w, y + win_h

        return None
    
    def process_annotation_file(self, annotation_file):
        character = annotation_file.split('_')[0]
        annotation_path = os.path.join(self.params.base_training_dir, annotation_file)
        
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
        
        image_annotations = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
                
            image_name, x1, y1, x2, y2, _ = parts
            if image_name not in image_annotations:
                image_annotations[image_name] = []
            
            image_annotations[image_name].append(list(map(int, [x1, y1, x2, y2])))
        
        img_counter = 0
        for image_name, bboxes in image_annotations.items():
            image_path = os.path.join(self.params.base_training_dir, character, image_name)
            
            if not os.path.exists(image_path):
                print(f"Image {image_path} does not exist!")
                continue
            
            image = cv.imread(image_path)
            if image is None:
                continue

            height, width = image.shape[:2]
            examples_found = 0
            
            for _ in range(self.retries):
                window_sizes = calculate_sliding_window_sizes(height, width)
                random.shuffle(window_sizes)
                
                for window_size in window_sizes:
                    if examples_found >= self.params.num_negative_examples_per_image:
                        break
                    
                    region = self.get_negative_region_with_iou(image, bboxes, window_size)
                    if region is None:
                        continue
                    
                    xmin, ymin, xmax, ymax = region
                    negative = image[ymin:ymax, xmin:xmax]
                    
                    negative = cv.cvtColor(negative, cv.COLOR_BGR2GRAY)
                    negative = cv.resize(negative, (self.params.dim_window, self.params.dim_window))
                                    
                    output_path = os.path.join(self.params.negative_examples_output_dir, character, f"{img_counter}.jpg")
                    cv.imwrite(output_path, negative)
                    
                    examples_found += 1
                    img_counter += 1

                if examples_found >= self.params.num_negative_examples_per_image:
                    break

    def generate_negative_examples(self):
        if os.listdir(self.params.negative_examples_output_dir):
            print("Negative examples already exist. Skipping generation.")
            return
        
        for character in self.params.characters:
            os.makedirs(os.path.join(self.params.negative_examples_output_dir, character), exist_ok=True)
        
        annotation_files = [f for f in os.listdir(self.params.base_training_dir) if f.endswith('_annotations.txt')]
        for annotation_file in annotation_files:
            self.process_annotation_file(annotation_file)
