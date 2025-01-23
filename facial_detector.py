from utils import calculate_iou, calculate_sliding_window_sizes
from character_face_classifier import CharacterFaceClassifier
from multiprocessing import Process, Queue, cpu_count
from yolo_facial_detector import YoloFacialDetector
from sklearn.svm import LinearSVC, SVC
from skimage.feature import hog
from params import Params
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pickle
import random
import glob
import math
import time
import os

class FacialDetector:
    def __init__(self, params: Params):
        self.params = params
        self.best_model = None
        
        self.classifier = CharacterFaceClassifier(self.params)
        self.yolo_model = YoloFacialDetector(self.params)

    def get_descriptors(self, example_type: str):
        if example_type == 'positive':
            descriptors_path = self.params.positive_descriptors_dir 
            examples_dir = self.params.positive_examples_output_dir
        elif example_type == 'negative':
            descriptors_path = self.params.negative_descriptors_dir
            examples_dir = self.params.negative_examples_output_dir
        else:
            raise ValueError(f"Unknown example type: {example_type}")

        if os.path.exists(descriptors_path):
            descriptors = np.load(descriptors_path)
            print(f'Loaded {example_type} descriptors from {descriptors_path}')
        else:
            descriptors = []
            for character in self.params.characters:
                if example_type == 'negative' and character == "unknown":
                    continue

                dir_path = os.path.join(examples_dir, character)
                images_path = os.path.join(dir_path, '*.jpg')
                files = glob.glob(images_path)
                num_images = len(files)

                print(f'Processing {num_images} images from the {dir_path} directory...')

                for i, file_path in enumerate(files):
                    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                    features = hog(
                        img,
                        pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                        cells_per_block=(2, 2),
                        feature_vector=True
                    )
                    descriptors.append(features.flatten())

            descriptors = np.array(descriptors)
            np.save(descriptors_path, descriptors)
            print(f'Saved {example_type} descriptors to {descriptors_path}')

        return descriptors

    def get_positive_descriptors(self):
        return self.get_descriptors('positive')

    def get_negative_descriptors(self):
        return self.get_descriptors('negative')
    
    def train_classifier(self):
        if os.path.exists(self.params.classifier_file_path):
            self.best_model = pickle.load(open(self.params.classifier_file_path, 'rb'))
            return
        
        positive_descriptors = self.get_positive_descriptors()
        negative_descriptors = self.get_negative_descriptors()
        
        print("Positive Descriptors shape:", positive_descriptors.shape)
        print("Negative Descriptors shape:", negative_descriptors.shape)
        
        training_examples = np.concatenate((positive_descriptors, negative_descriptors), axis=0)
        train_labels = np.concatenate((np.ones(positive_descriptors.shape[0]), np.zeros(negative_descriptors.shape[0])))
        
        best_accuracy = 0
        best_c = 0
        best_model = None

        Cs = [10]        
        for c in Cs:
            print(f'Training a classifier for c={c}')
            model = SVC(C=c, kernel='rbf')
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print(f'Optimal classifier performance for c = {best_c}')
        pickle.dump(best_model, open(self.params.classifier_file_path, 'wb'))
        self.best_model = best_model
        
    def process_single_image(self, img_path: str):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            return np.array([]), np.array([]), np.array([]), np.array([])
                
        img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        detections = []
        scores = []

        window_sizes = calculate_sliding_window_sizes(img.shape[0], img.shape[1])
        
        for window_h, window_w in window_sizes:
            for y in range(0, img.shape[0] - window_h + 1, self.params.stride):
                for x in range(0, img.shape[1] - window_w + 1, self.params.stride):
                    patch = img[y:y + window_h, x:x + window_w]
                    patch_resized = cv.resize(patch, (self.params.dim_window, self.params.dim_window))
                    
                    descr = hog(
                        patch_resized,
                        pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                        cells_per_block=(2, 2),
                        feature_vector=True
                    ).flatten()
                    
                    score = self.best_model.decision_function([descr])[0]
                    if score > self.params.threshold:
                        detections.append([x, y, x + window_w, y + window_h])
                        scores.append(score)

        if len(detections) > 0:
            detections_array = np.array(detections)
            scores_array = np.array(scores)
            
            filtered_detections, filtered_scores = self.non_maximal_suppression(
                detections_array,
                scores_array,
                img.shape
            )

            characters = []
            character_scores = []
            for x_min, y_min, x_max, y_max in filtered_detections:
                patch = img[int(y_min):int(y_max), int(x_min):int(x_max)]
                patch_resized = cv.resize(patch, (self.params.dim_window, self.params.dim_window))
                detected_character, char_score = self.classifier.run(patch_resized)
                characters.append(detected_character.lower())
                character_scores.append(char_score)

            return filtered_detections, filtered_scores, characters, character_scores
        
        return np.array([]), np.array([]), np.array([]), np.array([])

    def process_image_batch(self, image_paths: list, queue: Queue):
        for img_path in image_paths:
            print(f"Processing {img_path}")
            detections, scores, characters, char_scores = self.process_single_image(img_path)
            
            queue.put({
                'image_path': img_path,
                'detections': [] if detections is None else detections,
                'scores': [] if scores is None else scores,
                'characters': [] if characters is None else characters,
                'character_scores': [] if char_scores is None else char_scores
            })
            
    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]

        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3

        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i]:
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j]:
                        if calculate_iou(sorted_image_detections[i], sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False

        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def eval(self, images_directory):
        self.train_classifier()
        
        num_processes= cpu_count()
        
        image_files = glob.glob(os.path.join(images_directory, '*.jpg'))
        num_images = len(image_files)
        
        batch_size = math.ceil(num_images / num_processes)
        
        processes = []
        queue = Queue()
        
        character_detections = {char: [] for char in ['dad', 'deedee', 'dexter', 'mom']}
        character_scores = {char: [] for char in ['dad', 'deedee', 'dexter', 'mom']}
        character_classifier_scores = {char: [] for char in ['dad', 'deedee', 'dexter', 'mom']}
        character_files = {char: [] for char in ['dad', 'deedee', 'dexter', 'mom']}
        
        for i in range(num_processes):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_images)
            
            if start_idx >= num_images:
                break
                
            batch_paths = image_files[start_idx:end_idx]
            
            p = Process(
                target=self.process_image_batch,
                args=(batch_paths, queue)
            )
            processes.append(p)
            p.start()
        
        processed_images = 0
        while processed_images < num_images:
            result = queue.get()
            processed_images += 1
            
            img_path = result['image_path']
            detections = result['detections']
            scores = result['scores']
            characters = result['characters']
            char_scores = result['character_scores']
            
            if len(detections) > 0:
                for det, score, char, char_score in zip(detections, scores, characters, char_scores):
                    if char in character_detections:
                        character_detections[char].append(det)
                        character_scores[char].append(score)
                        character_classifier_scores[char].append(char_score)
                        character_files[char].append(os.path.basename(img_path))
                
                if self.params.visualize:
                    img = cv.imread(img_path)
                    for det, score, char, char_score in zip(detections, scores, characters, char_scores):
                        x_min, y_min, x_max, y_max = map(int, det)
                        cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv.putText(img, f"{score:.2f} - ({char}) {char_score:.2f}", 
                                (x_min, y_min - 5),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    plt.title(f'Final Detections - {os.path.basename(img_path)}')
                    plt.axis('off')
                    plt.show()
                
            print(f"Processed image {processed_images}/{num_images}")
        
        for p in processes:
            p.join()
        
        # Save results for task1
        task1_base_directory = os.path.join(self.params.base_npy_files_save_dir, 'task1')
        
        all_detections = []
        all_scores = []
        all_files = []
        
        for char_dets, char_scores, char_class_scores, char_files in zip(
            character_detections.values(),
            character_scores.values(),
            character_classifier_scores.values(),
            character_files.values()
        ):
            all_detections.extend(char_dets)
            all_scores.extend(char_scores)
            all_files.extend(char_files)
        
        np.save(os.path.join(task1_base_directory, 'detections_all_faces.npy'), np.array(all_detections))
        np.save(os.path.join(task1_base_directory, 'scores_all_faces.npy'), np.array(all_scores))
        np.save(os.path.join(task1_base_directory, 'file_names_all_faces.npy'), np.array(all_files))
        
        # Save results for task2
        task2_base_directory = os.path.join(self.params.base_npy_files_save_dir, 'task2')
        
        for character in character_detections.keys():
            detections_array = np.array(character_detections[character]) if character_detections[character] else np.array([])
            scores_array = np.array(character_scores[character]) if character_scores[character] else np.array([])
            classifier_scores_array = np.array(character_classifier_scores[character]) if character_classifier_scores[character] else np.array([])
            files_array = np.array(character_files[character]) if character_files[character] else np.array([])
            
            np.save(os.path.join(task2_base_directory, f'detections_{character}.npy'), detections_array)
            np.save(os.path.join(task2_base_directory, f'scores_{character}.npy'), scores_array)
            np.save(os.path.join(task2_base_directory, f'scores_{character}.npy'), classifier_scores_array)
            np.save(os.path.join(task2_base_directory, f'file_names_{character}.npy'), files_array)
   
    def eval_yolo(self, images_directory):
        image_files = glob.glob(os.path.join(images_directory, '*.jpg'))
        num_images = len(image_files)
        
        character_detections = {char: [] for char in self.params.characters}
        character_scores = {char: [] for char in self.params.characters}
        character_files = {char: [] for char in self.params.characters}
        
        model = self.yolo_model.train()
        
        processed_images = 0
        for img_path in image_files:
            img = cv.imread(img_path)
            
            results = model(img)[0]
            processed_images += 1
            
            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidence_scores = results.boxes.conf.cpu().numpy()
                class_indices = results.boxes.cls.cpu().numpy()
                
                for box, conf_score, class_idx in zip(boxes, confidence_scores, class_indices):
                    x_min, y_min, x_max, y_max = map(int, box)
                    character = self.params.characters[int(class_idx)]
                    
                    if character in character_detections:
                        character_detections[character].append([x_min, y_min, x_max, y_max])
                        character_scores[character].append(conf_score)
                        character_files[character].append(os.path.basename(img_path))
                    
                    if self.params.visualize:
                        cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv.putText(img, f"{conf_score:.2f} - {character}", 
                                (x_min, y_min - 5),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                    
                if self.params.visualize:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    plt.title(f'YOLO Detections - {os.path.basename(img_path)}')
                    plt.axis('off')
                    plt.show()
            
            print(f"Processed image {processed_images}/{num_images}")
        
        # Save results for task1
        task1_base_directory = os.path.join(self.params.base_npy_files_save_dir, 'task1')
        os.makedirs(task1_base_directory, exist_ok=True)
        
        all_detections = []
        all_scores = []
        all_files = []
        
        for char_dets, char_scores, char_files in zip(
            character_detections.values(),
            character_scores.values(),
            character_files.values()
        ):
            all_detections.extend(char_dets)
            all_scores.extend(char_scores)
            all_files.extend(char_files)
        
        np.save(os.path.join(task1_base_directory, 'detections_all_faces.npy'), np.array(all_detections))
        np.save(os.path.join(task1_base_directory, 'scores_all_faces.npy'), np.array(all_scores))
        np.save(os.path.join(task1_base_directory, 'file_names_all_faces.npy'), np.array(all_files))
        
        # Save results for task2
        task2_base_directory = os.path.join(self.params.base_npy_files_save_dir, 'task2')
        os.makedirs(task2_base_directory, exist_ok=True)
        
        for character in character_detections.keys():
            detections_array = np.array(character_detections[character]) if character_detections[character] else np.array([])
            scores_array = np.array(character_scores[character]) if character_scores[character] else np.array([])
            files_array = np.array(character_files[character]) if character_files[character] else np.array([])
            
            np.save(os.path.join(task2_base_directory, f'detections_{character}.npy'), detections_array)
            np.save(os.path.join(task2_base_directory, f'scores_{character}.npy'), scores_array)
            np.save(os.path.join(task2_base_directory, f'file_names_{character}.npy'), files_array)