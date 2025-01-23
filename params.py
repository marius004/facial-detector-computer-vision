from dataclasses import dataclass, field
from typing import List

import os

@dataclass
class Params:
    base_training_dir: str = "antrenare"
    base_output_dir: str = "data"
    base_descriptors_dir: str = os.path.join("data", "descriptors")
    
    positive_examples_output_dir: str = os.path.join("data", "positive")
    negative_examples_output_dir: str = os.path.join("data", "negative")
    
    positive_descriptors_dir: str = os.path.join("data", "descriptors", "positive.npy")
    negative_descriptors_dir: str = os.path.join("data", "descriptors", "negative.npy")
    
    num_negative_examples_per_image: int = 10
    dim_window: int = 64
    dim_hog_cell: int = 8
    dim_descriptor_cell: int = 64
    
    # test & output directories
    test_dir: str = "validare/validare/"
    base_npy_files_save_dir: str = "output"
    
    threshold: float = 1.0
    stride: int = 16
    
    visualize: bool = False    
    
    characters: List[str] = field(default_factory=lambda: ["dad", "deedee", "dexter", "mom", "unknown"])

    base_classifier_path = os.path.join("data", "models")
    classifier_file_path = os.path.join("data", "models", "model.pkl")

    def __post_init__(self):
        os.makedirs(self.base_descriptors_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.positive_descriptors_dir), exist_ok=True)
        os.makedirs(os.path.dirname(self.negative_descriptors_dir), exist_ok=True)
        os.makedirs(self.positive_examples_output_dir, exist_ok=True)
        os.makedirs(self.negative_examples_output_dir, exist_ok=True)
        os.makedirs(self.base_classifier_path, exist_ok=True)
        os.makedirs(self.base_npy_files_save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_npy_files_save_dir, "task1"), exist_ok=True)
        os.makedirs(os.path.join(self.base_npy_files_save_dir, "task2"), exist_ok=True)
        