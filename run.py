from positive_example_generator import PositiveExampleGenerator
from negative_example_generator import NegativeExampleGenerator
from facial_detector import FacialDetector
from params import Params

import os

if __name__ == "__main__":
    parameters = Params()
  
    print("Generating positive examples...")
    PositiveExampleGenerator(parameters).generate_positive_examples()
    
    print("Generating negative examples...")
    NegativeExampleGenerator(parameters).generate_negative_examples()
    
    fd = FacialDetector(parameters)
    
    # fd.eval(parameters.test_dir)
    fd.eval_yolo(parameters.test_dir)