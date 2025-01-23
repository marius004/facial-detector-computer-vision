# Dexter's Laboratory Facial Detector

## Documentation

Here you can find the [documentation](./DOCUMENTATION.md)

## Requirements

Here you can find the
[requirements](./requirements.txt) or run the following script in your venv:

```bash
pip install -r requirements.txt
```

## Running the facial detector
1. Configure the test and output directories in [params.py](./params.py):
```py
test_dir: str = "..."
base_npy_files_save_dir: str = "..."
```

2. Run the detector:
```
python3 run.py
```

To run the YOLO detector you need to comment out 
```py
fd.eval(parameters.test_dir)
```
and uncomment 
```py 
fd.eval_yolo(parameters.test_dir)
```
as both methods use the same output directory.