# models/yolov5.py
import torch

class YOLOv5:
    def __init__(self, model_path):
        # Load the model using Ultralytics YOLOv5 repository's torch hub.
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.eval()

    def __call__(self, image):
        # Perform inference
        return self.model(image)

    def fuse(self):
        # Fuse the model for faster inference (optimizes Conv2d layers)
        self.model.fuse()
