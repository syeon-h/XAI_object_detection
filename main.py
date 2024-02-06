import cv2
from ultralytics import YOLO 
import numpy as np 
import torch

class YOLOv8SAHP(torch.nn.Module): 

    def __init__(self, model:torch.nn.Module, device: torch.device=torch.device('cpu')):  
        super(YOLOv8SAHP, self).__init__() 
        self.model = model 
        self.device = device 
        self.prediction = None 

    def cpu(self): 
        self.device = torch.device("cpu", 0) 
        self.inverter.device = self.device 
        return super(YOLOv8SAHP, self).cpu() 
    
    img = cv2.imread("./data/test/car_road.jpg", cv2.IMREAD_COLOR) 
    img = cv2.resize(img, (160, 160))  
    #img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) 
    #model = YOLO("yolov8n.yaml").load("yolov8n.pt") 
    model = torch.load("../yolov8n.pt", map_location='cpu')['model'].float().eval()  
    prediction = model(img) 
    