import cv2
from ultralytics import YOLO 
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
from skimage.segmentation import mark_boundaries
from lime import lime_image

img = cv2.imread("./data/test/car_road.jpg", cv2.IMREAD_COLOR) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
img = cv2.resize(img, (640, 640)) 

model = YOLO("yolov8m.pt") 

prediction = model.predict(img) 
class_names = prediction[0].names 
print(class_names)
results = prediction[0].boxes.data

for pred in prediction: 
    im_array = pred.plot() 
    im = Image.fromarray(im_array[..., ::-1])
    im.show() 

yolo_results = []
confs = [] 
clss = [] 

for result in results: 
    temp = result.tolist() 
    x1 = temp[0]
    y1 = temp[1]
    x2 = temp[2] 
    y2 = temp[3] 
    confidence = temp[4] 
    confs.append(confidence)
    class_value = temp[5] 
    clss.append(class_value)
    output = [x1, y1, x2, y2, confidence, class_value] 
    yolo_results.append(output)
print(yolo_results) 

def generate_lime_explanation(image_path, yolo_results):
    # Load the image
    image = cv2.imread("./data/test/car_road.jpg", cv2.IMREAD_COLOR) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image_resized = cv2.resize(image, (640, 640))

    # Create a LIME explainer 
    explainer = lime_image.LimeImageExplainer()

    # Define the predict function 
    def predict_function(images):
        with torch.no_grad(): 
            output = model(img)
            data = output[0].boxes

        conf_array = data.conf.numpy() 
        class_indices = data.cls.numpy() 

        class_probabilities = np.zeros((len(images), len(class_names)))
        for i in range(len(images)):
            for j in range(len(class_names)): 
                if j in class_indices: 
                    idx = np.where(class_indices==j)[0][0]
                    class_probabilities[i, j] = conf_array[idx] 
        return class_probabilities
    
    #class_labels = [str(class_val) for class_val in clss] 

    # Generate LIME explanation
    #explanation = explainer.explain_instance(image_resized, predict_function, labels=class_labels,top_labels=len(class_labels))
    explanation = explainer.explain_instance(image_resized, predict_function, top_labels=5, hide_color=0)

    # Display the LIME explanation
    #temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    temp, mask = explanation.get_image_and_mask(clss[1], positive_only=False, num_features=10, hide_rest=False)
    segmented_image = mark_boundaries(temp / 2 + 0.5, mask)
    
    plt.imshow(segmented_image)
    plt.show()

image_path = "./data/test/car_road.jpg" 
generate_lime_explanation(image_path, yolo_results)