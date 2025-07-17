# RetinaNet

**RetinaNet** is a single-stage object detector that uses Focal Loss and two task-specific sub-networks for object detection.

- To tackle class imbalance, **Focal Loss** modifies the standard Cross Entropy Loss function to down-weight the loss assigned to well-classified examples. It gives less importance to easily classified objects and focuses more on hard, misclassified examples. This allows for increased accuracy without sacrificing speed.

- The backbone of the model for feature extraction is a **Feature Pyramid Network**. This network divides an input into multiple channels (usually 3 for RGB) and passes each channel into the subsequent networks, both in training and inference.

- Its feature pyramid network is build on a **hot-swappable CNN**. This means that any CNN that is trained on classification can be used (AlexNet, VGG,Inception, etc.). For this model, we utilized ResNet with various depths (ResNet50 until ResNet152).

- The **classification subnet** takes the FPN feature map, passes it to ReLU activation layers, before a final convolutional layer to output the predictions for each anchor.

- The **Box Regression Subnet** predicts precise bounding box offset for each anchor box, running parallel with the classification subnet, also attached to each level of the FPN. It uses a standard L1 regularization loss.


## Pipeline Documentation

The `RetinaNet_v1.ipynb` is a jupyter notebook serving as the documentation of the training and inference pipeline. Running the model itself is through executing the scripts, which will also be explained at the end of this markdown.

The notebook includes:
1. Dataset sources
2. Data Preparation
3. Model Development
4. Training Pipeline
5. Evaluation Metrics
6. Inference Pipeline
7. Loading Pretrained Weights

Use this notebook as a guide to integrate the pipeline in similar models, which is under the umbrella of **single-stage object detectors** which include YOLO, Faster R-CNN, and its variations.


## Validation


## Loading Pretrained Weights 


## Inference


## Grad-CAM Visualization

---
##### Written By:

Toni "Sniper" Yenisei Czar S. Castañares