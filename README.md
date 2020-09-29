# ColorTrack
Human tracking based on color of clothes

ColorTrack is a module that can track humans based on the color of their upper body garment (e.g. shirt). It is developed in python3.6. ColorTrack uses [detectron2](https://github.com/facebookresearch/detectron2) for human detection on a set of image data, kmeans clustering for shirt color extraction and uses a greedy algorithm for associating the detected humans of two consequent image frames.

# Requirements

When the module was developed, the following where used:

python 3.6  
opencv 4.2.0  
cuda 9.2-10.2 with compatible torch, torchvision (find the appropriate combinations [here](https://pytorch.org))  
numpy 1.19.2  
sklearn 0.23.2  
matplotlib 3.3.2  
detectron2 pre-built (depending on the cuda and torch versions, for pre-built installation follow [these](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) instructions)  
