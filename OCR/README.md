This project implements classical computer vision techniques for object detection and description in images.
Main features:

Object detection and segmentation: Uses OpenCV morphological operations to find and isolate regions of interest in images.

Feature extraction: Applies HOG (Histogram of Oriented Gradients) and LBP (Local Binary Pattern) descriptors with scikit-image for robust object representation.

Object comparison: Object descriptors are matched with reference samples via Euclidean distance; detected objects are highlighted on output images.

Batch processing: Supports automated processing of multiple images at once, without deep learning or neural networks.

To run the script, make sure you have the following dependencies installed:

numpy
matplotlib
opencv-python
scikit-image

Place your images in the appropriate folder and update the script paths as needed. Results are saved with bounding boxes for matched objects.
