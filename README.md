# American Sign Language detection
A Deep Learning Computer Vision Project

This project detects American Sign Language (ASL) alphabet gestures from images using a Convolutional Neural Network (CNN).
It demonstrates real-world image classification, preprocessing, model training, evaluation, and prediction.

 Key Features :-
   
ASL alphabet gesture recognition (A–Z, excluding ‘J’ & ‘Z’ if dataset uses static gestures)
CNN-based image classification
Image preprocessing pipeline
Model training, validation, and testing
Confusion matrix and accuracy evaluation
Real-time prediction support (if extended with webcam/OpenCV)
Jupyter Notebook for step-by-step analysis

How the Model Works
1️⃣ Data Preprocessing

Images are resized (typically 64×64 or 128×128)
Converted to grayscale or RGB depending on dataset
Normalized to range [0,1]
Augmentation applied (if enabled):
rotation
zoom
flip
brightness shift

2️⃣ Model Architecture (Typical CNN)
The model is built using Keras/TensorFlow with layers like:

Conv2D
MaxPooling2D
BatchNormalization
Dropout
Flatten
Dense (Output softmax layer)

🎯 Training

The notebook trains the CNN using:
Loss: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy
EarlyStopping & ModelCheckpoint to prevent overfitting
During training, the notebook visualizes:

Training vs validation accuracy

Training vs validation loss
