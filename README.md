Human-Emotion App

Human emotions play a crucial role in communication, decision-making and human–computer interaction. With the rapid development of Artificial Intelligence and Computer Vision, machines are now capable of understanding and interpreting human emotions through facial expressions. Emotion detection systems have applications in mental health monitoring, education, customer feedback analysis, security systems and entertainment industries.

This project focuses on building a Human Emotion Detection System using deep learning techniques. The developed system is capable of:
-Classifying human facial expressions into seven emotion categories,
-Predicting emotions from both uploaded images and real-time webcam input,
-Providing a user-friendly web interface for easy interaction.

The system is trained on the FER-2013 dataset using a MobileNetV2 deep convolutional neural network, achieving high accuracy with efficient real-time performance.

Dataset Background
The project uses the FER-2013 dataset, which is a widely used benchmark dataset for facial emotion recognition. It consists of thousands of grayscale facial images (Image Size: 48×48 pixels) categorized into seven emotion classes:
-Angry
-Disgust
-Fear
-Happy
-Sad
-Surprise
-Neutral

System Architecture
The complete Human Emotion Detection system consists of four major components:
-Data Preprocessing Module
-Deep Learning Model (MobileNetV2)
-Flask Backend Server
-Web-Based User Interface
User uploads an image or captures one using a webcam. The image is preprocessed (resized, normalized). The trained CNN model predicts the emotion. The predicted emotion and confidence score are displayed on the web interface.

Data preprocessing is a critical step to improve model performance and stability. The following preprocessing techniques are used:
-Resizing all images to 96×96 pixels
-Conversion to RGB format
-Pixel value normalization to [0,1] range
-Data augmentation (Rotation, Zooming, Horizontal flipping)
These steps help the model generalize better on real-world images.

Model Architecture
The model used in this project is based on MobileNetV2, a lightweight and efficient convolutional neural network architecture optimized for real-time applications.
To improve task-specific learning while maintaining computational efficiency, a partial fine-tuning strategy is adopted. The earlier layers of MobileNetV2 are frozen, while the last 30 layers are unfrozen and retrained on the FER-2013 dataset. This allows the network to adapt high-level features specifically for facial emotion recognition.

-Pretrained MobileNetV2 base network (ImageNet weights)
-Partial fine-tuning of the last 30 layers
-Global Average Pooling Layer to reduce feature dimensionality
-Batch Normalization Layer for training stability
-Fully Connected Dense Layer (512 neurons, ReLU activation)
-Dropout Layer (0.5) for regularization
-Fully Connected Dense Layer (256 neurons, ReLU activation)
-Dropout Layer (0.4) for further regularization
-Final Dense Output Layer (7 neurons, Softmax activation) corresponding to the seven emotion classes.

The model is trained using the following setup:
-Optimizer: Adam
-Learning Rate: 0.0001
-Loss Function: Categorical Crossentropy
-Batch Size: 64
-Number of Epochs: 20
-Evaluation Metric: Accuracy
The training process shows a significant reduction in loss with increasing accuracy across epochs.

To enhance the training process and prevent overfitting, several training callbacks are used:
-EarlyStopping: Monitors validation loss and stops training if performance does not improve for 7 consecutive epochs while restoring the best weights.
-ReduceLROnPlateau: Automatically reduces the learning rate by a factor of 0.3 when validation loss plateaus.
-ModelCheckpoint: Saves the best-performing model based on validation loss to emotion_mobilenet_best.h5.
These strategies significantly improve convergence, model stability and generalization performance.

Web Application Development
The web application is developed using Flask, a lightweight Python web framework.
Web App Features
-Image upload-based emotion prediction
-Webcam capture inside the browser
-Predicted emotion display
-Confidence score visualization

Technology Stack
-Responsive UI using HTML and CSS
-Backend: Flask (Python)
-Frontend: HTML, CSS, JavaScript
-Deep Learning Framework: TensorFlow & Keras
-Image Processing: OpenCV

The system supports real-time emotion prediction through the user’s webcam, where each captured frame is processed by the Flask server for instant inference and display of the predicted emotion. This enables real-world testing and validates the model’s practical usability across multiple domains. The Human Emotion Detection App can be effectively applied in mental health monitoring systems, smart classrooms and educational platforms, customer sentiment analysis, human–robot interaction, gaming and entertainment, security and surveillance and social media analytics. This project successfully demonstrates the application of Deep Learning and Computer Vision in real-time human emotion detection using the FER-2013 dataset and a MobileNetV2-based CNN model, achieving high accuracy while remaining computationally efficient. The integration of a user-friendly web interface with real-time webcam support makes the system highly practical for real-world deployment. Overall, the project highlights the transformative power of AI in enhancing human–machine interaction and establishes a strong foundation for future intelligent and emotionally aware systems.

## How to Run
1. pip install -r requirements.txt
2. python train.py
3. python webcam_gui.py
4. python app.py (for both webcam and select pic app (index.html and style.css))
5. python pic.py (for only select pic app (index1.html and style1.css))


current accuracy: 57%