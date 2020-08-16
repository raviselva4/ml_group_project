## ML Facial Emotion Detection

Purpose of project:

  - To detect human emotion via face recognition and measure behavioural statistics using Machine Learning
 
Team member:

   - Ravi Selva
   - Ram Sake
   - Travis Le
   
Technology enabler: 

  - Tensorflow and Keras
  - Computer vision (CV2 new library)
  - Python Pandas
  - Python Matplotlib
  - HTML/CSS/Website
  - Amazon AWS (S3)
  
Supervised Classification Prediction Models:
 
  - face_emotion.h5 (facial emotion detector - used Convolutional Nerual Network (CNN))
  - res10_300x300_ssd_iter_140000.caffemodel (for picture augmentation)
  - mask_detector.model (for facial mask detection - used MobileNetV2)

Deployment

  - Heroku deployment is done successfully using flask 
  
  - https://facialemotiondetection.herokuapp.com/

Data Sources

  - FER2013 dataset from Kaggle.com (35,887 images)
  - 4836 images used for mask detection (google search)

