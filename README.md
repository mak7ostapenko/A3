# A3: People counter

The main goal of the project is to count people on the streets. So all parameters are adjusted for the task.

### Quick Start

1. Clone repository.
2. Download converted weights of [yolo.h5 model file with tf-1.4.0](https://drive.google.com/file/d/1uvXFacPnrSMw6ldWTyLLjGLETlEsUvcE/view?usp=sharing)/ Put them into **model_data** folder.
3. Install requirements.
4. Specify path to input fileRun model with cmd :
   ```
   python demo.py --videofile="path/to/your/videofile/" --out_root_dir="path/to/outptu/dir/"
   ```

### Dependencies

  The code is compatible with Python 3. The following dependencies are needed to run the tracker:

    NumPy
    sklean
    OpenCV
    Pillow
    Keras

  Additionally, feature generation requires TensorFlow-1.4.0.

### Run for other classes

Be careful that the code ignores everything but person. Change class if you want run for other instance:
  
  [A3/yolo3/yolo.py]:
  
          if predicted_class != 'person': 
               continue 

### Notes for future work
  You can use any Detector you like to replace Keras_version YOLO to get bboxes , for it is to slow !
  
  Model file model_data/mars-small128.pb need by deep_sort had convert to tensorflow-1.4.0
 
**This work mainly based on   https://github.com/Qidian213/deep_sort_yolov3. Thanks a lot guy.**