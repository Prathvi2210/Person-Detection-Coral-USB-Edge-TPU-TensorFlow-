This is a person detection project based on TensorFLow framework. 
Person detection models are built on frameworks. For ex: Yolo is built in pytorch hence the .pt file.
Similarly SSD_MobileNet and EfficientDet are built on TensorFlow framework giving .tflite files.
These frameworks are translatable with the help of a universal framework such as ONXX. However some speed is lost in translation.

TPU-Tensor Processing Unit is a computing device made by google for fast deplyoment of Machine Learning models built on the tensorflow framework.
It uses very low power and is specifically optimized for the models it runs giving it the advantage.
Coral USB accelerator is a EDGE TPU by Google capable of 4TOPS max and 2TOPS operation on 1.5W power.
It can be used via USB connection (not bydefault in TPUs) with a CPU such as RPi.

Not all tensorflow models (.tflite files) can run on EDGE TPUs. They have to be compiled to run on EDGE TPU.

TensorFlow models specially EDGE TPU compiled are made for running on quantized formats i.e int8. The dev environment for coral is based on python 3.9. Current hardware only runs new OS which are based on 3.11 versions. 
In my case I am running RPi OS because it is more suitable for coral environment than ubuntu. In that Debian based 'TRIXIE' version has kernels unsupported for coral TPU so 'BOOKWORM' version is better suited.
In that there are 3 options: 
    A) Conversion pipeline: YOLO models based on CNNs are the most common and accurate for object detection but the framework is different. To run on coral we need to convert the .pt yolo files pytorch --> ONXX --> Tensorflow Lite --> EDGE TPU compiled.
       This conversion will lose accuracy as well as fps.
    B) Docker environment: A container for python 3.9. We need extra effort for the hardware interfacing in dockers, like the camera connected via USB.
    C) Virtual Environment: A python 3.9 environment built in RPi OS BookWorm named coral.
       >>>source coral/bin/activate.
       Then navigate to your code and run it.

Here I am trying to run the inbuilt models provided by google on the coral documentation page under object detection. These models are pre compiled to be run on EDGE TPU.
SSD_MobileNet_V2_coco_quant_postprocess_edgetpu.tflite is the basic model with 90 coco classes.
tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite model is little better in terms of accuracy.
NMS= Non Maximum supression: a post-processing technique used in machine learning, particularly in computer vision and object detection tasks, to eliminate redundant and overlapping bounding box detections for the same object
no_nms models dont have nms inbuilt i.e no post process here. you have to do it manually which gives a option of modifying the post process. It will need more code and run slowly, but it makes the model not frozen unlike those where post processing is inbuilt.
These SSD based models achieved 60-70 fps on my hardware.
SSDLite MobileDet was also tested giving 50 fps. Although docs give slightly more mAP for mobileDet model than EfficientDet-Lite0; it also has false positive problems.

A filter was applied to only detect person class to make prototype cleaner. No improvement in accuracy and performance yet.
EfficientDet-Lite3 gave 6-6.5 fps
EfficientDet-Lite2 gave 7 fps
EfficientDet-Lite1 gave 10-12 fps
EfficientDet-Lite0 gave 15-16 fps

Patch option: Edit the tensorflow model so it only processes the person class thus improving the fps.
Goals:
  Remove 79 COCO classes internally
  Shrink the classification head from 80 → 1 class
  Remove 79 class logits from the TFLite model
  Reduce NMS load
  Recompile for Edge TPU
  Get 20–25 FPS instead of 15–16 FPS
  Improve accuracy on person detection
  Reduce false positives
