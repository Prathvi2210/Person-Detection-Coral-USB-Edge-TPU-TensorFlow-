This is a person detection project based on TensorFLow framework. 
Person detection models are built on frameworks. For ex: Yolo is built in pytorch hence the .pt file.
Similarly SSD_MobileNet and EfficientDet are built on TensorFlow framework giving .tflite files.
These frameworks are translatable with the help of a universal framework such as ONXX. However some speed is lost in translation.

TPU-Tensor Processing Unit is a computing device made by google for fast deplyoment of Machine Learning models built on the tensorflow framework.
It uses very low power and is specifically optimized for the models it runs giving it the advantage.
Coral USB accelerator is a EDGE TPU by Google capable of 4TOPS max and 2TOPS operation on 1.5W power.
It can be used via USB connection (not bydefault in TPUs) with a CPU such as RPi.

Not all tensorflow models (.tflite files) can run on EDGE TPUs. They have to be compiled to run on EDGE TPU.

Here I am trying to run the inbuilt models provided by google on the coral documentation page under object detection. These models are pre compiled to be run on EDGE TPU.
SSD_MobileNet_V2_coco_quant_postprocess_edgetpu.tflite is the basic model with 90 coco classes.
tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite model is little better in terms of accuracy.
These SSD based models achieved 60-70 fps on my hardware.

EfficientDet-Lite2 gave 6-7 fps
EfficientDet-Lite0 gave 15-16 fps
