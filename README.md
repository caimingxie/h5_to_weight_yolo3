# keras yolov3 h5 file to yolov3.weights
convert keras tensorflow backend version yolov3 h5 model to darknet yolov3 weights.

keras tensorflow trained yolov3 saved to *.h5 model, then translate the h5 model weight to original darknet yolov3 weights.
Because train the original darknet yolov3 is not easy, but keras train yolov3 well.

1. voc_classes.txt:
   check there is not line break in the file end, otherwise line break will regard as one class.

2. yolo.cfg:
modify the yolo.cfg to corresponding own train dataset, change the below items.

width=960   # train image's width
height=512  # train image's height
....
filters=21  # 3*(5+num_classes)
activation=linear


[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=2   #   classes number
....
filters=21  # 3*(5+num_classes)
activation=linear


[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=2   # classes number
....
filters=21  # 3*(5+num_classes)
activation=linear
.....
[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=2   # classes number


3. open check_weight.py and modify model_path, config_path, weight_file of yourself.

    model_path = "./trained_weights_final.h5"     # keras yolov3 h5 model file
    config_path = 'yolov3.cfg'                   # .cfg  file path
    weight_file = open('yolov3.weights', 'wb')   # save darknet yolov3 weights file path

    run  python check_weight.py
