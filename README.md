# keras yolov3 h5 model file convert to darknet yolov3.weights

## Introduction

   This script according to https://github.com/qqwweee/keras-yolo3/blob/master/convert.py implementation yolov3 train saved      h5 model convert to darknet yolov3.weights.
   
## Check and modify files

   1. voc_classes.txt:
      Check there is not a line break in the file end, otherwise line break will regard as one class. Just delete the               line break it will be ok.
      
   2. yolo.cfg:
      Modify the yolo.cfg corresponding to own train config, change the below items.
      
      width=960        # train image's width  
      height=512       # train image's heigh  
      
      ......  
      filters=18       # 3*(5+num_classes)  
      activation=linear  


      [yolo]   
      mask = 6,7,8  
      anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326    
      classes=1      # classes number  
      
      ......
      
      filters=18     # 3*(5+num_classes)   
      activation=linear    


      [yolo]   
      mask = 3,4,5   
      anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326  
      classes=1      # classes number  
      
      ......
      
      filters=18     # 3*(5+num_classes)    
      activation=linear    
      
      [yolo]  
      mask = 0,1,2    
      anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326    
      classes=1      # classes number    


   3. open check_weight.py and modify model_path, config_path, weight_file of yourself.

    model_path = "./trained_weights_final.h5"     # keras yolov3 h5 model file
    config_path = 'yolov3.cfg'                   # .cfg  file path
    weight_file = open('yolov3.weights', 'wb')   # save darknet yolov3 weights file path

## Convert start
   run python check_weight.py





