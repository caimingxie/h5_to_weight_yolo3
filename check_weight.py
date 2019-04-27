from keras.models import load_model
import configparser
import io
from collections import defaultdict
import numpy as np
from yolo import YOLO

def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream

def _main():

    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    # major, minor, revision=[0,2,0] seen=32013312
    m_revision=[0,2,0]
    seen=[32013312]
    # convert to  bytes
    m_revision_const = np.array(m_revision,dtype=np.int32)
    m_revision_bytes=m_revision_const.tobytes()

    seen_const=np.array(seen,dtype=np.int64)
    seen_bytes=seen_const.tobytes()

    print('write revision information\n')
    weight_file.write(m_revision_bytes)
    weight_file.write(seen_bytes)

    # conv2d and batch_normalize layers
    b=0
    print('start write weights\n')
    for section in cfg_parser.sections():

        #print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            # get 'convolutional_'
            num = int(section.split('_')[-1])+1
            # get 'batch_normalize'
            batch_normalize = 'batch_normalize' in cfg_parser[section]
            # if batch_normalize write it three times and  activation='leaky'
            if batch_normalize:
                # from batch_normalization layer extract bn_weight_list
                batch_weight_name = 'batch_normalization_' + str(num-b)
                bn_weight_list_layer=model.get_layer(batch_weight_name)
                bn_weight_list =bn_weight_list_layer.get_weights()

                # from bn_weight_list extract bn_weight and con_bias
                conv_bias = bn_weight_list[1]
                bn_weight = [bn_weight_list[0], bn_weight_list[2], bn_weight_list[3]]

                # from conv2d layer extract conv_weight
                conv2d_weight_name = 'conv2d_' + str(num)
                # print conv2d_weight_name
                print(conv2d_weight_name,'\n')
                print(batch_weight_name, '\n')
                conv2d_weight_name_layer=model.get_layer(conv2d_weight_name)
                # list[ndarray]
                conv_weight = conv2d_weight_name_layer.get_weights()
                conv_weight=conv_weight[0]
                conv_weight = np.transpose(conv_weight, [3, 2, 0, 1])
                bias_weight = np.array(conv_bias,dtype=np.float32)
                bytes_bias_weight=bias_weight.tobytes()
                weight_file.write(bytes_bias_weight)
                print(bias_weight.shape,'\n')

                # convert bn_weight to bytes then write to file
                bn_weight_array=np.array(bn_weight,dtype=np.float32)
                bytes_bn_weight=bn_weight_array.tobytes()
                weight_file.write(bytes_bn_weight)
                print(bn_weight_array.shape,'\n')

                conv_weight_array=np.array(conv_weight,dtype=np.float32)
                bytes_conv_weight=conv_weight_array.tobytes()
                weight_file.write(bytes_conv_weight)
                print(conv_weight_array.shape,'\n')

            # not  existence batch_normalize layers, write it two times
            else:
                # b is disorder parameter
                b+=1
                # from conv2d layer extract conv_weight（include conv_bias)
                print('\n')
                conv2d_weight_name = 'conv2d_' + str(num)
                print('disorder',conv2d_weight_name,'\n\n')
                conv2d_weight_name_layer = model.get_layer(conv2d_weight_name)
                conv_weights =conv2d_weight_name_layer.get_weights()

                # extract conv_bias conv2d_weight
                conv_bias = conv_weights[-1]
                conv_weight = conv_weights[0]
                conv_weight=np.array(conv_weight)
                # transpose
                conv_weight = np.transpose(conv_weight, [3, 2, 0, 1])

                # write the file with order conv_bias、conv2d_weight
                # conv_bias convert to  bytes
                bias_weight = np.array(conv_bias,dtype=np.float32)
                bytes_bias_weight = bias_weight.tobytes()
                weight_file.write(bytes_bias_weight)
                print(bias_weight.shape)
                # conv_weight convert to bytes
                conv_weight_array = np.array(conv_weight,dtype=np.float32)
                bytes_conv_weight = conv_weight_array.tobytes()
                weight_file.write(bytes_conv_weight)
                # pritn the shape
                print(conv_weight_array.shape)

    weight_file.close()
    print("convert success!\n")


if __name__ == '__main__':

    model_path = "./trained_weights_final.h5"     # keras yolov3 h5 model file
    config_path = 'yolov3.cfg'                   # .cfg  file path
    weight_file = open('yolov3.weights', 'wb')   # save darknet yolov3 weights file path

    """
    The default keras yolov3 (https://github.com/qqwweee/keras-yolo3/blob/master/train.py)
    after trained save with method " model.save_weights(log_dir + 'trained_weights_final.h5')"
    it actually only saved weights, below call YOLO(modelpath) will check it's model,
    if it without model information, then automatic load model.

    """
    yoloobj = YOLO(model_path)
    model = yoloobj.yolo_model
    _main()