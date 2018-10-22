from keras.models import load_model
import configparser
import io
from collections import defaultdict

import numpy as np


# h5文件载入
model = load_model('model_weight.h5',compile=False)
# .cfg 路径
config_path='yolo.cfg'
# .weights写入
weight_file = open('yolo.weights', 'wb')




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

    # 版本号 major, minor, revision=[0,2,0] seen=32013312 写入
    m_revision=[0,2,0]
    seen=[32013312]
    # 转化成 bytes
    m_revision_const = np.array(m_revision,dtype=np.int32)
    m_revision_bytes=m_revision_const.tobytes()

    seen_const=np.array(seen,dtype=np.int64)
    seen_bytes=seen_const.tobytes()

    print('开始写入版本号\n')
    # bytes 写入
    weight_file.write(m_revision_bytes)
    weight_file.write(seen_bytes)

    # conv2d 与batch_normalize 层错位参数
    b=0
    print('开始写入权值\n')
    for section in cfg_parser.sections():
        #print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            # 获取 'convolutional_'序号 可能存在问题
            num = int(section.split('_')[-1])+1
            # batch_normalize 是否存在
            batch_normalize = 'batch_normalize' in cfg_parser[section]
            # 存在 batch_normalize 处理时 写入 三次 (存在时 activation='leaky')
            if batch_normalize:
                # 从batch_normalization layer 提取 bn_weight_list
                batch_weight_name = 'batch_normalization_' + str(num-b)
                bn_weight_list_layer=model.get_layer(batch_weight_name)
                bn_weight_list =bn_weight_list_layer.get_weights()

                # 从 bn_weight_list 提取bn_weight 和con_bias
                conv_bias = bn_weight_list[1]
                bn_weight = [bn_weight_list[0], bn_weight_list[2], bn_weight_list[3]]

                # 从 conv2d layer 提取conv_weight
                conv2d_weight_name = 'conv2d_' + str(num)
                # 打印编号
                print(conv2d_weight_name,'\n')
                print(batch_weight_name, '\n')
                conv2d_weight_name_layer=model.get_layer(conv2d_weight_name)
                # list[ndarray]
                conv_weight = conv2d_weight_name_layer.get_weights()
                conv_weight=conv_weight[0]
                # 更改
                conv_weight = np.transpose(conv_weight, [3, 2, 0, 1])
                # conv_bias 转 bytes 并写入
                bias_weight = np.array(conv_bias,dtype=np.float32)
                bytes_bias_weight=bias_weight.tobytes()
                weight_file.write(bytes_bias_weight)
                print(bias_weight.shape,'\n')

                # bn_weight 转 bytes 并写入
                bn_weight_array=np.array(bn_weight,dtype=np.float32)
                bytes_bn_weight=bn_weight_array.tobytes()
                weight_file.write(bytes_bn_weight)
                print(bn_weight_array.shape,'\n')

                # conv_weight 转 bytes 并写入
                conv_weight_array=np.array(conv_weight,dtype=np.float32)
                bytes_conv_weight=conv_weight_array.tobytes()
                weight_file.write(bytes_conv_weight)

                #打印形状
                print(conv_weight_array.shape,'\n')

            # 不存在 batch_normalize 处理时  写入两次
            else:
                # b作为错位参数
                b+=1
                # 从 conv2d layer 提取conv_weight（含conv_bias)
                print('\n')
                conv2d_weight_name = 'conv2d_' + str(num)
                print('错位',conv2d_weight_name,'\n\n')
                conv2d_weight_name_layer = model.get_layer(conv2d_weight_name)
                conv_weights =conv2d_weight_name_layer.get_weights()

                # 解析出 conv_bias和 conv2d_weight
                conv_bias = conv_weights[-1]
                conv_weight = conv_weights[0]
                conv_weight=np.array(conv_weight)
                # 转置还原
                conv_weight = np.transpose(conv_weight, [3, 2, 0, 1])

                # 按顺序写入 conv_bias、conv2d_weight
                # conv_bias 转 bytes 并写入
                bias_weight = np.array(conv_bias,dtype=np.float32)
                bytes_bias_weight = bias_weight.tobytes()
                weight_file.write(bytes_bias_weight)
                print(bias_weight.shape)
                # conv_weight 转 bytes 并写入
                conv_weight_array = np.array(conv_weight,dtype=np.float32)
                bytes_conv_weight = conv_weight_array.tobytes()
                weight_file.write(bytes_conv_weight)
                # 打印形状
                print(conv_weight_array.shape)

    weight_file.close()
    print("转换结束\n")





if __name__ == '__main__':
    _main()