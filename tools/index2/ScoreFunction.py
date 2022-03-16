import os
import json
import numpy as np
from collections import defaultdict

from numpy.lib.function_base import diff
def norm_dict(_dict):
    v_max = max(_dict.values())
    for k, v in _dict.items():
        _dict[k] = v / v_max
    _dict['max'] = v_max
    return _dict

def PreprocessData(file_name):
    print('Loading File {}...'.format(file_name))
    with open(file_name, 'r') as f:
        data = json.load(f)
    difficult_indicators = defaultdict(float)
    information_indicators = defaultdict(float)
    diversity_indicators = defaultdict(float)
    
    for index, (image_path, boxes_info) in enumerate(data.items()):
        # boxes_info: list[box0, box1, ...]
        _difficult = 0
        _information = 0
        _diversity = 0

        _cls_set = set()
        for box in boxes_info:
            _information += box['confidence score']
            _difficult = box['entropy']
            _cls_set.add(box['pred class'])
        _diversity = len(_cls_set)
        
        difficult_indicators[image_path] = _difficult
        information_indicators[image_path] = _information
        diversity_indicators[image_path] = _diversity

    return data, difficult_indicators, information_indicators, diversity_indicators

def CombineMetrics(data,
                   difficult_indicators,
                   information_indicators, 
                   diversity_indicators, 
                   weight=[1., 1., 1.], 
                   file=None):
    f = open(file + '.txt', 'w')
    final_value = defaultdict(float)
    for image_path, _ in data.items():
        _final_value = difficult_indicators[image_path] * weight[0]+ \
                        information_indicators[image_path] * weight[1]+ \
                        diversity_indicators[image_path] * weight[2]
        final_value[image_path] = _final_value
        f.write(str(_final_value) + '\n')
    f.close()
                        
    with open(file + '.json', 'w') as f:
        f.write(json.dumps(final_value))
    
    print('Finish {}'.format(file))

def special_():
    weight = [1., 1., 1.]

    data, difficult_indicators, information_indicators, diversity_indicators = PreprocessData(file_name=static_file)
    _difficult_indicators = norm_dict(difficult_indicators)
    _information_indicators = norm_dict(information_indicators)
    _diversity_indicators = norm_dict(diversity_indicators)
    CombineMetrics(data,
                   _difficult_indicators,
                   _information_indicators, 
                   _diversity_indicators, 
                   weight=weight, 
                   file=file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='score function')
    parser.add_argument("--static-file",type=str,default='temp/index2/productline2train.json')
    parser.add_argument("--indicator-file",type=str,default='results/productline2train')
    args = parser.parse_args()
    file = args.indicator_file #the final indictor file name
    static_file = args.static_file
    special_()
        
    