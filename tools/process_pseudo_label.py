def pseudoProcess(json_path_pseudo, json_path_real, json_path_pseudo_new, score=0.2):
    with open(json_path_pseudo, 'r') as f:
        data = json.load(f)
        print(str(json_path_pseudo) + "Load Done")
    with open(json_path_real, 'r') as f:
        data2 = json.load(f)
        print(str(json_path_real) + "Load Done")
    dataold = data
    datanew = list()
    for i in tqdm(dataold):
        # filter pseudo label according to confidence score
        if i['score'] > score:
            datanew.append(i)
    data2['annotations'] = datanew
    with open(json_path_pseudo_new, 'w') as f_obj3:
        json.dump(data2, f_obj3)
    print('pseudoProcess Done')

if __name__ == '__main__':
    import argparse
    import json
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description='statistics')
    parser.add_argument("--json_path_target_real", type=str, default='datasets/index1/coco20/annotations/instances_train2017_new.json')
    parser.add_argument("--json_path_target_pseudo", type=str, default='output/index1/baseline/voc07train+voc12trainval_sup100/pseudo_label/inference/coco_instances.json')
    parser.add_argument("--json_path_target_pseudo_new", type=str, default='output/pseudo_label/index1/index1_target_pseudo.json')
    parser.add_argument("--score", type=float, default=0.2)
    args = parser.parse_args()
    pseudoProcess(args.json_path_target_pseudo, args.json_path_target_real, args.json_path_target_pseudo_new, score=args.score)