import os
import sys
import json
import torch

sys.path.append(os.path.dirname(__file__))
print(os.path.dirname(__file__))

from redimnet.model import ReDimNetWrap


# 读取本地 pretrained 权重文件
def load_custom(model_name='b0', train_type='ptn', dataset='vox2'):
    # 拼接本地路径
    filename = f'{model_name}-{dataset}-{train_type}.pt'
    weights_path = os.path.join(os.path.dirname(__file__), 'pretrained', filename)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"预训练模型文件未找到: {weights_path}")

    full_state_dict = torch.load(weights_path, map_location='cpu')

    model_config = full_state_dict['model_config']
    state_dict = full_state_dict['state_dict']

    model = ReDimNetWrap(**model_config)
    if train_type is not None:
        load_res = model.load_state_dict(state_dict)
        print(f"load_res : {load_res}")

    return model


def ReDimNet(model_name, train_type='ptn', dataset='vox2'):
    return load_custom(model_name, train_type=train_type, dataset=dataset)