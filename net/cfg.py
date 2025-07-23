# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/18 15:11
@Author  : Kend
@FileName: cfg
@Software: PyCharm
@modifier:
"""

""" 
自定义配置而不使用（CfgNode），构建一个字典结构来传递给 MoViNet 模型。MoViNet 的构造函数接受一个 cfg 参数，它是一个类似 CfgNode 的对象，只要具备相应的字段即可。
手动构造一个 cfg 字典，模拟 _C.MODEL.MoViNetA0 的结构，用于加载 MoViNet 模型。
"""

from types import SimpleNamespace


def dict_to_namespace(d):
    """递归将嵌套字典转换为 SimpleNamespace 对象"""
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = dict_to_namespace(v)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d



def build_movinet_a0_cfg():

    cfg_dict = {
        "name": "A0",
        # "weights": "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA0_statedict_v3?raw=true",
        "weights": "/home/kend/Guanxin/work/workspace/movinet-pytorch/net/modelA0_statedict",
        "stream_weights": "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA0_stream_statedict_v3?raw=true",

        # conv1
        "conv1": {
            "input_channels": 3,
            "out_channels": 8,
            "kernel_size": (1, 3, 3),
            "stride": (1, 2, 2),
            "padding": (0, 1, 1)
        },

        # blocks
        "blocks": [
            [
                {
                    "input_channels": 8,
                    "out_channels": 8,
                    "expanded_channels": 24,
                    "kernel_size": (1, 5, 5),
                    "stride": (1, 2, 2),
                    "padding": (0, 2, 2),
                    "padding_avg": (0, 1, 1)
                }
            ],
            [
                {
                    "input_channels": 8,
                    "out_channels": 32,
                    "expanded_channels": 80,
                    "kernel_size": (3, 3, 3),
                    "stride": (1, 2, 2),
                    "padding": (1, 0, 0),
                    "padding_avg": (0, 0, 0)
                },
                {
                    "input_channels": 32,
                    "out_channels": 32,
                    "expanded_channels": 80,
                    "kernel_size": (3, 3, 3),
                    "stride": (1, 1, 1),
                    "padding": (1, 1, 1),
                    "padding_avg": (0, 1, 1)
                },
                {
                    "input_channels": 32,
                    "out_channels": 32,
                    "expanded_channels": 80,
                    "kernel_size": (3, 3, 3),
                    "stride": (1, 1, 1),
                    "padding": (1, 1, 1),
                    "padding_avg": (0, 1, 1)
                }
            ],
            [
                {
                    "input_channels": 32,
                    "out_channels": 56,
                    "expanded_channels": 184,
                    "kernel_size": (5, 3, 3),
                    "stride": (1, 2, 2),
                    "padding": (2, 0, 0),
                    "padding_avg": (0, 0, 0)
                },
                {
                    "input_channels": 56,
                    "out_channels": 56,
                    "expanded_channels": 112,
                    "kernel_size": (3, 3, 3),
                    "stride": (1, 1, 1),
                    "padding": (1, 1, 1),
                    "padding_avg": (0, 1, 1)
                },
                {
                    "input_channels": 56,
                    "out_channels": 56,
                    "expanded_channels": 184,
                    "kernel_size": (3, 3, 3),
                    "stride": (1, 1, 1),
                    "padding": (1, 1, 1),
                    "padding_avg": (0, 1, 1)
                }
            ],
            [
                {
                    "input_channels": 56,
                    "out_channels": 56,
                    "expanded_channels": 184,
                    "kernel_size": (5, 3, 3),
                    "stride": (1, 1, 1),
                    "padding": (2, 1, 1),
                    "padding_avg": (0, 1, 1)
                },
                {
                    "input_channels": 56,
                    "out_channels": 56,
                    "expanded_channels": 184,
                    "kernel_size": (3, 3, 3),
                    "stride": (1, 1, 1),
                    "padding": (1, 1, 1),
                    "padding_avg": (0, 1, 1)
                },
                {
                    "input_channels": 56,
                    "out_channels": 56,
                    "expanded_channels": 184,
                    "kernel_size": (3, 3, 3),
                    "stride": (1, 1, 1),
                    "padding": (1, 1, 1),
                    "padding_avg": (0, 1, 1)
                },
                {
                    "input_channels": 56,
                    "out_channels": 56,
                    "expanded_channels": 184,
                    "kernel_size": (3, 3, 3),
                    "stride": (1, 1, 1),
                    "padding": (1, 1, 1),
                    "padding_avg": (0, 1, 1)
                }
            ],
            [
                {
                    "input_channels": 56,
                    "out_channels": 104,
                    "expanded_channels": 384,
                    "kernel_size": (5, 3, 3),
                    "stride": (1, 2, 2),
                    "padding": (2, 1, 1),
                    "padding_avg": (0, 1, 1)
                },
                {
                    "input_channels": 104,
                    "out_channels": 104,
                    "expanded_channels": 280,
                    "kernel_size": (1, 5, 5),
                    "stride": (1, 1, 1),
                    "padding": (0, 2, 2),
                    "padding_avg": (0, 1, 1)
                },
                {
                    "input_channels": 104,
                    "out_channels": 104,
                    "expanded_channels": 280,
                    "kernel_size": (1, 5, 5),
                    "stride": (1, 1, 1),
                    "padding": (0, 2, 2),
                    "padding_avg": (0, 1, 1)
                },
                {
                    "input_channels": 104,
                    "out_channels": 104,
                    "expanded_channels": 344,
                    "kernel_size": (1, 5, 5),
                    "stride": (1, 1, 1),
                    "padding": (0, 2, 2),
                    "padding_avg": (0, 1, 1)
                }
            ]
        ],

        # conv7
        "conv7": {
            "input_channels": 104,
            "out_channels": 480,
            "kernel_size": (1, 1, 1),
            "stride": (1, 1, 1),
            "padding": (0, 0, 0)
        },

        # dense9
        "dense9": {
            "hidden_dim": 2048
        }
    }
    # TODO: 使用 types.SimpleNamespace 来递归地将嵌套字典转换为对象，从而对象访问方式, 需要注意多层的转换- DBUG 0718
    return dict_to_namespace(cfg_dict)

