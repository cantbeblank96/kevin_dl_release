import os
import torch
import numpy as np
from kevin_toolbox.data_flow.file import json_
from kevin_toolbox.patches.for_test import check_consistency
from kevin_toolbox.computer_science.algorithm.scheduler import Trigger
import kevin_toolbox.nested_dict_list as ndl
from kevin_dl.workflow.config_handler import load_config, build_exp_from_config
from kevin_dl.workers.variable import DATASETS, MODELS, OPTIMIZERS
from kevin_dl.workers.optimizers import Advanced_Optimizer

data_folder = os.path.join(os.path.dirname(__file__), "test_data")


def test_load_config_json():
    print("test config_handler.load_config() for json file")

    # b_parse_ref=False
    cfg = load_config(file_path=os.path.join(data_folder, "config_0.json"), b_parse_ref=False)
    check_consistency(json_.read(file_path=os.path.join(data_folder, "config_0.json")), cfg)

    # b_parse_ref=True
    #   解释带有 <cfg>{ } 标记的引用
    cfg = load_config(file_path=os.path.join(data_folder, "config_0.json"), b_parse_ref=True)
    check_consistency(json_.read(file_path=os.path.join(data_folder, "config_0_expected.json")), cfg)


def test_load_config_ndl():
    print("test config_handler.load_config()  for ndl file")

    # b_parse_ref=False
    cfg_0 = load_config(file_path=os.path.join(data_folder, "config_1"), b_parse_ref=False)

    # b_parse_ref=True
    #   解释带有 <cfg>{ } 标记的引用
    cfg_1 = load_config(file_path=os.path.join(data_folder, "config_1"), b_parse_ref=True)


def test_build_exp_from_config():
    print("test config_handler.build_exp_from_config()")

    cfg_ = load_config(file_path=os.path.join(data_folder, "config_1"), b_parse_ref=True)

    exp_ = build_exp_from_config(cfg=cfg_)
    # 检查是否成功构建 模型、数据集 等的实例
    for name, v in ndl.get_nodes(var=exp_["dataset"], level=-1, b_strict=True):
        if "name" in ndl.get_value(var=cfg_["dataset"], name=name):
            assert isinstance(v, DATASETS.get(name=ndl.get_value(var=cfg_["dataset"], name=name)["name"]))
    assert isinstance(exp_["model"], MODELS.get(name=cfg_["model"]["name"])) or \
           isinstance(exp_["model"], torch.nn.DataParallel)
    assert isinstance(exp_["optimizer"], OPTIMIZERS.get(name=cfg_["optimizer"]["name"])) or \
           isinstance(exp_["optimizer"], Advanced_Optimizer)
    assert isinstance(exp_["trigger"], Trigger)
