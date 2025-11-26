import os
import pytest
import torch
import numpy as np
from kevin_toolbox.data_flow.file import json_
import kevin_toolbox.nested_dict_list as ndl
from kevin_toolbox.patches.for_test import check_consistency
from kevin_dl.workflow.config_handler import load_config, build_exp_from_config
from kevin_dl.workflow.state_manager import save_state, load_state

temp_folder = os.path.join(os.path.dirname(__file__), "temp")
data_folder = os.path.join(os.path.dirname(__file__), "test_data")


def test_save_and_load_state():
    print("test state_manager.save_state() and load_state()")

    cfg_ = load_config(file_path=os.path.join(data_folder, "config_1"), b_parse_ref=True)

    exp_ = build_exp_from_config(cfg=cfg_)
    exp_["trigger"].update_by_state(cur_state=dict(epoch=99))
    exp_["trigger"].update_by_state(cur_state=dict(epoch=100))
    exp_["trigger"].update_by_state(cur_state=dict(epoch=101))

    #
    def func(exp):
        torch_rand_nums, np_rand_nums = torch.rand(10), np.random.rand(10)
        dataset_batch = None
        for i in exp["dataset"]["ILSVRC2012mini_with_Randomly_Blurred"]["for_train"]["data_loader"]:
            dataset_batch = i
            break
        exp["model"].eval()
        model_out = exp["model"](dataset_batch["fin_wout_blur"], dataset_batch["fin_with_blur"])
        lr = exp["optimizer"].param_groups[0]['lr']
        return torch_rand_nums, np_rand_nums, dataset_batch, model_out, lr

    # 保存状态
    save_state(exp=exp_, output_dir=temp_folder, file_name="state_0", b_save_non_state_part=False, b_verbose=True)

    # 获取该状态下的结果
    res = func(exp=exp_)

    # 新建一个实验
    del exp_
    exp_ = build_exp_from_config(cfg=cfg_)

    # 新状态下的结果
    res_1 = func(exp=exp_)

    # 加载状态
    load_state(exp=exp_, input_dir=temp_folder, file_name="state_0", b_load_non_state_part=False, b_verbose=True)

    # 获取该状态下的结果
    res_2 = func(exp=exp_)

    # 检查
    #   新状态与旧状态不一致
    with pytest.raises(AssertionError):
        check_consistency(list(res), list(res_1))
    #   但是经过加载之后，状态变成一致的了
    check_consistency(list(res), list(res_2))
