import os
import pytest
from PIL import Image
from kevin_dl.utils.variable import root_dir
from kevin_dl.workers.transforms.for_images.utils import get_format, convert_format, Image_Format
from kevin_dl.workers.transforms import Pipeline
from kevin_dl.workers.transforms.pipeline import map_to_list, map_to_dict
from kevin_toolbox.patches.for_test import check_consistency

settings_ = [
    {
        "name": ':for_images:blur:Gaussian_Blur',
        "paras": {
            "sigma": {
                "p_type": "float",
                "p_prob": "uniform",
                "high": 10,
                "low": 0,
            }
        }
    },
    {
        "name": ':for_images:color:Brightness_Shift',
        "paras": {
            "beta": {
                "p_type": "float",
                "p_prob": "uniform",
                "high": 0.2,
                "low": -0.2,
            }
        }
    },
    {
        "name": ':for_images:blur:Motion_Blur',
        "paras": {
            "kernel_size": {
                "p_type": "categorical",
                "choices": [1, 2, 4, 8, 16]
            },
            "angle": {
                "p_type": "int",
                "p_prob": "uniform",
                "high": 180,
                "low": 0,
            }
        }
    }
]

image_ = Image.open(
    os.path.join(root_dir, "kevin_dl/workers/transforms/for_images/test/test_data/ILSVRC2012_val_00040001.JPEG"))


def test_call():
    print("test Pipeline.__call__()")

    global settings_, image_

    pp = Pipeline(settings=settings_, drop_prob=[0.2, 0.2, 0.2], order_prob=[0.5, 0.5, 0.5], b_include_details=True)

    output_s = pp(input_s=dict(image=image_))
    check_consistency(Image_Format.NP_ARRAY, get_format(image=output_s["image"]))


def test_replay_last_process():
    print("test Pipeline.replay_last_process()")

    global settings_, image_

    pp = Pipeline(settings=settings_, drop_prob=[0.0, 0.2, 0.2], order_prob=[0.5, 0.5, 0.5], b_include_details=True,
                  replay_times=2)

    # 前 replay_times 次保持不变
    output_s = pp(input_s=dict(image=image_))
    output_s_1 = pp(input_s=dict(image=image_))
    # 即使临时改了参数范围，还是不变
    pp.paras['drop_prob'] = [1, 1, 1]
    output_s_2 = pp(input_s=dict(image=image_))
    check_consistency(*[i['details_ls'] for i in [output_s, output_s_1, output_s_2]])
    check_consistency(*[i['image'] for i in [output_s, output_s_1, output_s_2]])

    # 后面将发生变化
    output_s_3 = pp(input_s=dict(image=image_))  # 由于设置了 drop_prob 为全一，因此将抛弃所有组件，不进行任何处理
    check_consistency(output_s_3.get("details_ls", []), [])
    check_consistency(output_s_3['image'], image_)
    #
    pp.paras['drop_prob'] = [1, 0.0, 0.0]
    output_s_4 = pp(input_s=dict(image=image_))
    with pytest.raises(AssertionError):
        check_consistency(*[i['details_ls'] for i in [output_s_4, output_s]])
    with pytest.raises(AssertionError):
        check_consistency(*[i['image'] for i in [output_s_4, output_s]])

    # 重设 replay_times 为 -1，将一直保持重播
    pp.replay_last_process(replay_times=-1)
    #
    output_s_ls = []
    pp.paras['drop_prob'] = [0.0, 1, 1]
    while len(output_s_ls) < 5:
        output_s_ls.append(pp(input_s=dict(image=image_)))
    check_consistency(*[i['details_ls'] for i in output_s_ls])
    check_consistency(*[i['image'] for i in output_s_ls])

    # 重设 replay_times 为 None 时，将取消重播
    pp.replay_last_process(replay_times=None)
    output_s_5 = pp(input_s=dict(image=image_))
    with pytest.raises(AssertionError):
        check_consistency(*[i['details_ls'] for i in [output_s_5, output_s_ls[-1]]])


def test_set_rng():
    print("test Pipeline.set_rng()")

    global settings_, image_

    pp = Pipeline(settings=settings_, drop_prob=[0.2, 0.2, 0.2], order_prob=[0.5, 0.5, 0.5], b_include_details=True,
                  seed=114514)

    output_s_ls = []
    while len(output_s_ls) < 5:
        output_s_ls.append(pp(input_s=dict(image=image_)))
    with pytest.raises(AssertionError):
        check_consistency(*[i.get("details_ls", []) for i in output_s_ls])

    # 重设 seed
    pp.set_rng(seed=114514)
    output_s = pp(input_s=dict(image=image_))
    check_consistency(*[i['details_ls'] for i in [output_s, output_s_ls[0]]])


def test_map_to_dict():
    print("test map_to_dict()")

    # 正常使用
    args, kwargs = [123, ], {"b_blur": True}
    mapping_ls = ["image", "b_blur", ("seed", 114514)]
    res = map_to_dict(mapping_ls=mapping_ls, args=args, kwargs=kwargs)
    check_consistency(
        list(res),
        [[], {"input_s": {"image": 123, "b_blur": True, "seed": 114514}}]
    )

    # 不使用映射
    check_consistency(
        list(map_to_dict(mapping_ls=None, args=args, kwargs=kwargs)),
        [args, kwargs]
    )

    # 缺参数报错
    with pytest.raises(AssertionError):
        _ = map_to_dict(mapping_ls=mapping_ls, args=args, kwargs=dict())  # 将因为缺少必要参数 b_blur 而报错
    with pytest.raises(AssertionError):
        _ = map_to_dict(mapping_ls=mapping_ls, args=[], kwargs=kwargs)  # 将因为缺少必要参数 image 而报错


def test_map_to_list():
    print("test map_to_list()")

    #
    res = {"image": 123, "b_blur": True, "seed": 114514}
    mapping_ls = ["seed", "image"]
    check_consistency(
        map_to_list(mapping_ls=mapping_ls, res_s=res),
        [res[k] for k in mapping_ls]
    )

    # 不使用映射
    mapping_ls = "image"
    check_consistency(
        map_to_list(mapping_ls=mapping_ls, res_s=res),
        res["image"]
    )
