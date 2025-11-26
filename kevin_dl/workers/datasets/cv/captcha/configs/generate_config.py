def get_visible_ascii_chars():
    """
    返回所有标准 ASCII 中的可见（非空）字符列表。
    范围是 ASCII 编码 32 到 126，包括空格、标点、数字、字母等。

    返回：
        List[str]: 95 个可见字符组成的列表
    """
    return [chr(i) for i in range(32, 127)]


size = 64

transforms_for_train = {
    "settings": [
        {
            "name": ':for_images:torchvision:ToTensor',
            "paras": {}
        },
        # 数据扩增
        {
            "name": ":for_images:torchvision:RandomAffine",
            "paras": {
                "degrees": 15, "translate": (0.02, 0.05), "shear": 15, "fill": (1.0, 1.0, 1.0),
            }
        },
        {
            "name": ":for_images:torchvision:GaussianBlur",
            "paras": {
                "kernel_size": 3, "sigma": (0.01, 1.5)
            }
        },
        {
            "name": ":for_images:torchvision:ColorJitter",
            "paras": {
                "brightness": 0.1, "contrast": 0.1, "saturation": 0.1
            }
        },
        # 归一化
        {
            "name": ':for_images:torchvision:Normalize',
            "paras": {
                "mean": (0.8643, 0.8644, 0.8643),
                "std": (0.1666, 0.1666, 0.1666)
            }
        }
    ]
}

transforms_for_test = {
    "settings": [
        # 归一化
        transforms_for_train["settings"][0],
        transforms_for_train["settings"][-1]
    ]
}

label_maker = {
    "options": get_visible_ascii_chars(),
    "weights": 1.0,
}

captcha_maker = {
    #
    "width": size,
    "height": size,
    "font_size": int(size * 0.8),
    "color_options_of_text": None,
    "color_of_bg": 255,
    #
    "max_offset_of_text": (0.5, 0.5),
    "b_use_transforms": True,
    "transforms": [
        {
            "name": ":for_images:torchvision:ToTensor",
            "paras": {}
        },
        {
            "name": ":for_images:torchvision:RandomAffine",
            "paras": {
                "degrees": 30, "translate": (0.05, 0.1), "shear": 30, "fill": (1.0, 1.0, 1.0),
            }
        },
        {
            "name": ":for_images:torchvision:GaussianBlur",
            "paras": {
                "kernel_size": 3, "sigma": (0.1, 1.5)
            }
        },

        {
            "name": ":for_images:torchvision:ColorJitter",
            "paras": {
                "brightness": 0.3, "contrast": 0.3, "saturation": 0.3
            }
        }
    ],
    "b_add_noise_line": True,
    "color_options_of_line": ("gray",),
    "b_add_noise_point": True,
}

dataset_s = dict()
# CAPTCHA95_m_n
#   其中 CAPTCHA95 表示 95 个可见字符（含空格）
#   m 表示训练集中每个类别的数据量
#   n 表示测试集的每个类别的数据量。
for dataset_name in ("CAPTCHA95_500_100", "CAPTCHA95_200000_1000"):
    dataset_s[dataset_name] = dict()
    num_per_cls_of_train, num_per_cls_of_test = [int(i) for i in dataset_name.split("_")[1:]]
    for task_type in ("train", "val", "test"):
        dataset_size = num_per_cls_of_test if task_type == "test" else num_per_cls_of_train
        dataset_size *= len(label_maker["options"])
        dataset_s[dataset_name][f'for_{task_type}'] = {
            "dataset": {
                "name": f':cv:Captcha_Dataset:0.0',
                "paras": dict(
                    captcha_maker=captcha_maker,
                    label_maker=label_maker,
                    cache_dir=f'~/data/captcha/{dataset_name}/{task_type if task_type == "test" else "train"}',
                    dataset_size=dataset_size,
                    transforms=transforms_for_train if task_type == "train" else transforms_for_test,
                    task_type=task_type,
                    output_contents=["raw", "fin", "label"],
                ),
            },
            "subset": None,
            "adjuster": None,
            "data_loader": {
                "batch_size": 128 if task_type == "train" else 100,
                "num_workers": 8,
                "drop_last": False,
                "shuffle": True if task_type == "train" else False,
                "pin_memory": True
            },
            "seed": 114514,
            "task_type": task_type
        }

if __name__ == '__main__':
    import os
    import kevin_toolbox.nested_dict_list as ndl

    output_dir = os.path.abspath(os.path.dirname(__file__))

    for dataset_name, it in dataset_s.items():
        print(f'generating config for dataset {dataset_name}')
        out_file = ndl.serializer.write(
            var=it,
            output_dir=os.path.join(output_dir, dataset_name),
            settings=[
                {"match_cond": lambda _, idx, value: ndl.name_handler.parse_name(idx)[-1][-1] in ["transform",
                                                                                                  "transforms",
                                                                                                  "label_maker",
                                                                                                  "captcha_maker"],
                 "backend": (":json",)},
                {"match_cond": "<level>-1", "backend": (":skip:simple",)},
            ],
            b_pack_into_tar=False, b_keep_identical_relations=True
        )

        # test
        from kevin_dl.workers.datasets import build_dataset
        from kevin_dl.workers.transforms.for_images.utils import convert_format, Image_Format

        it = ndl.serializer.read(input_path=out_file)

        for k in it.keys():
            it[k] = build_dataset(**it[k])

        # show image
        for k, details in it.items():
            data = details["dataset"][0]
            # convert_format(image=data.pop('fin'), output_format=Image_Format.PIL_IMAGE).show()
            print(
                f'image shape:{convert_format(image=data["fin"], output_format=Image_Format.NP_ARRAY).shape}, label:{data["label"]}')
