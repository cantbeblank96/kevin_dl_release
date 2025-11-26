# 关于 imagenet 的数据预处理方式，参考自：
#   - https://github.com/jiweibo/ImageNet/blob/master/data_loader.py
#   - https://stackoverflow.com/questions/67185623/image-net-preprocessing-using-torch-transforms
#   - Mobilenets: Efficient convolutional neural networks for mobile vision applications.
#       arXiv preprint arXiv:1704.04861 (2017)
#   - Inverted residuals and linear bottlenecks: Mobile networks for classification, detection and segmentation.
#       arXiv preprint arXiv:1801.04381 (2018)
#   - https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2

transforms_for_train = {
    "settings": [
        # 数据扩增
        {
            "name": ':for_images:torchvision:RandomResizedCrop',
            "paras": {
                "size": 224
            }
        },
        {
            "name": ':for_images:torchvision:ColorJitter',
            "paras": {
                "brightness": 0.4,
                "contrast": 0.4,
                "saturation": 0.4
            }
        },
        {
            "name": ':for_images:torchvision:RandomHorizontalFlip',
            "paras": {
                "p": 0.5
            }
        },
        # 归一化
        {
            "name": ':for_images:torchvision:Normalize',
            "paras": {
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225)
            }
        }
    ]
}

transforms_for_test = {
    "settings": [
        # 裁剪缩放
        {
            "name": ':for_images:torchvision:Resize',
            "paras": {
                "size": 256
            }
        },
        {
            "name": ':for_images:torchvision:CenterCrop',
            "paras": {
                "size": 224
            }
        },
        # 归一化
        transforms_for_train["settings"][-1]
    ]
}

ILSVRC2012 = dict()
for task_type in ("train", "val", "test"):
    ILSVRC2012[f'for_{task_type}'] = {
        "dataset": {
            "name": ":cv:classification:Imagenet_Dataset",
            "paras": {
                "ann_path": f"<ceph>openmmlab:s3://openmmlab/datasets/classification/imagenet/meta/{task_type}.txt",
                "prefix": f"<ceph>openmmlab:s3://openmmlab/datasets/classification/imagenet/{task_type}",
                "transforms": transforms_for_train if task_type == "train" else transforms_for_test,
                # 输出
                "output_contents": ("fin", "image_path", "label"),
            }
        },
        "subset": None,
        "data_loader": {
            "batch_size": 1024 if task_type == "train" else 200,
            "num_workers": 4,
            "drop_last": True if task_type == "train" else False,
            "shuffle": True if task_type == "train" else False,
            "pin_memory": True
        },
        "seed": 114514,
        "task_type": task_type
    }

if __name__ == '__main__':
    import os
    import kevin_toolbox.nested_dict_list as ndl
    from kevin_toolbox.patches import for_logging

    output_dir = os.path.abspath(os.path.dirname(__file__))

    logger = for_logging.build_logger(
        name=":temp",
        handler_ls=[
            dict(target=os.path.join(output_dir, "log.txt"), level="DEBUG"),
            dict(target=None, level="DEBUG"),
        ]
    )

    # ---------------- ILSVRC2012 for ceph ------------------- #

    logger.info(f'generating config for dataset {"ILSVRC2012_for_ceph"}')
    ndl.serializer.write(
        var=ILSVRC2012,
        output_dir=os.path.join(output_dir, "ILSVRC2012_for_ceph"),
        settings=[
            {"match_cond": lambda _, idx, value: ndl.name_handler.parse_name(idx)[-1][-1] == "transforms",
             "backend": (":json",)},
            {"match_cond": "<level>-1",
             "backend": (":skip:simple",)},
        ],
        b_pack_into_tar=False
    )

    # test
    from kevin_dl.workers.datasets import build_dataset
    from kevin_toolbox.data_flow.file import markdown
    from kevin_dl.workers.transforms.for_images.utils import convert_format, Image_Format

    for k in ILSVRC2012.keys():
        ILSVRC2012[k] = build_dataset(**ILSVRC2012[k])

    # show image
    for k, details in ILSVRC2012.items():
        logger.info(k)
        data = details["dataset"][0]
        # convert_format(image=data.pop('fin'), output_format=Image_Format.PIL_IMAGE).show()
        logger.info(convert_format(image=data.pop('fin'), output_format=Image_Format.NP_ARRAY).shape)
        logger.info(data)

    # show nums
    ndl.traverse(var=ILSVRC2012, match_cond=lambda _, __, v: not isinstance(v, (list, dict)) and hasattr(v, "__len__"),
                 action_mode="replace", converter=lambda _, v: len(v))
    logger.info(markdown.generate_list(var=ILSVRC2012))
