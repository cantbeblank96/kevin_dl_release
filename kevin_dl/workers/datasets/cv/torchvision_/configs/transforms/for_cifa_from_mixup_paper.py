def for_cifa_from_mixup_paper():
    """
        关于 cifa 的数据预处理方式，参考自：
          - mixup 官方实现：https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
    """

    transforms_for_train = {
        "settings": [
            # 数据扩增
            {
                "name": ':for_images:torchvision:RandomCrop',
                "paras": {
                    "size": 32,
                    "padding": 4
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
                "name": ':for_images:torchvision:ToTensor',
                "paras": {}
            },
            {
                "name": ':for_images:torchvision:Normalize',
                "paras": {
                    "mean": (0.4914, 0.4822, 0.4465),
                    "std": (0.2023, 0.1994, 0.2010)
                }
            }
        ]
    }

    transforms_for_test = {
        "settings": [
            # 归一化
            transforms_for_train["settings"][-2],
            transforms_for_train["settings"][-1]
        ]
    }

    return transforms_for_train, transforms_for_test
