from kevin_dl.workers.datasets.cv.torchvision_.configs.transforms import for_cifa_from_mixup_paper

transforms_for_train, transforms_for_test = for_cifa_from_mixup_paper()

dataset_s = dict()
for dataset_name in ("CIFAR10", "CIFAR100"):
    dataset_s[dataset_name] = dict()
    for task_type in ("train", "val", "test"):
        dataset_s[dataset_name][f'for_{task_type}'] = {
            "dataset": {
                "name": f':cv:torchvision:{dataset_name}',
                "paras": {
                    "root": '~/data',
                    "download": True,
                    "transform": transforms_for_train if task_type == "train" else transforms_for_test
                }
            },
            "subset": None,
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
    from kevin_toolbox.patches import for_logging

    output_dir = os.path.abspath(os.path.dirname(__file__))

    logger = for_logging.build_logger(
        name=":temp",
        handler_ls=[
            dict(target=os.path.join(output_dir, "log.txt"), level="DEBUG"),
            dict(target=None, level="DEBUG"),
        ]
    )

    # ---------------- CIFA10/100 ------------------- #

    for dataset_name, it in dataset_s.items():
        logger.info(f'generating config for dataset {dataset_name}')
        ndl.serializer.write(
            var=it,
            output_dir=os.path.join(output_dir, dataset_name),
            settings=[
                {"match_cond": lambda _, idx, value: ndl.name_handler.parse_name(idx)[-1][-1] in ["transform",
                                                                                                  "transforms"],
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

        for k in it.keys():
            it[k] = build_dataset(**it[k])

        # show image
        for k, details in it.items():
            logger.info(k)
            data = details["dataset"][0]
            # convert_format(image=data.pop('fin'), output_format=Image_Format.PIL_IMAGE).show()
            logger.info(
                f'image shape:{convert_format(image=data[0], output_format=Image_Format.NP_ARRAY).shape}, label:{data[1]}')
            logger.info(data)

        # show nums
        ndl.traverse(var=it, match_cond=lambda _, __, v: not isinstance(v, (list, dict)) and hasattr(v, "__len__"),
                     action_mode="replace", converter=lambda _, v: len(v))
        logger.info(markdown.generate_list(var=it))
