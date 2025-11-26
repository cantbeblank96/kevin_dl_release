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
                    "train": task_type != "test",
                    "transform": transforms_for_train if task_type == "train" else transforms_for_test
                }
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
        if task_type == "train":
            dataset_s[dataset_name][f'for_{task_type}']["adjuster"] = {
                "name": f':adjuster:Class_Imbalance_Maker',
                "paras": {
                    "imbalanced_dist": {
                        "type_": "exponential_decay",
                        "paras": {"gamma": 1 / 20}
                    },
                    "class_order_func": "randomly",
                    "label_func": "<eval>lambda x: x[1]",
                    "dst_num": 1.0,
                    "b_quick_loading": True
                }
            }

if __name__ == '__main__':
    import os
    from collections import defaultdict
    from kevin_toolbox.patches.for_matplotlib import common_charts
    import kevin_toolbox.nested_dict_list as ndl

    output_dir = os.path.abspath(os.path.dirname(__file__))

    for dataset_name, it in dataset_s.items():
        print(f'generating config for dataset {dataset_name}')
        out_file = ndl.serializer.write(
            var=it,
            output_dir=os.path.join(output_dir, dataset_name + "_class_imbalance"),
            settings=[
                {"match_cond": lambda _, idx, value: ndl.name_handler.parse_name(idx)[-1][-1] in ["transform",
                                                                                                  "transforms"],
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
                f'image shape:{convert_format(image=data[0], output_format=Image_Format.NP_ARRAY).shape}, label:{data[1]}')
            # print(data)

        # 统计各个分类的样本数量
        for k, details in it.items():
            print(k)
            count_s = defaultdict(int)
            labels = []
            for x_ls, y_ls in details["data_loader"]:
                for y in y_ls:
                    y = int(y)
                    count_s[y] += 1
                    labels.append(y)
            print(count_s)
            common_charts.plot_distribution(data_s={"label": labels}, title=f"imb cifa {len(count_s)} {k}",
                                            type_="category",                                            x_name="label")
