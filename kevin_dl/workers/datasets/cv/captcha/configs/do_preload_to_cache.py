if __name__ == '__main__':
    import os
    import kevin_toolbox.nested_dict_list as ndl
    from kevin_dl.workers.datasets import build_dataset

    it = ndl.serializer.read(input_path=os.path.join(os.path.dirname(__file__), "CAPTCHA95_for_mixup_with_real"))

    for k in it.keys():
        print(k)
        it[k] = build_dataset(**it[k])
        if "train" in k or "test" in k:
            it[k]["dataset"].preload_to_cache()