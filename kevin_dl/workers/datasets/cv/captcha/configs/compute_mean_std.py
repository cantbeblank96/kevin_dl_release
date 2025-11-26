# import numpy as np
# import torchvision
# from torchvision import transforms
# from torch.utils.data import DataLoader

def compute_mean_std(data_loader):
    mean = 0.
    std = 0.
    total_images = 0
    for data in data_loader:
        images = data["fin"]
        batch_samples = images.size(0)  # 当前 batch 的样本数
        images = images.view(batch_samples, images.size(1), -1)  # reshape: N, C, H*W
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean, std


if __name__ == '__main__':
    import os
    import kevin_toolbox.nested_dict_list as ndl
    from kevin_dl.workers.datasets import build_dataset

    it = ndl.serializer.read(input_path=os.path.join(os.path.dirname(__file__), "CAPTCHA95_for_mixup_with_real"))
    # 只保留 to_tensor 部分
    it["for_val"]["dataset"]["paras"]["transforms"]["settings"].pop(-1)
    print(it["for_val"]["dataset"]["paras"]["transforms"])
    data_loader = build_dataset(**it["for_val"])["data_loader"]
    mean, std = compute_mean_std(data_loader)
    print(mean, std)
    # mean: [0.8643, 0.8644, 0.8643]
    # std: [0.1666, 0.1666, 0.1666]
