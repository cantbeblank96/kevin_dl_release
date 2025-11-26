## 2023-09-13重要更新

将原来自定义的 ":for_images:torchvision:Normalize" 重命名为 ":for_images:torchvision:ToTensor_and_Normalize"，以便与torchvision中的归一化函数进行区分。



新旧版本之间的等效性：

旧版本：

```json
[
  {
    "name": ":for_images:torchvision:Resize",
    "paras": {
      "size": 110
    }
  },
  {
    "name": ":for_images:torchvision:CenterCrop",
    "paras": {
      "size": 96
    }
  },
  {
    "name": ":for_images:torchvision:Normalize",
    "paras": {
      "mean": [
        0.485,
        0.456,
        0.406
      ],
      "std": [
        0.229,
        0.224,
        0.225
      ]
    }
  }
]
```

等效于新版本的：

```json
[
  {
    "name": ":for_images:torchvision:Resize",
    "paras": {
      "size": 110
    }
  },
  {
    "name": ":for_images:torchvision:CenterCrop",
    "paras": {
      "size": 96
    }
  },
  {
    "name": ":for_images:torchvision:ToTensor",
    "paras": {
    }
  },
  {
    "name": ":for_images:torchvision:Normalize",
    "paras": {
      "mean": [
        0.485,
        0.456,
        0.406
      ],
      "std": [
        0.229,
        0.224,
        0.225
      ]
    }
  }
]
```

或者：

```json
[
  {
    "name": ":for_images:torchvision:Resize",
    "paras": {
      "size": 110
    }
  },
  {
    "name": ":for_images:torchvision:CenterCrop",
    "paras": {
      "size": 96
    }
  },
  {
    "name": ":for_images:torchvision:ToTensor_and_Normalize",
    "paras": {
      "mean": [
        0.485,
        0.456,
        0.406
      ],
      "std": [
        0.229,
        0.224,
        0.225
      ]
    }
  }
]
```



其 torchvison 等效代码为：

```python
    pipeline = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Resize(size=110),
            transforms.CenterCrop(size=96),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(**{
                "mean": [
                    0.485,
                    0.456,
                    0.406
                ],
                "std": [
                    0.229,
                    0.224,
                    0.225
                ]
            })
        ]
    )
```

注意由于 kevin_dl/workers/transforms/for_images/torchvision_/collect_from_torchvision.py 中 accepted_format_s 对未具体指定要求输入格式的函数都是默认输入格式为 Image_Format.TORCH_TENSOR，因此在上面代码中要先使用 transforms.PILToTensor() 转换到 tensor，才能完全一致。

如果直接使用：

```python
    pipeline = transforms.Compose(
        [
            transforms.Resize(size=110),
            transforms.CenterCrop(size=96),
            transforms.ToTensor(),
            transforms.Normalize(**{
                "mean": [
                    0.485,
                    0.456,
                    0.406
                ],
                "std": [
                    0.229,
                    0.224,
                    0.225
                ]
            })
        ]
    )
```

会有一定差异。





建议在pipline最开始首先使用ToTensor，可以简化对应的 torchvison 等效代码，比如配置：

```json
[
  {
    "name": ":for_images:torchvision:ToTensor",
    "paras": {
    }
  },
  {
    "name": ":for_images:torchvision:Resize",
    "paras": {
      "size": 110
    }
  },
  {
    "name": ":for_images:torchvision:CenterCrop",
    "paras": {
      "size": 96
    }
  },
  {
    "name": ":for_images:torchvision:Normalize",
    "paras": {
      "mean": [
        0.485,
        0.456,
        0.406
      ],
      "std": [
        0.229,
        0.224,
        0.225
      ]
    }
  }
]
```

对应于：

```python
    pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=110),
            transforms.CenterCrop(size=96),
            transforms.Normalize(**{
                "mean": [
                    0.485,
                    0.456,
                    0.406
                ],
                "std": [
                    0.229,
                    0.224,
                    0.225
                ]
            })
        ]
    )
```

