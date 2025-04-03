# kevin_dl

一个面向深度学习的工具库



环境要求

```shell
numpy>=1.19
pytorch>=1.2
kevin-toolbox>=1.4.5
```

安装方法：

```shell
pip install kevin-dl  --no-dependencies
```



[项目地址 Repo](https://github.com/cantbeblank96/kevin_dl_release)

[免责声明 Disclaimer](./notes/Disclaimer.md)

[版本更新记录](./notes/Release_Record.md)：

- v 0.0.1 （2025-04-03）【new feature】
  - utils.ceph：新增 ceph 模块，其中包含与 ceph 交互相关的函数
    - download() 人脸转正
    - read_file() 使用 client 读取 file_path 指向的文件内容
    - read_image() 使用 client 读取 file_path 指向的图片。默认以 BGR 顺序读取图片。
    - variable.CLIENTS 注册区，保存已注册的 client。
    - set_client() 将新的 client 添加到注册区。
    - set_default_client() 设定默认的 client。
