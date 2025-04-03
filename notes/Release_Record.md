#### v0.0

- v 0.0.0 （2025-01-10）【new feature】
  - tools
    - 新增 face 模块，其中包含人脸处理相关功能
      - alignment 人脸转正
      - detect 人脸检测
- v 0.0.1 （2025-04-03）【new feature】
  - utils.ceph：新增 ceph 模块，其中包含与 ceph 交互相关的函数
    - download() 人脸转正
    - read_file() 使用 client 读取 file_path 指向的文件内容
    - read_image() 使用 client 读取 file_path 指向的图片。默认以 BGR 顺序读取图片。
    - variable.CLIENTS 注册区，保存已注册的 client。
    - set_client() 将新的 client 添加到注册区。
    - set_default_client() 设定默认的 client。

