import os
import pytest
import numpy as np
import cv2
from kevin_toolbox.data_flow.file import json_
from kevin_dl.tools.face import alignment, detect

data_dir = os.path.join(os.path.dirname(__file__), "test_data")


# def test_align_and_crop():
#     print("test face.align_and_crop()")
#
#     # 读取关键点
#     ann_s = json_.read(file_path=os.path.join(data_dir, "ann_s_for_Tadokor_Koji_the_Japanese_representative.json"),
#                        b_use_suggested_converter=True)
#     landmarks = ann_s["detect_faces"][0]["landmarks"]
#
#     # 读取图片
#     image = cv2.imread(os.path.join(data_dir, ann_s["image_path"]))
#
#     crop_image = align_and_crop(image=image, landmarks=landmarks, face_size=112, padding_ls=[0, -8, 0, -8])
#     cv2.imwrite(os.path.join(data_dir, "crop_image2.jpg"), crop_image)
#     assert crop_image.shape == (112, 96, 3)


def test_detect():
    print("test face.detect()")

    # 读取标注
    ann_s = json_.read(file_path=os.path.join(data_dir, "ann_s_for_Tadokor_Koji_the_Japanese_representative.json"),
                       b_use_suggested_converter=True)

    # 读取图片
    image = cv2.imread(os.path.join(data_dir, ann_s["image_path"]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = detect.MTCNN_Detector(b_use_gpu=True)
    res = detector.detect_face(image=image)

    # 检验
    assert len(res) == len(ann_s["detect_faces"])
    for it_0, it_1 in zip(ann_s["detect_faces"], res):
        assert np.allclose(it_0['bbox'], it_1['bbox'], atol=2)
        assert np.allclose(it_0['landmarks'], it_1['landmarks'], atol=1)
