import os
import cv2
import numpy as np
from kevin_toolbox.data_flow.file import markdown
import kevin_toolbox.nested_dict_list as ndl
from kevin_dl.tools.face.utils import plot_bbox_and_landmarks, rotate_image
from kevin_dl.tools.face.detect import MTCNN_Detector, SFD_Detector
from kevin_dl.utils.variable import root_dir

data_dir = os.path.join(root_dir, "kevin_dl/tools/face/test/test_data")

for Detector in [MTCNN_Detector, SFD_Detector]:
    output_dir = os.path.join(root_dir, "kevin_dl/tools/face/summary/detect", Detector.__name__)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    detector = Detector(b_use_gpu=False)

    doc = f"# summary of {Detector.__name__}\n\n"

    # -------------------------- detect face -------------------------- #
    doc += "## detect_face()\n\n"

    doc += "检出图片中的人脸框和关键点。\n\n"

    # ----------------------------- #
    doc += "### 单人脸（关键点示意图）\n\n"
    ori_image = cv2.cvtColor(cv2.imread(os.path.join(data_dir, "head_pose/raw_face/0.png")),
                             cv2.COLOR_BGR2RGB)
    res = detector.detect_face(image=ori_image)[0]
    # 标上关键点和人脸框
    ann_image = plot_bbox_and_landmarks(image=ori_image, landmarks=res["landmarks"], bbox=res["bbox"],
                                        person_id=f'score:{res["score"]:.2f}', b_inplace=False)
    # 标准图
    scale = 1.8
    ann_image_2 = np.ones(shape=[j if i == 2 else int(j * scale) for i, j in enumerate(ori_image.shape)],
                          dtype=ori_image.dtype) * 255
    ann_image_2 = plot_bbox_and_landmarks(image=ann_image_2, landmarks=res["landmarks"] * scale, b_inplace=True)
    #
    table_s = {
        "": ["原图", "带关键点和人脸框", "关键点标准图"],
        "image": [ori_image, ann_image, ann_image_2],
    }
    table_s = markdown.save_images_in_ndl(var=table_s,
                                          setting_s={":image": dict(b_is_rgb=True, saved_image_format=".jpg")},
                                          plot_dir=os.path.join(plot_dir, "detect_face"),
                                          doc_dir=output_dir)
    doc += markdown.generate_table(content_s=table_s, orientation="h") + "\n\n"

    # ----------------------------- #
    doc += "### 多人脸\n\n"
    # 读取图片
    ori_image = cv2.cvtColor(cv2.imread(os.path.join(data_dir, "good_dog.jpg")), cv2.COLOR_BGR2RGB)
    # 标上关键点和人脸框
    ann_image = ori_image.copy()
    for i, res in enumerate(detector.detect_face(image=ori_image)):
        ann_image = plot_bbox_and_landmarks(image=ann_image, landmarks=res["landmarks"], bbox=res["bbox"],
                                            person_id=f'face:{i}', b_inplace=True)
    #
    table_s = {
        "": ["原图", "带关键点和人脸框", ],
        "image": [ori_image, ann_image, ],
    }
    table_s = markdown.save_images_in_ndl(var=table_s,
                                          setting_s={":image": dict(b_is_rgb=True, saved_image_format=".jpg")},
                                          plot_dir=os.path.join(plot_dir, "detect_face_multi_faces"),
                                          doc_dir=output_dir)
    doc += markdown.generate_table(content_s=table_s, orientation="h") + "\n\n"

    # ----------------------------- #
    doc += "### 不同 head pose 下的人脸检测\n\n"
    doc += "人头模型数据来自：Head scan 13 (photogrammetry) by yaro.pro on Sketchfab，遵守 CC BY 4.0 DEED 协议。\n\n"
    doc += f'<div class="sketchfab-embed-wrapper"> <iframe title="Head scan 13 (photogrammetry)" frameborder="0" ' \
           f'allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; ' \
           f'xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered ' \
           f'web-share src="https://sketchfab.com/models/5e6d2804405449e6b3bd96cd12d8b1ab/embed"> ' \
           f'</iframe> <p style="font-size: 13px; font-weight: normal; margin: 5px; color: #4A4A4A;"> ' \
           f'<a href="https://sketchfab.com/3d-models/head-scan-13-photogrammetry-5e6d2804405449e6b3bd96cd12d8b1ab?u' \
           f'tm_medium=embed&utm_campaign=share-popup&utm_content=5e6d2804405449e6b3bd96cd12d8b1ab" target="_blank" re' \
           f'l="nofollow" style="font-weight: bold; color: #1CAAD9;"> Head scan 13 (photogrammetry) </a> by <a href="htt' \
           f'ps://sketchfab.com/yaro.pro?utm_medium=embed&utm_campaign=share-popup&utm_content=5e6d2804405449e6b3bd96cd1' \
           f'2d8b1ab" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;"> yaro.pro </a> on <a hr' \
           f'ef="https://sketchfab.com?utm_medium=embed&utm_campaign=share-popup&utm_content=5e6d2804405449e6b3bd96cd1' \
           f'2d8b1ab" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;">Sketchfab</a></p></div>\n\n'

    for head_pose in ["pitch", "roll", "yaw"]:
        doc += f'#### {head_pose}\n\n'
        table_s = {
            "pose": [],
            "raw_image": [],
            "detect_image": [],
            "detect_res": [],
        }
        for file in sorted(os.listdir(os.path.join(data_dir, "head_pose", head_pose)),
                           key=lambda x: int(x.split(".")[0])):
            table_s["pose"].append(int(file.split(".")[0]))
            # 读取图片
            ori_image = cv2.cvtColor(cv2.imread(os.path.join(data_dir, "head_pose", head_pose, file)),
                                     cv2.COLOR_BGR2RGB)
            table_s["raw_image"].append(ori_image)
            #
            res_ls = detector.detect_face(image=ori_image)
            if len(res_ls) == 0:
                table_s["detect_res"].append(None)
                table_s["detect_image"].append(None)
                continue
            # 标上关键点和人脸框
            ann_image = ori_image.copy()
            for i, res in enumerate(res_ls):
                ann_image = plot_bbox_and_landmarks(image=ann_image, landmarks=res["landmarks"], bbox=res["bbox"],
                                                    person_id=f'face:{i}', b_inplace=True)
            table_s["detect_image"].append(ann_image)
            table_s["detect_res"].append(ndl.traverse(var=res_ls, match_cond=lambda _, __, x: isinstance(x, np.ndarray),
                                                      action_mode="replace", converter=lambda _, x: x.tolist() if len(
                    x) < 8 else f'{x[:8].tolist()}'.replace("]", "... ]")))

        table_s = markdown.save_images_in_ndl(var=table_s,
                                              setting_s={":raw_image": dict(b_is_rgb=True, saved_image_format=".jpg"),
                                                         ":detect_image": dict(b_is_rgb=True,
                                                                               saved_image_format=".jpg")},
                                              plot_dir=os.path.join(plot_dir, "detect_face_at_diff_pose", head_pose),
                                              doc_dir=output_dir)
        doc += markdown.generate_table(content_s=table_s, orientation="h") + "\n\n"

    # -------------------------- find_best_rotate_angle -------------------------- #
    doc += "## find_best_rotate_angle()\n\n"
    doc += "选取检出的分数最高的人脸，计算出要将图片旋转多少角度，才能让该人脸竖直。\n\n"
    table_s = {
        "raw_image": [],
        "detect_image": [],
        "rotate_angle": [],
        "rotated_image": []
    }
    for file in os.listdir(os.path.join(data_dir, "rotate")):
        ori_image = cv2.cvtColor(cv2.imread(os.path.join(data_dir, "rotate", file)), cv2.COLOR_BGR2RGB)
        table_s["raw_image"].append(ori_image)
        res_ls = detector.detect_face(image=ori_image)
        ann_image = ori_image.copy()
        for i, res in enumerate(res_ls):
            ann_image = plot_bbox_and_landmarks(image=ann_image, landmarks=res["landmarks"], bbox=res["bbox"],
                                                person_id=f'face:{i};score:{res["score"]}', b_inplace=True)
        table_s["detect_image"].append(ann_image)
        #
        rotate_angle = detector.find_best_rotate_angle(image=ori_image)
        table_s["rotate_angle"].append(rotate_angle)
        #
        table_s["rotated_image"].append(rotate_image(image=ann_image, angle=rotate_angle))
    table_s = markdown.save_images_in_ndl(table_s=table_s,
                                          setting_s={f":{k}": dict(b_is_rgb=True, saved_image_format=".jpg") for k in
                                                     ["raw_image", "detect_image", "rotated_image"]},
                                          plot_dir=os.path.join(plot_dir, "find_best_rotate_angle", head_pose),
                                          doc_dir=output_dir)
    doc += markdown.generate_table(content_s=table_s, orientation="h") + "\n\n"

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(doc)
