import os
import cv2
import numpy as np
from kevin_toolbox.data_flow.file import markdown, json_
from kevin_dl.tools.face import align_and_crop, plot_bbox_and_landmarks, detect

output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "summary")
os.makedirs(output_dir, exist_ok=True)

data_dir = os.path.join(os.path.dirname(__file__), "test_data")

doc = "# summary\n\n"

# -------------------------- detect -------------------------- #
doc += "## detect()\n\n"
doc += "检出图片中的人脸框和关键点。使用 MTCNN。\n\n"

doc += "### 使用示例\n\n"

# 读取图片
ori_image = cv2.imread(os.path.join(data_dir, "good_dog.jpg"))
ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
res_ls = detect(image=ori_image, b_use_gpu=False)

# 标上关键点和人脸框
ann_image = ori_image.copy()
for i, res in enumerate(res_ls):
    ann_image = plot_bbox_and_landmarks(image=ann_image, landmarks=res["landmarks"], bbox=res["bbox"],
                                        person_id=f'face:{i}', b_inplace=True)

table_s = {
    "": ["原图", "带关键点和人脸框", ],
    "image": [ori_image, ann_image, ],
}
table_s = markdown.save_images_in_ndl(var=table_s, plot_dir=os.path.join(output_dir, "plots", "detect"),
                                      doc_dir=output_dir)
doc += markdown.generate_table(content_s=table_s, orientation="h") + "\n\n"

# -------------------------- align_and_crop -------------------------- #

doc += "## align_and_crop()\n\n"
doc += "根据关键点对人脸进行转正，并按照指定的人脸位置和画幅大小进行裁切。\n\n"

doc += "### 使用示例\n\n"
# 读取关键点
ann_s = json_.read(file_path=os.path.join(data_dir, "ann_s_for_Tadokor_Koji_the_Japanese_representative.json"),
                   b_use_suggested_converter=True)
landmarks = ann_s["detect_faces"][0]["landmarks"]
bbox = ann_s["detect_faces"][0]["bbox"]

# 读取图片
ori_image = cv2.imread(os.path.join(data_dir, ann_s["image_path"]))
ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
# 标上关键点和人脸框
ann_image = plot_bbox_and_landmarks(image=ori_image, landmarks=landmarks, bbox=bbox, person_id=f'face:{0}',
                                    b_inplace=False)

table_s = {
    "": ["原图", "带关键点和人脸框", ],
    "image": [ori_image, ann_image, ],
    "paras": ["/", "/", ]
}

# 使用不同参数进行人脸转正
settings_s = {
    "align_and_crop": {
        "face_size": 224
    },
    "using padding_ls 01": {
        "face_size": 112,
        "padding_ls": (0, -8, 0, -8)
    },
    "using padding_ls 02": {
        "face_size": 112,
        "padding_ls": (15, 0, 10, -8)
    },
    "using desired_size": {
        "face_size": 150,
        "desired_size": 224
    },
}

for k, settings in settings_s.items():
    crop_image = align_and_crop(image=ann_image, landmarks=landmarks, **settings)
    table_s[""].append(k)
    table_s["image"].append(crop_image)
    table_s["paras"].append(settings)

table_s["image_shape"] = [tuple(i.shape) for i in table_s["image"]]
table_s = markdown.save_images_in_ndl(var=table_s,
                                      plot_dir=os.path.join(output_dir, "plots", "align_and_crop"),
                                      doc_dir=output_dir)
#
doc += markdown.generate_table(content_s=table_s, orientation="h") + "\n\n"
doc += '注意，为了更好地展示转正、裁剪前后的人脸相对位置，因此在 align_and_crop 中使用的是"带关键点和人脸框"的图片作为输入。\n\n'

# -------------------------- detect and align_and_crop 全流程 -------------------------- #
doc += "## detect() align_and_crop() 串联\n\n"

doc += "### 不同head pose下的人脸检测校正示例\n\n"
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

settings = {
    "face_size": 200,
    "desired_size": 256
}
settings_for_detect = {
    "thresholds": [0.4, 0.6, 0.6],  # Proposal Network (P-Net)、Refine Network (R-Net) 和 Output Network (O-Net) 的阈值
    # p 阈值越低，越容易产生更多的 proposal，跟容易检出大角度人脸
}
doc += f'检测 detect() 所用的参数为：\n\n```\n{json_.write(content=settings_for_detect, file_path=None)}\n```\n\n'
doc += f'人脸转正 align_and_crop() 所用的参数为：\n\n```\n{json_.write(content=settings, file_path=None)}\n```\n\n'

for head_pose in ["pitch", "roll", "yaw"]:
    doc += f'#### {head_pose}\n\n'
    table_s = {
        "pose": [],
        "raw_image": [],
        "detect_image": [],
        "detect_res": [],
        "align_image": []
    }
    for file in sorted(os.listdir(os.path.join(data_dir, "head_pose", head_pose)), key=lambda x: int(x.split(".")[0])):
        table_s["pose"].append(int(file.split(".")[0]))
        # 读取图片
        ori_image = cv2.imread(os.path.join(data_dir, "head_pose", head_pose, file))
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        table_s["raw_image"].append(ori_image)
        #
        res_ls = detect(image=ori_image, b_use_gpu=False, **settings_for_detect)
        if len(res_ls) == 0:
            table_s["detect_res"].append(None)
            table_s["detect_image"].append(None)
            table_s["align_image"].append(None)
            continue
        # 标上关键点和人脸框
        ann_image = ori_image.copy()
        for i, res in enumerate(res_ls):
            ann_image = plot_bbox_and_landmarks(image=ann_image, landmarks=res["landmarks"], bbox=res["bbox"],
                                                person_id=f'face:{i}', b_inplace=True)
        table_s["detect_image"].append(ann_image)
        table_s["detect_res"].append(res_ls)
        # 取分数最高的人脸进行转正
        crop_image = align_and_crop(image=ann_image, landmarks=res_ls[0]["landmarks"], **settings)
        table_s["align_image"].append(crop_image)

    table_s = markdown.save_images_in_ndl(var=table_s,
                                          plot_dir=os.path.join(output_dir, "plots", "detect_and_align_and_crop",
                                                                head_pose),
                                          doc_dir=output_dir)
    doc += markdown.generate_table(content_s=table_s, orientation="h") + "\n\n"

with open(os.path.join(output_dir, "README.md"), "w") as f:
    f.write(doc)
