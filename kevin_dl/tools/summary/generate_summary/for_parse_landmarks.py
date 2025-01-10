import os
import cv2
import numpy as np
from kevin_toolbox.data_flow.file import markdown, json_
import kevin_toolbox.nested_dict_list as ndl
from kevin_dl.tools.face.utils import plot_bbox_and_landmarks, parse_landmarks
from kevin_dl.tools.face.detect import MTCNN_Detector, SFD_Detector
from kevin_dl.utils.variable import root_dir

data_dir = os.path.join(root_dir, "kevin_dl/tools/face/test/test_data")
ori_image = cv2.cvtColor(cv2.imread(os.path.join(data_dir, "head_pose/raw_face/0.png")), cv2.COLOR_BGR2RGB)
output_dir = os.path.join(root_dir, "kevin_dl/tools/face/summary/parse_landmarks")
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

doc = "# summary of parse_landmarks\n\n"

for Detector, pt_nums in zip([MTCNN_Detector, SFD_Detector], [5, 68]):
    detector = Detector(b_use_gpu=False)

    doc += f"## {pt_nums}关键点\n\n"

    res = detector.detect_face(image=ori_image)[0]
    # 标上关键点和人脸框
    ann_image = plot_bbox_and_landmarks(image=ori_image, landmarks=res["landmarks"], bbox=res["bbox"],
                                        person_id=f'score:{res["score"]:.2f}', b_inplace=False)
    # 标准图
    scale = 1.2
    ann_image_2 = np.ones(shape=[j if i == 2 else int(j * scale) for i, j in enumerate(ori_image.shape)],
                          dtype=ori_image.dtype) * 255
    ann_image_2 = plot_bbox_and_landmarks(image=ann_image_2, landmarks=res["landmarks"] * scale, b_inplace=True)
    # 标上部位
    part_s = parse_landmarks(landmarks=res["landmarks"], output_contents=None)
    ann_image_3 = plot_bbox_and_landmarks(image=ori_image, landmarks=np.asarray(list(part_s.values())),
                                          landmarks_names=list(part_s.keys()), b_inplace=False)
    # 标准图
    ann_image_4 = np.ones_like(ann_image_2) * 255
    ann_image_4 = plot_bbox_and_landmarks(image=ann_image_4, landmarks=np.asarray(list(part_s.values())) * scale,
                                          landmarks_names=list(part_s.keys()), b_inplace=True)
    #
    table_s = {
        "": ["原图", "带关键点和人脸框", "关键点标准图", "带部位坐标", "部位坐标标准图"],
        "image": [ori_image, ann_image, ann_image_2, ann_image_3, ann_image_4],
    }
    table_s = markdown.save_images_in_ndl(var=table_s,
                                          setting_s={":image": dict(b_is_rgb=True, saved_image_format=".jpg")},
                                          plot_dir=os.path.join(plot_dir, f"{pt_nums}"),
                                          doc_dir=output_dir)
    doc += markdown.generate_table(content_s=table_s, orientation="h") + "\n\n"

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(doc)
