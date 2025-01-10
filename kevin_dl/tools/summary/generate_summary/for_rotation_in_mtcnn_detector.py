import os
import cv2
import numpy as np
from kevin_toolbox.data_flow.file import markdown, json_
from kevin_dl.tools.face.detect import MTCNN_Detector
from kevin_dl.tools.face.utils import rotate_image
from kevin_dl.utils.variable import root_dir

data_dir = os.path.join(root_dir, "kevin_dl/tools/face/test/test_data/rotate")
output_dir = os.path.join(root_dir, "kevin_dl/tools/face/summary/detect",
                          f'rotation_calculation_in_{MTCNN_Detector.__name__}')
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

doc = "# summary of rotation calculation in mtcnn_detector\n\n"
doc += f"分析 {MTCNN_Detector.__name__} 中的 find_best_rotate_angle() 是如何通过调用 " \
       f"evaluate_scores_at_diff_rotate_angles()计算出旋转角度的。\n\n"
doc += "## 函数介绍\n\n"
doc += "find_best_rotate_angle():\n\n"
doc += f'```python\nfind_best_rotate_angle():\n{MTCNN_Detector.find_best_rotate_angle.__doc__}\n```\n\n'
doc += "evaluate_scores_at_diff_rotate_angles():\n\n"
doc += f'```python\nrotate_image():\n{MTCNN_Detector.evaluate_scores_at_diff_rotate_angles.__doc__}\n```\n\n'

doc += "## 计算流程\n\n"

doc += "下面演示 find_best_rotate_angle() 内部是如何多次调用 evaluate_scores_at_diff_rotate_angles() 迭代获取最佳旋转角度。分为以下几步：" + markdown.generate_list(
    var=[
        r"取原图0°、旋转±45°、±90°、±135°和180°下检测分数最大的角度，记为$$\alpha_{0}$$。",
        r"以$$\alpha_{0}$$为基础，以15°为步长，在$$\alpha_{0}-45$$和$$\alpha_{0}+45$$之间检测，同样取分数最大的角度，记为$$\alpha_{1}$$。"
        r"（若$$\alpha_{0}$$为空，亦即上一步无法检测出人脸，则令$$\alpha_{0}$$依次等于上面的四个角度，分别尝试得出最佳的$$\alpha_{1}$$。）",
        r"以$$\alpha_{1}$$为基础，以5°为步长，在$$\alpha_{0}-15$$和$$\alpha_{0}+15$$之间检测，同样取分数最大的角度，记为$$\alpha_{2}$$。",
        r"以$$\alpha_{2}$$为基础，以1°为步长，在$$\alpha_{0}-5$$和$$\alpha_{0}+5$$之间检测，使用 kernel=5，sigma=1 的高斯卷积核对分数进行平滑，"
        r"同样取分数最大的角度（精确到0.1°），记为$$\alpha_{3}$$。即为最佳角度。"
    ])

detector = MTCNN_Detector(b_use_gpu=False)
for file in os.listdir(data_dir):
    image_path = os.path.join(data_dir, file)

    ori_image = cv2.imread(image_path)

    doc += f'input image:\n\n'
    out_file = os.path.join(plot_dir, "input_image", os.path.basename(image_path))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    cv2.imwrite(out_file, ori_image)
    doc += markdown.generate_link(name=os.path.basename(out_file),
                                  target=os.path.relpath(out_file, output_dir), type_="image") + "\n\n"

    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

    #
    table_s = {
        "paras": [],
        "outputs": [],
        "rotated_image": []
    }
    #
    known_res_s = dict()
    base_angles = None
    for range_, step, b_get_fine_angle in [
        ((-180, 180), 45, False),
        ((-45, 45), 15, False),
        ((-15, 15), 5, False),
        ((-5, 5), 1, True)
    ]:
        table_s["paras"].append(
            dict(base_angles=base_angles, range_=range_, step=step, b_get_fine_angle=b_get_fine_angle,
                 known_res_s=f'...;len(known_res_s) is {len(known_res_s)}'))
        #
        best_angle, res_s = detector.evaluate_scores_at_diff_rotate_angles(
            image=ori_image, base_angles=base_angles, step=step, range_=range_, decimals=1,
            b_get_fine_angle=b_get_fine_angle, known_res_s=known_res_s
        )
        base_angles = np.arange(*range_, step) if best_angle is None else [best_angle]
        known_res_s.update(res_s)
        #
        table_s["outputs"].append(
            dict(best_angle=best_angle, res_s=f'...;len(res_s) is {len(res_s)}'))
        if best_angle is not None:
            table_s["rotated_image"].append(rotate_image(image=ori_image, angle=best_angle))
        else:
            table_s["rotated_image"].append(None)
    #
    table_s = markdown.save_images_in_ndl(
        table_s=table_s, setting_s={":image": dict(b_is_rgb=True, saved_image_format=".jpg")},
        plot_dir=os.path.join(plot_dir, f'rotate_of_{os.path.basename(image_path)}'),
        doc_dir=output_dir)
    doc += markdown.generate_table(content_s=table_s, orientation="h") + "\n\n"

with open(os.path.join(output_dir, "README.md"), "w") as f:
    f.write(doc)
