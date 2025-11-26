import os
from PIL import Image


def rotate_and_save_image(input_path, output_dir, angles):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开图片
    with Image.open(input_path) as img:
        # 遍历角度列表
        for angle in angles:
            # 旋转图片
            rotated_img = img.rotate(angle, expand=True)

            # 构造输出文件名
            output_filename = f"{angle}.png"
            output_path = os.path.join(output_dir, output_filename)

            # 保存旋转后的图片
            rotated_img.save(output_path)
            print(f"Saved: {output_path}")


# 使用示例
input_image_path = '~/Desktop/gitlab_repos/kevin_dl/kevin_dl/tools/face/test/test_data/head_pose/roll/0.png'  # 替换为你的图片路径
output_directory = os.path.dirname(input_image_path)  # 替换为你想要保存图片的目录

# 角度列表，包括正负30度、正负60度、正负90度、正负120度、正负150度和180度
angles = [-60, 60]

rotate_and_save_image(input_image_path, output_directory, angles)
