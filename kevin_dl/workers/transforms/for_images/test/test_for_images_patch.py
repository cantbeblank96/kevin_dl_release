import os
from PIL import Image
from kevin_dl.workers.transforms.for_images.utils import Image_Format, convert_format
from kevin_dl.workers.variable import TRANSFORMS

temp_folder = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(temp_folder, exist_ok=True)
data_folder = os.path.join(os.path.dirname(__file__), "test_data", "data_1")


def test_select_and_paste_patch():
    image = Image.open(os.path.join(data_folder, "Tadokor_Koji_the_Japanese_representative.jpg"))

    desired_size = 224
    src_bbox_ls, dst_bbox_ls = [], []
    for i in range(4):
        for j in range(4):
            src_bbox_ls.append([i / 4 + 1 / 8, j / 4 + 1 / 8, desired_size // 4, desired_size // 4])
            dst_bbox_ls.append([i / 4 + 1 / 8, j / 4 + 1 / 8, 1 / 4, 1 / 4])

    output_s = TRANSFORMS.get(name=":for_images:patch:Select_and_Paste_Patch")(
        b_include_details=True, desired_size=desired_size, default_value=0,
        src_bbox_ls=src_bbox_ls, src_bbox_type="center-wh",
        dst_bbox_ls=dst_bbox_ls, dst_bbox_type="center-wh"
    )(
        input_s=dict(image=image)
    )
    output = convert_format(image=output_s["image"], output_format=Image_Format.PIL_IMAGE)
    output.save(os.path.join(temp_folder, "output_select_and_paste_patch.png"))
