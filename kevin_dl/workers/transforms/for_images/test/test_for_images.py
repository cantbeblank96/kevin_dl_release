import os
from collections import OrderedDict
import torch
import numpy as np
from PIL import Image
from kevin_toolbox.patches.for_test import check_consistency
from kevin_toolbox.data_flow.file import markdown
import kevin_toolbox.nested_dict_list as ndl
from kevin_dl.workers.transforms.for_images.utils import get_format, Image_Format, convert_format
from generate_briefs import generate_brief_of_motion_blur, generate_brief_of_gaussian_blur, \
    generate_brief_of_brightness_shift

temp_folder = os.path.join(os.path.dirname(__file__), "temp")
summary_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "summary")
data_folder = os.path.join(os.path.dirname(__file__), "test_data", "data_0")


def test_convert_format():
    print("test utils.convert_format()")

    # begin with pil image
    pil_image_0 = Image.open(os.path.join(data_folder, "ILSVRC2012_val_00040001.JPEG"))

    # pil image ==> torch tensor
    tensor_0 = convert_format(image=pil_image_0, output_format=Image_Format.TORCH_TENSOR)
    check_consistency(Image_Format.TORCH_TENSOR, get_format(image=tensor_0))

    # torch tensor ==> np array
    np_array_0 = convert_format(image=tensor_0, output_format=Image_Format.NP_ARRAY)
    check_consistency(Image_Format.NP_ARRAY, get_format(image=np_array_0))

    # np array ==> pil image
    pil_image_1 = convert_format(image=np_array_0, output_format=Image_Format.PIL_IMAGE)
    check_consistency(Image_Format.PIL_IMAGE, get_format(image=pil_image_1))

    # pil image ==> np array
    np_array_1 = convert_format(image=pil_image_1, output_format=Image_Format.NP_ARRAY)
    check_consistency(Image_Format.NP_ARRAY, get_format(image=np_array_1))

    # np array ==> torch tensor
    tensor_1 = convert_format(image=np_array_1, output_format=Image_Format.TORCH_TENSOR)
    check_consistency(Image_Format.TORCH_TENSOR, get_format(image=tensor_1))

    # torch tensor ==> pil image
    pil_image_2 = convert_format(image=tensor_1, output_format=Image_Format.PIL_IMAGE)
    check_consistency(Image_Format.PIL_IMAGE, get_format(image=pil_image_2))

    # 检验一致性
    check_consistency(tensor_0, tensor_1)
    check_consistency(np_array_0, np_array_1)
    check_consistency(np.array(pil_image_0), np.array(pil_image_1), np.array(pil_image_2))
    check_consistency(np.transpose(tensor_0.cpu().numpy(), (1, 2, 0)), np_array_0, np.array(pil_image_0))


def test_for_images():
    global summary_folder
    print("test blur, color in for_images")

    brief_s = OrderedDict({
        "blur": {
            "motion_blur": generate_brief_of_motion_blur(
                output_dir=os.path.join(summary_folder, "blur", "motion_blur"),
                example_image_path=os.path.join(data_folder, "ILSVRC2012_val_00040001.JPEG")
            ),
            "gaussian_blur": generate_brief_of_gaussian_blur(
                output_dir=os.path.join(summary_folder, "blur", "gaussian_blur"),
                example_image_path=os.path.join(data_folder, "ILSVRC2012_val_00040001.JPEG")
            )
        },
        "color": {
            "brightness_shift": generate_brief_of_brightness_shift(
                output_dir=os.path.join(summary_folder, "color", "brightness_shift"),
                example_image_path=os.path.join(data_folder, "ILSVRC2012_val_00040001.JPEG")
            )
        }
    })

    for n, v in ndl.get_nodes(var=brief_s, level=-1, b_strict=True):
        ndl.set_value(
            var=brief_s, name=n,
            value=markdown.generate_link(name="brief.md", target=os.path.relpath(v, start=summary_folder), type_="url"),
            b_force=False
        )

    with open(os.path.join(summary_folder, "README.md"), "w") as f:
        doc = ""
        doc += "# Summaries of transforms.images\n\n"
        doc += markdown.generate_list(var=brief_s)
        f.write(doc)
