import os
import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
from kevin_toolbox.data_flow.file import markdown
import kevin_toolbox.nested_dict_list as ndl
from kevin_dl.workers.transforms.for_images.blur import Motion_Blur, Gaussian_Blur
from kevin_dl.workers.transforms.for_images.color import Brightness_Shift
from kevin_dl.workers.transforms.for_images.utils import convert_format, Image_Format


def _2d_table(shared_paras, para_range_s, output_dir, func, input_s):
    p_names = list(para_range_s.keys())
    p_values_ls = list(para_range_s.values())
    doc = f"for diff {p_names[0]} and {p_names[1]}:\n\n"
    doc += markdown.generate_list(var=dict(shared_paras=shared_paras))
    doc += "\n"

    table_s = OrderedDict()
    table_s[f"{p_names[0]}\\{p_names[1]}"] = p_values_ls[1]
    for p_0 in p_values_ls[0]:
        table_s[p_0] = []
        for p_1 in p_values_ls[1]:
            output = func(**shared_paras, **{p_names[0]: p_0, p_names[1]: p_1})(
                ndl.copy_(var=input_s, b_deepcopy=True))["image"]
            title = f'{p_names[0]}-{p_0}-{p_names[1]}-{p_1}'
            output_path = os.path.join(output_dir, "images", f'{title}.png')
            convert_format(image=output, output_format=Image_Format.PIL_IMAGE).save(output_path)
            table_s[p_0].append(
                markdown.generate_link(name=title, target=os.path.relpath(output_path, start=output_dir),
                                       type_="image"))
    doc += markdown.generate_table(content_s=table_s, orientation="h")
    doc += "\n"
    return doc


def _1d_table(shared_paras, para_range_s, output_dir, func, input_s):
    p_name = list(para_range_s.keys())[0]
    p_values = list(para_range_s.values())[0]
    doc = f"for diff {p_name}:\n\n"
    doc += markdown.generate_list(var=dict(shared_paras=shared_paras))
    doc += "\n"

    table_s = OrderedDict()
    table_s[p_name] = p_values
    table_s["image"] = []
    for v in p_values:
        output = func(**shared_paras, **{p_name: v})(ndl.copy_(var=input_s, b_deepcopy=True))["image"]
        title = f'{p_name}-{v}'
        output_path = os.path.join(output_dir, "images", f'{title}.png')
        convert_format(image=output, output_format=Image_Format.PIL_IMAGE).save(output_path)
        table_s["image"].append(
            markdown.generate_link(name=title, target=os.path.relpath(output_path, start=output_dir),
                                   type_="image"))
    doc += markdown.generate_table(content_s=table_s, orientation="h")
    doc += "\n"
    return doc


def _head(func):
    doc = f"# {func.__name__}()\n\n"

    doc += "## introduction\n\n"
    doc += f'```python\n{func.__doc__}\n```\n\n'

    return doc


def _examples(raw_image_path, output_dir):
    doc = "## examples\n\n"
    doc += "raw image:\n\n"
    doc += markdown.generate_link(name="raw_image", target=os.path.relpath(raw_image_path, start=output_dir),
                                  type_="image")
    doc += "\n"

    return doc


def generate_brief_of_brightness_shift(output_dir, example_image_path):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    doc = _head(func=Brightness_Shift)
    doc += _examples(raw_image_path=example_image_path, output_dir=output_dir)
    image = Image.open(example_image_path)

    doc += _1d_table(shared_paras=dict(keep_ratio=1.0, alpha=0), func=Brightness_Shift,
                     output_dir=output_dir, input_s=dict(image=image),
                     para_range_s=dict(beta=[round(i, 2) for i in np.linspace(-1, 1, 7)]))

    doc += _2d_table(shared_paras=dict(beta=0), func=Brightness_Shift, output_dir=output_dir,
                     input_s=dict(image=image),
                     para_range_s=dict(alpha=[round(i, 2) for i in np.linspace(0, 1, 5)],
                                       keep_ratio=[round(i, 2) for i in np.linspace(0, 1, 5)]))

    with open(os.path.join(output_dir, "brief.md"), "w") as f:
        f.write(doc)

    return os.path.join(output_dir, "brief.md")


def generate_brief_of_motion_blur(output_dir, example_image_path):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    doc = _head(func=Motion_Blur)
    doc += _examples(raw_image_path=example_image_path, output_dir=output_dir)
    image = Image.open(example_image_path)

    doc += _2d_table(shared_paras=dict(),
                     func=Motion_Blur, output_dir=output_dir, input_s=dict(image=image),
                     para_range_s=dict(kernel_size=[round(i, 0) for i in np.linspace(0, 30, 4)],
                                       angle=[round(i, 0) for i in np.linspace(0, 180, 5)][:-1]))

    with open(os.path.join(output_dir, "brief.md"), "w") as f:
        f.write(doc)

    return os.path.join(output_dir, "brief.md")


def generate_brief_of_gaussian_blur(output_dir, example_image_path):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    doc = _head(func=Gaussian_Blur)
    doc += _examples(raw_image_path=example_image_path, output_dir=output_dir)
    image = Image.open(example_image_path)

    doc += _1d_table(shared_paras=dict(kernel_size=None), func=Gaussian_Blur, output_dir=output_dir,
                     input_s=dict(image=image),
                     para_range_s=dict(sigma=[round(i, 0) for i in np.linspace(0, 8, 5)]))

    doc += _2d_table(shared_paras=dict(), func=Gaussian_Blur, output_dir=output_dir, input_s=dict(image=image),
                     para_range_s=dict(kernel_size=[int(i) // 2 * 2 + 1 for i in np.linspace(0, 30, 5)][1:],
                                       sigma=[round(i, 0) for i in np.linspace(0, 8, 5)]))

    with open(os.path.join(output_dir, "brief.md"), "w") as f:
        f.write(doc)

    return os.path.join(output_dir, "brief.md")
