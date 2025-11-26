import os
from PIL import Image
from kevin_toolbox.patches.for_test import check_consistency
from kevin_dl.workers.transforms.for_images.utils import get_format, Image_Format, convert_format
from kevin_dl.workers.variable import TRANSFORMS
from kevin_toolbox.data_flow.file import json_

temp_folder = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(temp_folder, exist_ok=True)
data_folder = os.path.join(os.path.dirname(__file__), "test_data", "data_1")


def test_by_bbox_affine_trans():
    image = Image.open(os.path.join(data_folder, "Tadokor_Koji_the_Japanese_representative.jpg"))
    ann_s = json_.read(file_path=os.path.join(data_folder, "ann_s_for_Tadokor_Koji_the_Japanese_representative.json"),
                       b_use_suggested_converter=True)

    output_s = TRANSFORMS.get(name=":for_images:face:Align_and_Crop_Face")(
        b_include_details=True, method=":face:alignment:by_bbox:affine_trans",
        template="edge_corner", match_pattern="expanded_bbox",
        desired_face_size=108, desired_image_size=1.3
    )(
        input_s=dict(image=image, bbox=ann_s["detect_faces"][0]["bbox"])
    )
    output = convert_format(image=output_s["image"], output_format=Image_Format.PIL_IMAGE)
    output.save(os.path.join(temp_folder, "output_by_bbox_affine_trans.jpg"))


def test_by_landmarks_affine_trans():
    image = Image.open(os.path.join(data_folder, "Tadokor_Koji_the_Japanese_representative.jpg"))
    ann_s = json_.read(file_path=os.path.join(data_folder, "ann_s_for_Tadokor_Koji_the_Japanese_representative.json"),
                       b_use_suggested_converter=True)

    output_s = TRANSFORMS.get(name=":for_images:face:Align_and_Crop_Face")(
        b_include_details=True, method=":face:alignment:by_landmarks:affine_trans",
        template="insightface", match_pattern="eyes_nose_mouth", border_value=0.0,
        desired_face_size=224, desired_image_size=224
    )(
        input_s=dict(image=image, landmarks=ann_s["detect_faces"][0]["landmarks"])
    )
    output = convert_format(image=output_s["image"], output_format=Image_Format.PIL_IMAGE)
    output.save(os.path.join(temp_folder, "output_by_landmarks_affine_trans.jpg"))
