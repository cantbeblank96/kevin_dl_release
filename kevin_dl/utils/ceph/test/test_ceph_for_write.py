import os
import cv2
from kevin_dl.utils.ceph import write_image, write_file, set_default_client, get_default_client
from kevin_dl.utils.variable import root_dir
from kevin_toolbox.data_flow.file import json_

set_default_client(cfg_path="~/petreloss.conf")
default_client = get_default_client()
test_data_dir = os.path.join(root_dir, "kevin_dl/utils/ceph/test/test_data")


def test_write_image():
    # image
    image_path = os.path.join(test_data_dir, "image_158.jpg")
    img_url = "aoss_sco_b:s3://face-id/temp/image_158.jpg"
    try:
        default_client.delete(img_url)
    except:
        pass
    write_image(file_path=img_url, image=cv2.imread(image_path), client=default_client, b_bgr_image=True)
    assert default_client.contains(img_url)


def test_write_file_0():
    # file
    file_url = "aoss_sco_b:s3://face-id/temp/hello.json"
    try:
        default_client.delete(file_url)
    except:
        pass
    content = json_.read(file_path=os.path.join(test_data_dir, "hello.json"),
                         b_use_suggested_converter=True)
    write_file(file_path=file_url,
               content=json_.write(content=content, file_path=None, b_use_suggested_converter=True),
               client=default_client)
    assert default_client.contains(file_url)


def test_write_file_1():
    # file
    file_url = "aoss_sco_b:s3://face-id/temp/hello.txt"
    try:
        default_client.delete(file_url)
    except:
        pass
    with open(os.path.join(test_data_dir, "hello.txt"), "rb") as f:
        content = f.read()
    write_file(file_path=file_url,
               content=content,
               client=default_client)
    assert default_client.contains(file_url)
