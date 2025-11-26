from io import StringIO
from kevin_toolbox.data_flow.file import json_
from kevin_dl.utils.ceph import read_image, read_file, set_default_client

set_default_client(cfg_path="~/petreloss.conf")


def test_read_image():
    img_url = "aoss_sco_b:s3://face-id/temp/image_158.jpg"
    image = read_image(file_path=img_url)
    print(image.shape)


def test_read_file_0():
    file_url = "aoss_sco_b:s3://face-id/temp/hello.json"
    content = read_file(file_path=file_url, b_ignore_error=True)
    assert content is not None
    file_obj = StringIO(initial_value=content)
    res = json_.read(file_obj=file_obj, b_use_suggested_converter=True)
    print(res)


def test_read_file_1():
    file_url = "aoss_sco_b:s3://face-id/temp/hello.txt"
    content = read_file(file_path=file_url)
    ann = [i.strip() for i in content.strip().split('\n', -1)]
    print(len(ann), ann[0])
