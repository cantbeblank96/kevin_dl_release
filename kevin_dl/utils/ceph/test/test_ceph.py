import os
from kevin_dl.utils.ceph import download, set_default_client

set_default_client(cfg_path="~/petreloss.conf")


def test_download():
    img_url = "aoss_sco_b:s3://face-id/temp/image_158.jpg"
    download(file_path=img_url, output_dir=os.path.join(os.path.dirname(__file__), "temp"))
    assert os.path.isfile(os.path.join(os.path.dirname(__file__), "temp", "image_158.jpg"))
