import os
from kevin_dl.utils.ceph import read_image, read_file, download


def test_read_image():
    img_url = 'openmmlab:s3://openmmlab/datasets/classification/imagenet/train/n09835506/n09835506_15459.JPEG'
    image = read_image(file_path=img_url)
    print(image.shape)


def test_read_file():
    file_url = 'openmmlab:s3://openmmlab/datasets/classification/imagenet/meta/train.txt'
    content = read_file(file_path=file_url)
    ann = [i.strip() for i in content.strip().split('\n', -1)]
    print(len(ann), ann[0])


def test_download():
    img_url = 'openmmlab:s3://openmmlab/datasets/classification/imagenet/train/n09835506/n09835506_15459.JPEG'
    download(file_path=img_url, output_dir=os.path.join(os.path.dirname(__file__), "temp"))
    assert os.path.isfile(os.path.join(os.path.dirname(__file__), "temp", "n09835506_15459.JPEG"))


if __name__ == '__main__':
    test_read_image()
    test_read_file()
    test_download()
