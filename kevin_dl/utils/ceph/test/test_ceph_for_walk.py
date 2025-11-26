import os
from kevin_dl.utils.ceph import walk, set_default_client, get_default_client
from kevin_dl.utils.variable import root_dir

set_default_client(cfg_path="~/petreloss.conf")
default_client = get_default_client()


def test_walk():
    url_ = "aoss_sco_b:s3://face-id/temp/"
    for root, dirs, files in walk(path=url_, client=default_client):
        print(f'under root: {root}')
        for file in files:
            print(f'\tfile: {file}')
        for dir_ in dirs:
            print(f'\tfolder: {dir_}')


def test_walk_local():
    test_data_dir = os.path.join(root_dir, "kevin_dl/utils/ceph/test/test_data")
    for root, dirs, files in walk(path=test_data_dir):
        print(f'under root: {root}')
        for file in files:
            print(f'\tfile: {file}')
        for dir_ in dirs:
            print(f'\tfolder: {dir_}')
