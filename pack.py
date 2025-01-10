import os
import argparse
import shutil

"""
使用本脚本进行打包和发布

# 打包
python pack.py
"""

out_parser = argparse.ArgumentParser(description='pack and dist')
out_parser.add_argument('--verbose', type=int, required=False, default=1)
args = out_parser.parse_args().__dict__
assert args["verbose"] in [0, 1]

root_dir = os.path.abspath(os.path.split(__file__)[0])

# 打包
for folder in ["build", "dist", "kevin_toolbox.egg-info"]:
    shutil.rmtree(os.path.join(root_dir, folder), ignore_errors=True)
os.system(f'cd {root_dir};python setup.py sdist bdist_wheel')
