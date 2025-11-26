import os
import shutil
from kevin_toolbox.patches.for_os import pack, remove
from kevin_toolbox.data_flow.file import json_
from kevin_dl.tools.ckpts_management import Ckpt_Manager

if __name__ == '__main__':
    import argparse

    # 参数
    out_parser = argparse.ArgumentParser(description="")
    out_parser.add_argument("--input_dir", type=str, required=True)
    out_parser.add_argument("--b_pack", type=lambda x: bool(eval(x)), required=False, default=False)
    args = out_parser.parse_args().__dict__
    print(args)

    # ---------------------- #

    exp_dir_ls = []
    for root, dirs, files in os.walk(args["input_dir"]):
        for dir_ in dirs:
            if dir_.startswith("study_"):
                exp_dir_ls.append(os.path.dirname(os.path.join(root, dir_)))

    for exp_dir in exp_dir_ls:
        print(f'dealing {exp_dir}')
        if args["b_pack"]:
            temp_dir = os.path.join(exp_dir, "other_non-optimal_experiments")
            os.makedirs(temp_dir, exist_ok=True)
            trial_idx = json_.read(file_path=os.path.join(exp_dir, "study_0/var.json"),
                                   b_use_suggested_converter=True)["best_trial"]["number"]
            for folder in os.listdir(exp_dir):
                if folder.isdigit() and int(folder) != trial_idx:
                    ckpt_manager = Ckpt_Manager(input_dir=os.path.join(exp_dir, folder))
                    ckpt_manager.add_task(match_cond=".*", task_type="remove")
                    ckpt_manager.run()
                    shutil.move(src=os.path.join(exp_dir, folder), dst=temp_dir)

            pack(source=temp_dir, target=temp_dir + ".tar")
            remove(temp_dir, ignore_errors=True)
        else:
            trial_idx = json_.read(file_path=os.path.join(exp_dir, "study_0/var.json"),
                                   b_use_suggested_converter=True)["best_trial"]["number"]
            for folder in os.listdir(exp_dir):
                if folder.isdigit() and int(folder) != trial_idx:
                    ckpt_manager = Ckpt_Manager(input_dir=os.path.join(exp_dir, folder))
                    ckpt_manager.add_task(match_cond=".*", task_type="remove")
                    ckpt_manager.run()
