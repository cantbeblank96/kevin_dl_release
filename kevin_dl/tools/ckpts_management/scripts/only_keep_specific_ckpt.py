from kevin_dl.tools.ckpts_management import Ckpt_Manager
import kevin_toolbox.nested_dict_list as ndl

import argparse

# 参数
out_parser = argparse.ArgumentParser(description="")
out_parser.add_argument("--input_dir", type=str, required=True)
out_parser.add_argument("--bak_dir", type=str, required=False, default=None)
out_parser.add_argument("--b_moved_to_bak", type=lambda x: bool(eval(x)), required=False, default=True)
out_parser.add_argument("--b_dry_run", type=lambda x: bool(eval(x)), required=False, default=False)
out_parser.add_argument('--argmin_metric', nargs='+', type=str, help='A list of name', required=False,
                        default=[])
out_parser.add_argument('--argmax_metric', nargs='+', type=str, help='A list of name', required=False,
                        default=[])
out_parser.add_argument('--specific_epoch', nargs='+', type=str, help='A list of integers', required=False,
                        default=[])
args = out_parser.parse_args().__dict__
print(args)

# ---------------------- #


ckpt_manager = Ckpt_Manager(input_dir=args["input_dir"], bak_dir=args["bak_dir"])
if args["b_moved_to_bak"]:
    ckpt_manager.add_task(match_cond=".*", task_type="move_to_bak")
else:
    ckpt_manager.add_task(match_cond=".*", task_type="remove")
for i in args["specific_epoch"]:
    ckpt_manager.add_task(match_cond="^" + i + "$", task_type="recover_from_bak")
for i in args["argmax_metric"]:
    _, _, node_ls = ndl.name_handler.parse_name(name=i)
    ckpt_manager.add_task(match_cond=(f":{node_ls[0]}:epoch", "argmax", i), task_type="recover_from_bak")
for i in args["argmin_metric"]:
    _, _, node_ls = ndl.name_handler.parse_name(name=i)
    ckpt_manager.add_task(match_cond=(f":{node_ls[0]}:epoch", "argmin", i), task_type="recover_from_bak")
ckpt_manager.run(b_dry_run=args["b_dry_run"])
