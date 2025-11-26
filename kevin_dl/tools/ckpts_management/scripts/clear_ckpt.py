from kevin_dl.tools.ckpts_management import Ckpt_Manager
import argparse

# 参数
out_parser = argparse.ArgumentParser(description="")
out_parser.add_argument("--input_dir", type=str, required=True)
out_parser.add_argument("--bak_dir", type=str, required=False, default=None)
out_parser.add_argument("--b_moved_to_bak", type=lambda x: bool(eval(x)), required=False, default=True)
out_parser.add_argument("--b_dry_run", type=lambda x: bool(eval(x)), required=False, default=False)
args = out_parser.parse_args().__dict__
print(args)

ckpt_manager = Ckpt_Manager(input_dir=args["input_dir"], bak_dir=args["bak_dir"])
if args["b_moved_to_bak"]:
    ckpt_manager.add_task(match_cond=".*", task_type="move_to_bak")
else:
    ckpt_manager.add_task(match_cond=".*", task_type="remove")
ckpt_manager.run(b_dry_run=args["b_dry_run"])
