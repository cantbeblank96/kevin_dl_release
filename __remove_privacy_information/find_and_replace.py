import os
from kevin_toolbox.patches.for_os import find_files_in_dir
from kevin_toolbox.patches.for_logging import build_logger
from kevin_toolbox.data_flow.file import json_
import re


def find_matches(root_dir, target, suffix_ls, b_use_regex):
    results = []  # (relative_path, line_no, line_text)

    if b_use_regex:
        target = re.compile(target)

    for file_path in find_files_in_dir(input_dir=root_dir, suffix_ls=suffix_ls, b_relative_path=True,
                                       b_ignore_case=True):
        with open(os.path.join(root_dir, file_path), "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, start=1):
                if b_use_regex:
                    if target.search(line):
                        results.append((file_path, idx, line.rstrip("\n")))
                else:
                    if target in line:
                        results.append((file_path, idx, line.rstrip("\n")))
    return results


def print_table(results):
    logger.info("\n=== 搜索结果 ===")
    logger.info(f"{'编号':<6}{'文件路径':<40}{'行号':<8}{'内容'}")
    logger.info("-" * 120)
    for i, (path, lineno, text) in enumerate(results):
        logger.info(f"{i:<6}{path:<40}{lineno:<8}{text}")
    logger.info("-" * 120)
    logger.info(f"共计 {len(results)} 处匹配\n")


def ask_user_filter(results):
    if not results:
        logger.info("未找到任何匹配。")
        return []

    logger.info("是否需要忽略部分结果？")
    logger.info("输入以下选项之一：")
    logger.info("  a) 输入要 **忽略的编号**，用逗号分隔，例如: 1,3,5")
    logger.info("  b) 输入 'rev' 进入 ‘反选模式’，只保留你输入的编号，其余全部忽略")
    logger.info("  c) 直接回车 = 不忽略任何内容")

    user_input = logger_input("\n请输入: ").strip()

    if user_input == "":
        return results  # 全部保留

    if user_input.lower() == "rev":
        keep = logger_input("请输入要保留的编号 (逗号分隔): ").strip()
        keep_ids = set(map(int, keep.split(",")))
        return [r for i, r in enumerate(results) if i in keep_ids]

    # 普通忽略模式
    ignore_ids = set(map(int, user_input.split(",")))
    return [r for i, r in enumerate(results) if i not in ignore_ids]


def apply_replacements(root_dir, results, target, replacement, b_use_regex):
    """对结果中的文件执行替换"""
    if b_use_regex:
        target = re.compile(target)
    # 按文件聚合
    file_map = {}
    for rel_path, lineno, _ in results:
        file_map.setdefault(rel_path, []).append(lineno)

    for rel_path, line_numbers in file_map.items():
        full_path = os.path.join(root_dir, rel_path)
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        line_numbers = set(line_numbers)
        # 执行逐行替换
        if b_use_regex:
            for i in range(len(lines)):
                if (i + 1) in line_numbers:
                    lines[i] = target.sub(replacement, lines[i])
        else:
            for i in range(len(lines)):
                if (i + 1) in line_numbers:
                    lines[i] = lines[i].replace(target, replacement)

        with open(full_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    logger.info("\n替换完成！")


def main(root, target, replacement, suffix_ls, b_use_regex):
    logger.info(f"正在搜索目录: {root}")
    matches = find_matches(root, target, suffix_ls, b_use_regex)

    print_table(matches)

    filtered = ask_user_filter(matches)

    if not filtered:
        logger.info("没有需要替换的部分，退出。")
        return

    logger.info("\n将执行替换以下内容：")
    print_table(filtered)

    confirm = logger_input("确认执行替换吗？(y/n): ").strip().lower()
    if confirm == "y":
        apply_replacements(root, filtered, target, replacement, b_use_regex)
    else:
        logger.info("已取消。")


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    # import logging

    # 参数
    out_parser = argparse.ArgumentParser(description="")
    out_parser.add_argument("--input_dir", type=str, required=True)
    out_parser.add_argument("--record_dir", type=str, required=False)
    out_parser.add_argument("--target", type=str, required=True)
    out_parser.add_argument("--replacement", type=str, required=True)
    out_parser.add_argument("--suffix_ls", nargs='+', type=str, required=True)
    out_parser.add_argument("--b_use_regex", type=lambda x: x.lower() == "true", required=False, default=False)
    args = out_parser.parse_args().__dict__
    print(args)
    record_dir = os.path.join(args["record_dir"], datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) if args[
        "record_dir"] else None
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        json_.write(content=args, file_path=os.path.join(record_dir, "args.json"))

    #
    handler_ls = [dict(target=None, level="INFO", formatter="%(message)s"), ]
    if record_dir:
        handler_ls.append(dict(target=os.path.join(record_dir, f'log.txt'), level="DEBUG"))
    logger = build_logger(
        name=":global",
        handler_ls=handler_ls,
        registry=None
    )


    def logger_input(x):
        logger.info(x)
        res = input(x)
        logger.info(f'got: {res}')
        return res


    main(root=args["input_dir"], target=args["target"], replacement=args["replacement"], suffix_ls=args["suffix_ls"],
         b_use_regex=args["b_use_regex"])
