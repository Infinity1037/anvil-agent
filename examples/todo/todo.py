"""todo.py - 命令行待办事项管理工具（纯标准库，无第三方依赖）。

用法示例:
    python todo.py add "买牛奶"                     # 添加任务
    python todo.py add "写周报" --due 2025-06-01    # 带截止日期
    python todo.py list                             # 查看未完成任务
    python todo.py list --all                       # 查看全部任务
    python todo.py list --done                      # 只查看已完成
    python todo.py done 2                           # 标记 2 号为完成
    python todo.py undo 2                           # 取消完成
    python todo.py rm 2                             # 删除任务
    python todo.py clear --done                     # 清空所有已完成任务
    python todo.py clear                            # 清空全部任务
    python todo.py stats                            # 统计信息
    python todo.py --file my.json add "xxx"         # 使用自定义数据文件

数据默认保存在当前目录的 todo.json。
"""

import argparse
import json
import os
import sys
from datetime import datetime

DEFAULT_FILE = "todo.json"


class TodoList:
    """待办事项集合，负责数据的加载、保存与操作。"""

    def __init__(self, path=DEFAULT_FILE):
        self.path = path
        self.items = []  # 每个元素: {id, text, done, created, due, finished}
        self._next_id = 1
        self.load()

    # ---------- 持久化 ----------
    def load(self):
        """从文件加载数据；文件不存在或损坏时重置为空列表。"""
        if not os.path.exists(self.path):
            self.items = []
            self._next_id = 1
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.items = data.get("items", [])
            self._next_id = data.get(
                "next_id", (max((i["id"] for i in self.items), default=0) + 1))
        except (json.JSONDecodeError, OSError, AttributeError, KeyError):
            self.items = []
            self._next_id = 1

    def save(self):
        """把数据写回文件。"""
        data = {"items": self.items, "next_id": self._next_id}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---------- 操作 ----------
    def add(self, text, due=None):
        """添加一条任务，返回新任务字典。"""
        item = {
            "id": self._next_id,
            "text": text,
            "done": False,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "due": due,
            "finished": None,
        }
        self.items.append(item)
        self._next_id += 1
        self.save()
        return item

    def find(self, task_id):
        """按编号查找任务，找不到返回 None。"""
        for item in self.items:
            if item["id"] == task_id:
                return item
        return None

    def set_done(self, task_id, done=True):
        """设置任务完成状态，返回任务或 None（不存在）。"""
        item = self.find(task_id)
        if item is None:
            return None
        item["done"] = done
        item["finished"] = (datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            if done else None)
        self.save()
        return item

    def remove(self, task_id):
        """删除任务，返回被删除的任务或 None（不存在）。"""
        item = self.find(task_id)
        if item is None:
            return None
        self.items.remove(item)
        self.save()
        return item

    def clear_done(self):
        """删除所有已完成任务，返回删除数量。"""
        before = len(self.items)
        self.items = [i for i in self.items if not i["done"]]
        self.save()
        return before - len(self.items)

    def clear_all(self):
        """清空所有任务，返回删除数量。"""
        n = len(self.items)
        self.items = []
        self._next_id = 1
        self.save()
        return n

    # ---------- 查询 ----------
    def pending(self):
        return [i for i in self.items if not i["done"]]

    def done_items(self):
        return [i for i in self.items if i["done"]]

    def stats(self):
        total = len(self.items)
        done = len(self.done_items())
        return total, done, total - done


def format_item(item):
    """把一条任务格式化成可读文本。"""
    mark = "✓" if item["done"] else " "
    due = f" (截止 {item['due']})" if item.get("due") else ""
    line = f"[{mark}] #{item['id']} {item['text']}{due}"
    if item["done"] and item.get("finished"):
        line += f"  完成于 {item['finished']}"
    return line


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="todo", description="命令行待办事项管理工具")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help=f"数据文件路径（默认 {DEFAULT_FILE}）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="添加任务")
    p_add.add_argument("text", help="任务内容")
    p_add.add_argument("--due", default=None, help="截止日期，如 2025-06-01")

    p_list = sub.add_parser("list", help="列出任务")
    p_list.add_argument("--all", action="store_true", help="显示全部任务")
    p_list.add_argument("--done", action="store_true", help="只显示已完成任务")

    for name, help_ in (("done", "标记任务为完成"),
                        ("undo", "取消完成"),
                        ("rm", "删除任务")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("id", type=int, help="任务编号")

    p_clear = sub.add_parser("clear", help="清空任务")
    p_clear.add_argument("--done", action="store_true", help="只清空已完成任务")

    sub.add_parser("stats", help="统计信息")

    args = parser.parse_args(argv)
    todos = TodoList(args.file)

    if args.command == "add":
        item = todos.add(args.text, args.due)
        print(f"已添加 #{item['id']}: {item['text']}")
    elif args.command == "list":
        if args.done:
            items = todos.done_items()
        elif args.all:
            items = todos.items
        else:
            items = todos.pending()
        if not items:
            hint = ("(空) 没有任务" if args.all or args.done
                    else "(空) 没有未完成任务，试试: python todo.py add \"任务内容\"")
            print(hint)
            return
        for item in items:
            print(format_item(item))
    elif args.command in ("done", "undo"):
        item = todos.set_done(args.id, done=(args.command == "done"))
        if item is None:
            print(f"错误: 找不到编号 {args.id}", file=sys.stderr)
            sys.exit(1)
        state = "已完成" if item["done"] else "已恢复为未完成"
        print(f"#{args.id} {state}: {item['text']}")
    elif args.command == "rm":
        item = todos.remove(args.id)
        if item is None:
            print(f"错误: 找不到编号 {args.id}", file=sys.stderr)
            sys.exit(1)
        print(f"已删除 #{args.id}: {item['text']}")
    elif args.command == "clear":
        n = todos.clear_done() if args.done else todos.clear_all()
        print(f"已清空 {n} 条任务")
    elif args.command == "stats":
        total, done, pending = todos.stats()
        print(f"总共: {total}  已完成: {done}  未完成: {pending}")


if __name__ == "__main__":
    main()
